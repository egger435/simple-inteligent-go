import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from sgfmill import boards, ascii_boards
from common import *
import strategy.go_strategy as go_sg
import random as rd
import torch
import subprocess
import sys
import time
import json

class GoValuePredictor:
    '''终局价值预测类'''
    def __init__(self):
        self.model = VALUE_MODEL.to(DEVICE)
        self._load_model()
        self.model.eval()
        self.go_sg_selector = go_sg.GoStrategySelector()  # 落子策略选择器
    
    def _load_model(self):
        state_dict = torch.load(VA_SAVE_MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state_dict)

    def _preprocess_input(self, board: boards.Board, komi: float):
        '''将当前棋盘状态和贴目信息转化为模型输入格式'''
        # 生成棋盘矩阵
        board_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = board.get(r, c)
                if stone == 'b':
                    board_matrix[r, c] = 0.5
                elif stone == 'w':
                    board_matrix[r, c] = 1.0
        
        # 生成贴目通道
        komi_channel = np.full((BOARD_SIZE, BOARD_SIZE), komi, dtype=np.float32)

        # 合并得到模型输入
        model_input = np.stack([board_matrix, komi_channel], axis=0)
        model_input = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return model_input
    
    def predict_value(self, board: boards.Board, komi: float):
        '''根据当前终局预测价值'''
        input_tensor = self._preprocess_input(board, komi)

        with torch.no_grad():
            output = self.model(input_tensor)
        
        prob = output.cpu().numpy()[0]
        return (prob[0], prob[1])
    
    def get_rollout_value(self, board: boards.Board, cur_color, komi, rollout_deepth=100):
        '''对当前局面进行推演, 得到推演后的终局局面预测价值, 默认推演100手'''
        for i in range(rollout_deepth):
            candidates = self.go_sg_selector.predict(board, cur_color)
            candidates = rd.choice(candidates)[0]
            if candidates == 'pass':
                continue
            nr, nc = candidates
            board.play(nr, nc, cur_color)
            cur_color = 'b' if cur_color == 'w' else 'w'
        return self.predict_value(board, komi)
    
    def get_monte_carlo_rollout_value(self, board: boards.Board, cur_color, komi, rollout_deepth=100, times=50):
        '''对当前局面进行蒙特卡洛推演, 得到推演后的平均终局局面预测价值和双方胜率, 默认推演100手, 推演次数50次'''
        b_win = 0
        b_total_value = w_total_value = 0.0
        for i in range(times):
            init_board = board.copy()
            b_rollout_value, w_rollout_value = self.get_rollout_value(init_board, cur_color, komi, rollout_deepth)
            if b_rollout_value > w_rollout_value:
                b_win += 1
            b_total_value += b_rollout_value
            w_total_value += w_rollout_value
            del init_board
        b_MCR_val = b_total_value / times
        w_MCR_val = w_total_value / times
        b_win_rate = b_win / times
        return (b_MCR_val, w_MCR_val), (b_win_rate, 1 - b_win_rate)

class KataGoEngine:
    '''katago交互类'''
    def __init__(self):
        self.process = None
        self.request_id = 0
        self._start_engine()

    def _start_engine(self):
        try:
            self.process = subprocess.Popen(
                [
                    KATA_EXE_PATH,
                    'analysis',
                    '-model', KATA_MODEL_PATH,
                    '-config', KATA_CONFIG_PATH
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            ready = False
            for _ in range(300):
                line = self.process.stdout.readline()
                if not line:
                    continue
                if 'ready to begin handling requests' in line:
                    ready = True
                    break
            
            if not ready:
                raise RuntimeError('katago启动超时')
        
        except Exception as e:
            print(f'启动失败: {e}')
            sys.exit(1)
    
    def restart(self):
        self.close()
        self._start_engine()
        print('重启KataGo...')
    
    def close(self):
        if self.process:
            print('关闭KataGo...')
            self.process.terminate()
            self.process.wait()
    
    def get_value(self, root_player, moves):
        '''根据当前落子列表给出下一个行棋方的胜率'''
        self.request_id += 1
        request_id = str(self.request_id)
        request = {
            'id': request_id,
            'boardXSize': BOARD_SIZE,
            'boardYSize': BOARD_SIZE,
            'initialStones': [],
            'moves': moves,
            'rules': 'chinese',
            'komi': GAME_KOMI,
            'visits': 10,
            'includePolicy': False,
            'includeOwnership': False,
            'includeMovesOwnership': False
        }

        # 写入请求
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()

        response = None
        start_read = time.time()
        while time.time() - start_read < 5:  # 5秒超时
            line = self.process.stdout.readline()
            if not line:
                continue
            
            kata_message = line.strip()
            print(f'[KataGo] {kata_message}')
            if ('error' or 'Error' in kata_message) and 'rawStWrError' not in kata_message:
                print(f"KataGo返回错误: {kata_message}")
                self.restart()
                return -2
            try:
                resp = json.loads(line)
                if 'error' in resp:
                    print(f"KataGo返回错误: {resp['error']}")
                    self.restart()
                    return -2
                if 'rootInfo' not in resp:
                    continue
                response = resp
                break
            except json.JSONDecodeError:
                continue
        
        if not response:
            print('KataGo响应超时')
            return -1
        
        # 解析结果
        player = 'b' if len(moves) % 2 == 0 else 'w'  # 下一个行棋方
        winrate = response['rootInfo']['winrate']
        print(f'我方胜率: {100*(1-winrate):.2f}%\n')

        return winrate if root_player == player else (1-winrate)

if __name__ == '__main__':
    total_samples = correct_samples = 0
    value_predictor = GoValuePredictor()
    for root, dirs, files in os.walk("D:\\02_EdgeDownload\\varied_models_commentary_sgfs\\varied_models_all"):
        for file in files:
            if not file.endswith('.sgf'):
                continue
            total_samples += 1
            sgf_path = os.path.join(root, file)
            with open(sgf_path, 'rb') as f:
                sgf_content = f.read()
            board, komi, winner = get_final_board_from_sgf(sgf_content)
            value = value_predictor.predict_value(board, komi)
            if (value[0] > value[1] and winner == 'b') or (value[0] < value[1] and winner == 'w'):
                correct_samples += 1
            print(value, winner)
            if total_samples == 10000:
                print(f'acc: {(correct_samples / total_samples):.4f}')
                exit()