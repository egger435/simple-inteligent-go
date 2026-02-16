import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from sgfmill import boards, ascii_boards
from common import *
import strategy.go_strategy as go_sg
import random as rd
import torch

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
    
    

    # test_sgf_path = "D:\\02_EdgeDownload\\varied_models_commentary_sgfs\\varied_models_all\save-0.bin\\2011-05-03s.sgf"
    # with open(test_sgf_path, 'rb') as f:
    #     sgf_content = f.read()
    # value_predictor = GoValuePredictor()
    # board, komi, winner = get_final_board_from_sgf(sgf_content)
    # print(ascii_boards.render_board(board))
    # value = value_predictor.predict_value(board, komi)
    

    # print(value, winner)
    # print(ascii_boards.render_board(board))