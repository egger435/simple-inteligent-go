import subprocess
import json
import numpy as np
import os
import gc
import sys
import time
from sgfmill import sgf

BOARD_SIZE = 19
PASS_LABEL = 361

KATA_EXE_PATH = r'D:\01_EGGER\program\python\simple-inteligent-go\katago\katago.exe'
KATA_MODEL_PATH = r'D:\01_EGGER\program\python\simple-inteligent-go\katago\kata1-b40c256-s5109387264-d1232289301.bin.gz'
KATA_CONFIG_PATH = r'D:\01_EGGER\program\python\simple-inteligent-go\katago\analysis_config.cfg'

INPUT_CHUNK_DIR = r'E:\go_dataset\strategy_net'
OUTPUT_CHUNK_DIR = r'E:\go_dataset\value_net'

BATCH_SIZE_KATA = 4  # 批量发送给KataGo的局面数
MAX_RETRIES = 3        # 单个样本失败的最大重试次数
RESTART_KATAGO_EVERY = 500  # 每处理N个样本重启一次KataGo

def check_environment() -> bool:
    """
    在启动前验证所有路径和文件是否存在
    """
    print("正在进行环境预检查...")
    
    # 检查KataGo相关文件
    if not os.path.exists(KATA_EXE_PATH):
        print(f"错误: 找不到KataGo可执行文件: {KATA_EXE_PATH}")
        return False
    if not os.path.exists(KATA_MODEL_PATH):
        print(f"错误: 找不到KataGo权重文件: {KATA_MODEL_PATH}")
        return False
    if not os.path.exists(KATA_CONFIG_PATH):
        print(f"错误: 找不到KataGo配置文件: {KATA_CONFIG_PATH}")
        return False
    
    # 检查数据目录
    if not os.path.exists(INPUT_CHUNK_DIR):
        print(f"错误：找不到原始数据目录：{INPUT_CHUNK_DIR}")
        return False
    os.makedirs(OUTPUT_CHUNK_DIR, exist_ok=True)
    
    print("环境预检查通过")
    return True

def idx_to_coord(i: int, j: int) -> str:
    """
    将数组索引 (i, j) 转换为KataGo识别的坐标 A1
    """
    col_letters = "ABCDEFGHJKLMNOPQRST"[:BOARD_SIZE]  # 跳过字母I
    return f"{col_letters[j]}{i+1}"

class KataGoEngine:
    '''katago交互类'''
    def __init__(self):
        self.process = None
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
    
    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

    def _get_value(self, moves):
        '''根据当前的当前棋局每一个局面的落子列表给出每一个局面下一个行棋方的局面价值'''
        request = {
            'id': '',
            'boardXSize': BOARD_SIZE,
            'boardYSize': BOARD_SIZE,
            'initialStones': [],
            'moves': moves,
            'rules': 'chinese',
            'komi': 7.5,
            'visits': 20,
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

            # print(f'[KataGo] {line.strip()}')
            try:
                resp = json.loads(line)
                if 'rootInfo' not in resp:
                    continue
                response = resp
                break
            except json.JSONDecodeError:
                continue
        
        if not response:
            print('KataGo响应超时')
            return -1
        
        if 'error' in response:
            print(f"KataGo返回错误: {response['error']}")
            return -2
        
        # 解析结果
        player = 'b' if len(moves) % 2 == 0 else 'w'  # 下一个行棋方
        winrate = response['rootInfo']['winrate']
        score_lead = response['rootInfo']['scoreLead']

        return player, winrate


    def analyze_single_sgf_file(self, sgf_file_path):
        '''对一个sgf文件进行分析每一步的局面价值'''
        results = []
        requests = []

        with open(sgf_file_path, 'rb') as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        
        print(game.get_winner())
        
        # 得到落子主序列
        main_sequence = list(game.get_main_sequence())

        moves = []
        for idx, step in enumerate(main_sequence[1:], 1):
            # 将sgf读取到的坐标转换为katago识别的落子序列
            player, pos = step.get_move()
            if player is None:  # 第一步为空落子
                move_kata = []
            else:
                player = player.upper()
                pos_kata = idx_to_coord(pos[0], pos[1])
                move_kata = [player, pos_kata]

            moves.append(move_kata)

            next_player, winrate = self._get_value(idx, moves)
            print(next_player, winrate)



if __name__ == '__main__':
    engine = KataGoEngine()
    engine.analyze_single_sgf_file(r'data\gogod_commentary_sgfs\gogod_commentary\varied_selfplay_commentary\59\1.sgf')
            



        

