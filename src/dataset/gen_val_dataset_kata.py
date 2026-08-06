'''
使用 KataGo 生成价值网络训练数据。

对 SGF 棋谱中的每一个局面调用 KataGo 分析引擎，生成局面价值标签。
'''

import json
import os
import sys
import time

import numpy as np
from sgfmill import sgf

from config import config
from engine.katago import KataGoEngine

# ---- 数据集生成专用常量 ----
BOARD_SIZE = 19
INPUT_CHUNK_DIR = r'E:\go_dataset\strategy_net'
OUTPUT_CHUNK_DIR = r'E:\go_dataset\value_net'
BATCH_SIZE_KATA = 4
MAX_RETRIES = 3
RESTART_KATAGO_EVERY = 500


# =====================================================================
def check_environment() -> bool:
    '''启动前验证所有路径和文件是否存在。'''
    print('正在进行环境预检查...')

    if not os.path.exists(config.kata_exe_path):
        print(f'错误: 找不到 KataGo 可执行文件: {config.kata_exe_path}')
        return False
    if not os.path.exists(config.kata_model_path):
        print(f'错误: 找不到 KataGo 权重文件: {config.kata_model_path}')
        return False
    if not os.path.exists(config.kata_config_path):
        print(f'错误: 找不到 KataGo 配置文件: {config.kata_config_path}')
        return False

    if not os.path.exists(INPUT_CHUNK_DIR):
        print(f'错误：找不到原始数据目录：{INPUT_CHUNK_DIR}')
        return False
    os.makedirs(OUTPUT_CHUNK_DIR, exist_ok=True)

    print('环境预检查通过')
    return True


# =====================================================================
def idx_to_coord(i: int, j: int) -> str:
    '''将数组索引 (i, j) 转换为 KataGo 识别的坐标字符串（如 A1）。'''
    col_letters = 'ABCDEFGHJKLMNOPQRST'[:BOARD_SIZE]  # 跳过 I
    return f'{col_letters[j]}{i + 1}'


# =====================================================================
class KataGoAnalyzer:
    '''使用共享 KataGoEngine 对 SGF 棋谱进行批量分析。'''

    def __init__(self):
        self.engine = KataGoEngine()

    def close(self):
        self.engine.close()

    def analyze_single_sgf_file(self, sgf_file_path: str):
        '''对一个 SGF 文件的每一步进行局面价值分析。'''
        results = []

        with open(sgf_file_path, 'rb') as f:
            game = sgf.Sgf_game.from_bytes(f.read())

        print(game.get_winner())
        main_sequence = list(game.get_main_sequence())

        moves = []
        for idx, step in enumerate(main_sequence[1:], 1):
            player, pos = step.get_move()
            if player is None:
                move_kata = []
            else:
                player = player.upper()
                pos_kata = idx_to_coord(pos[0], pos[1])
                move_kata = [player, pos_kata]

            moves.append(move_kata)
            next_player, winrate = self.engine._get_value(moves)  # noqa
            print(next_player, winrate)


# =====================================================================
if __name__ == '__main__':
    engine = KataGoEngine()
    engine.analyze_single_sgf_file(
        r'data\gogod_commentary_sgfs\gogod_commentary\varied_selfplay_commentary\59\1.sgf'
    )
