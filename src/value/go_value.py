'''局面价值判断。

提供基于自训练价值网络的终局预测、推演评估，以及 KataGo 引擎的接口。
'''

import os
import random as rd

import numpy as np
import torch
from sgfmill import boards

from common import (
    BOARD_SIZE,
    get_value_model,
)
from engine.katago import KataGoEngine  # noqa: F401 — 向后兼容
import strategy.go_strategy as go_sg


# =====================================================================
# GoValuePredictor
# =====================================================================

class GoValuePredictor:
    '''终局价值预测器。

    使用懒加载的价值网络模型，在首次实例化时才加载权重。
    '''

    def __init__(self):
        self.model = get_value_model()  # 懒加载
        self.model.eval()
        self.go_sg_selector = go_sg.GoStrategySelector()

    # ------------------------------------------------------------------
    def _preprocess_input(self, board: boards.Board, cur_color: str, komi: float):
        '''棋盘 → (2, 19, 19) 模型输入。

        Channel 0: 棋子（空=0, 黑=0.5, 白=1.0）
        Channel 1: 行棋方（黑=0.0, 白=1.0）
        '''
        board_ch = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = board.get(r, c)
                if stone == 'b':
                    board_ch[r, c] = 0.5
                elif stone == 'w':
                    board_ch[r, c] = 1.0

        color_val = 0.0 if cur_color == 'b' else 1.0
        color_ch = np.full((BOARD_SIZE, BOARD_SIZE), color_val, dtype=np.float32)

        model_input = np.stack([board_ch, color_ch], axis=0)
        model_input = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0)
        return model_input

    # ------------------------------------------------------------------
    def _preprocess_batch(self, boards: list, colors: list, komi: float):
        '''批量预处理：多个局面 → (B, 2, 19, 19) 张量。用 list_occupied_points 加速。'''
        n = len(boards)
        batch = np.zeros((n, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for i, (board, color) in enumerate(zip(boards, colors)):
            occupied, _ = board.list_occupied_points()
            for stone, (r, c) in occupied:
                batch[i, 0, r, c] = 0.5 if stone == 'b' else 1.0
            color_val = 0.0 if color == 'b' else 1.0
            batch[i, 1] = color_val
        return torch.tensor(batch, dtype=torch.float32)

    # ------------------------------------------------------------------
    def predict_value(self, board: boards.Board, cur_color: str, komi: float):
        '''返回 (黑方胜率, 白方胜率)。'''
        input_tensor = self._preprocess_input(board, cur_color, komi)
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = self.model(input_tensor)

        prob = output.cpu().numpy()[0]
        return (prob[0], prob[1])

    # ------------------------------------------------------------------
    def get_rollout_value(self, board: boards.Board, cur_color: str, komi: float,
                          rollout_deepth: int = 100):
        '''从当前局面随机推演到终局，返回预测价值。'''
        for _ in range(rollout_deepth):
            candidates = self.go_sg_selector.predict(board, cur_color)
            candidates = rd.choice(candidates)[0]
            if candidates == 'pass':
                continue
            nr, nc = candidates
            board.play(nr, nc, cur_color)
            cur_color = 'b' if cur_color == 'w' else 'w'
        return self.predict_value(board, cur_color, komi)

    # ------------------------------------------------------------------
    def get_monte_carlo_rollout_value(self, board: boards.Board, cur_color: str,
                                      komi: float, rollout_deepth: int = 100,
                                      times: int = 50):
        '''蒙特卡洛推演，返回 (平均价值, (黑胜率, 白胜率))。'''
        b_win = 0
        b_total_value = w_total_value = 0.0
        for _ in range(times):
            init_board = board.copy()
            b_rollout_value, w_rollout_value = self.get_rollout_value(
                init_board, cur_color, komi, rollout_deepth,
            )
            if b_rollout_value > w_rollout_value:
                b_win += 1
            b_total_value += b_rollout_value
            w_total_value += w_rollout_value
            del init_board

        b_MCR_val = b_total_value / times
        w_MCR_val = w_total_value / times
        b_win_rate = b_win / times
        return (b_MCR_val, w_MCR_val), (b_win_rate, 1 - b_win_rate)


# =====================================================================
# 自测入口
# =====================================================================

if __name__ == '__main__':
    from common import get_final_board_from_sgf

    total_samples = correct_samples = 0
    value_predictor = GoValuePredictor()
    for root, dirs, files in os.walk(
        r'D:\02_EdgeDownload\varied_models_commentary_sgfs\varied_models_all'
    ):
        for file in files:
            if not file.endswith('.sgf'):
                continue
            total_samples += 1
            sgf_path = os.path.join(root, file)
            with open(sgf_path, 'rb') as f:
                sgf_content = f.read()
            board, komi, winner = get_final_board_from_sgf(sgf_content)
            value = value_predictor.predict_value(board, komi)
            if ((value[0] > value[1] and winner == 'b') or
                    (value[0] < value[1] and winner == 'w')):
                correct_samples += 1
            print(value, winner)
            if total_samples == 10000:
                print(f'acc: {(correct_samples / total_samples):.4f}')
                exit()
