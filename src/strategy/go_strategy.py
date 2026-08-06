'''落子策略选择。

通过策略网络模型从当前局面预测 Top-K 合法候选落子及其概率。
'''

import numpy as np
import torch
from sgfmill import boards

from common import (
    BOARD_SIZE, PASS_LABEL, COLOR_MAP, TOP_K,
    get_strategy_model,
)


class GoStrategySelector:
    '''落子策略选择器。

    使用懒加载的策略网络模型，在首次实例化时才加载权重，
    因此必须在 common.update_config_from_args() 之后创建。
    '''

    def __init__(self):
        self.model = get_strategy_model()  # 懒加载，自动使用当前 device
        self.model.eval()

    # ------------------------------------------------------------------
    def _preprocess_input(self, cur_board: boards.Board, cur_color: str):
        '''将当前棋盘状态和行棋方信息转换为模型输入张量。'''
        # 棋盘矩阵
        board_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = cur_board.get(r, c)
                if stone == 'b':
                    board_matrix[r, c] = 1.0
                elif stone == 'w':
                    board_matrix[r, c] = 2.0
        board_matrix = board_matrix / 2.0  # 归一化

        # 行棋方通道
        color_code = 1 if cur_color == 'b' else 2
        player_channel = np.full(
            (BOARD_SIZE, BOARD_SIZE), COLOR_MAP[color_code], dtype=np.float32,
        )

        model_input = np.stack([board_matrix, player_channel], axis=0)
        model_input = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0)
        return model_input

    # ------------------------------------------------------------------
    def _preprocess_batch(self, boards: list, colors: list):
        '''批量预处理 → (N, 2, 19, 19) 张量，用 list_occupied_points 加速。'''
        n = len(boards)
        batch = np.zeros((n, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for i, (board, color) in enumerate(zip(boards, colors)):
            # Channel 0: 棋子
            occupied, _ = board.list_occupied_points()
            for stone, (r, c) in occupied:
                batch[i, 0, r, c] = 1.0 if stone == 'b' else 2.0
            batch[i, 0] /= 2.0
            # Channel 1: 行棋方
            color_code = 1 if color == 'b' else 2
            batch[i, 1] = COLOR_MAP[color_code]
        return torch.tensor(batch, dtype=torch.float32)

    # ------------------------------------------------------------------
    def predict_full_batch(self, boards: list, colors: list):
        """批量返回完整 362 维概率分布 (B, 362)。"""
        batch = self._preprocess_batch(boards, colors)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            output = self.model(batch.to(device))
            probs = torch.softmax(output, dim=1).cpu().numpy()
        # 屏蔽占位
        for i, (board, color) in enumerate(zip(boards, colors)):
            occupied, _ = board.list_occupied_points()
            for _, (r, c) in occupied:
                probs[i, r * BOARD_SIZE + c] = 0.0
            p_sum = probs[i].sum()
            if p_sum > 0:
                probs[i] /= p_sum
        return probs  # (B, 362)

    def predict(self, cur_board: boards.Board, cur_color: str, k: int = None):
        '''根据当前局面返回 Top-K 合法候选落子 [(pos, prob), ...]。

        Args:
            k: 返回候选数量，默认 TOP_K。
               给 GUI 展示用较大值（如 TOP_K*3），渲染层过滤后仍有足够数量。

        复用 predict_full_probs（已自动屏蔽占位）。
        '''
        if k is None:
            k = TOP_K
        probs = self.predict_full_probs(cur_board, cur_color)
        sorted_indices = np.argsort(probs)[::-1]

        candidates = []
        for idx in sorted_indices:
            if idx == PASS_LABEL:
                continue
            r = idx // BOARD_SIZE
            c = idx % BOARD_SIZE
            if probs[idx] <= 0:
                continue
            if cur_board.get(r, c) is not None:
                continue
            candidates.append(((r, c), probs[idx]))
            if len(candidates) >= k:
                break

        return candidates

    # ------------------------------------------------------------------
    def predict_full_probs(self, cur_board: boards.Board, cur_color: str):
        '''返回完整 362 维落子概率分布（含 pass）。

        自动屏蔽棋盘上已有棋子的位置（概率置 0），调用方无需额外处理。
        '''
        input_tensor = self._preprocess_input(cur_board, cur_color)
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.softmax(output, dim=1)

        probs = prob.cpu().numpy()[0]  # shape (362,)

        # ---- 自动屏蔽已有棋子的位置 ----
        occupied, _ = cur_board.list_occupied_points()
        for _, (r, c) in occupied:
            probs[r * BOARD_SIZE + c] = 0.0

        # 重新归一化
        p_sum = probs.sum()
        if p_sum > 0:
            probs /= p_sum
        else:
            probs[PASS_LABEL] = 1.0  # 无合法落子 → 只能弃行

        return probs
    # ------------------------------------------------------------------
    def predict_batch(self, boards: list, colors: list):
        """批量预测每块棋盘的贪心落子（用于 MC 推演）。

        Args:
            boards: [Board, ...] sgfmill 棋盘列表
            colors: ['b'|'w', ...] 对应行棋方

        Returns:
            [(row,col)|None, ...] 最优落子坐标，None 为弃行
        """
        batch = self._preprocess_batch(boards, colors)

        device = next(self.model.parameters()).device
        with torch.no_grad():
            output = self.model(batch.to(device))
            probs = torch.softmax(output, dim=1).cpu().numpy()

        moves = []
        for i, (board, color) in enumerate(zip(boards, colors)):
            p = probs[i].copy()
            occupied, _ = board.list_occupied_points()
            for _, (r, c) in occupied:
                p[r * BOARD_SIZE + c] = 0.0
            p[PASS_LABEL] = 0.0

            best_idx = int(np.argmax(p))
            if p[best_idx] <= 0:
                moves.append(None)
            else:
                moves.append((best_idx // BOARD_SIZE, best_idx % BOARD_SIZE))
        return moves

