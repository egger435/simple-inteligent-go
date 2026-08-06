'''自对弈引擎。

用当前模型与自身对弈，生成 (局面, 落子, 奖励) 训练数据。
'''

import numpy as np
import torch
from sgfmill import boards

from common import BOARD_SIZE, PASS_LABEL, COLOR_MAP, GAME_KOMI, idx_to_go_str
from strategy.go_strategy import GoStrategySelector
from value.go_value import GoValuePredictor
from engine.katago import get_katago_engine


class SelfPlayEngine:
    '''自对弈引擎。

    用同一策略网络的两个副本互相博弈，通过温度采样保证棋局多样性。
    价值网络充当"裁判"，评估每步棋的局面质量变化。

    用法::

        engine = SelfPlayEngine(temperature_early=1.0, temperature_late=0.5)
        records, winner = engine.play_one_game()
        # records: [(board, move_idx, reward), ...]
    '''

    def __init__(self, temperature_early: float = 1.0,
                 temperature_late: float = 0.5,
                 temp_switch_step: int = 10,
                 max_steps: int = 400):
        self.tau_early = temperature_early
        self.tau_late = temperature_late
        self.temp_switch = temp_switch_step
        self.max_steps = max_steps

        # 模型（懒加载，首次使用时初始化）
        self._sg_selector: GoStrategySelector | None = None
        self._va_predictor: GoValuePredictor | None = None

    # ------------------------------------------------------------------
    def _init_models(self):
        if self._sg_selector is None:
            self._sg_selector = GoStrategySelector()
        if self._va_predictor is None:
            self._va_predictor = GoValuePredictor()

    # ------------------------------------------------------------------
    def play_one_game(self) -> tuple:
        '''进行一局自对弈。

        Returns:
            records: [(board_np, move_idx, reward), ...]
                board_np: (2, 19, 19) 归一化的局面+行棋方通道
                move_idx: 0~361 落子索引
                reward: 策略梯度奖励信号
            winner: 'b' | 'w'
        '''
        self._init_models()
        board = boards.Board(BOARD_SIZE)
        current_color = 'b'
        records = []
        consecutive_pass = 0
        ko_pos = None          # 当前劫争禁着点
        kata_moves = []  # KataGo 格式落子序列 [['B', 'Q16'], ...]

        for step in range(self.max_steps):
            # ---- 温度 ----
            tau = self.tau_early if step < self.temp_switch else self.tau_late

            # ---- 局面评估（落子前） ----
            b_val_before, w_val_before = (
                self._va_predictor.predict_value(board, current_color, GAME_KOMI)
            )

            # ---- 策略网络预测（已自动屏蔽占位，无需再掩码） ----
            full_probs = self._sg_selector.predict_full_probs(board, current_color)

            # ---- 劫争禁着点屏蔽（禁止立即提回） ----
            if ko_pos is not None:
                full_probs = full_probs.copy()
                full_probs[ko_pos[0] * BOARD_SIZE + ko_pos[1]] = 0.0

            # ---- 温度采样 ----
            if full_probs[PASS_LABEL] >= 0.999:
                move_idx = PASS_LABEL  # 无合法位置，只能弃行
            else:
                move_idx = self._sample_with_temperature(full_probs, tau)

            # ---- 落子 ----
            board_before = self._board_to_np(board, current_color)

            if move_idx == PASS_LABEL:
                consecutive_pass += 1
                kata_moves.append([current_color.upper(), 'pass'])
            else:
                consecutive_pass = 0
                row = move_idx // BOARD_SIZE
                col = move_idx % BOARD_SIZE
                try:
                    ko_pos, _ = board.play(row, col, current_color)
                    kata_moves.append([
                        current_color.upper(),
                        idx_to_go_str((row, col), have_i=False),
                    ])
                except ValueError:
                    # 非法着（自杀/已落子）→ 按弃行处理
                    kata_moves.append([current_color.upper(), 'pass'])

            # ---- 局面评估（落子后） ----
            b_val_after, w_val_after = (
                self._va_predictor.predict_value(board, current_color, GAME_KOMI)
            )

            # ---- 计算 Δ（当前落子方视角） ----
            val_before = b_val_before if current_color == 'b' else w_val_before
            val_after = b_val_after if current_color == 'b' else w_val_after
            delta = val_after - val_before

            # 暂存（终局后统一计算奖励）
            records.append((
                board_before, move_idx, delta,
                current_color,  # 等终局后判断胜负
            ))

            # ---- 切换行棋方 ----
            current_color = 'w' if current_color == 'b' else 'b'

            # ---- 终局判定 ----
            if consecutive_pass >= 2:
                break

        # ---- 判定胜负（KataGo，复用已验证的 get_value） ----
        b_winrate = get_katago_engine().get_value('b', kata_moves)
        if b_winrate < 0:
            # KataGo 失败 → 回退到规则数目
            b_count, w_count = self._count_stones(board)
            winner = 'b' if b_count > (w_count + GAME_KOMI) else 'w'
            score_lead = b_count - (w_count + GAME_KOMI)
        else:
            winner = 'b' if b_winrate > 0.5 else 'w'
            score_lead = (b_winrate - 0.5) * 100  # 近似目数差

        # ---- 计算最终奖励 ----
        final_records = []
        for board_np, move_idx, delta, move_color in records:
            # 终局信号：这步棋的玩家赢了吗
            win_signal = 1.0 if move_color == winner else -1.0
            # 综合奖励 = 局面变化 + 0.5 × 终局结果
            reward = delta + 0.5 * win_signal
            final_records.append((board_np, move_idx, reward))

        # 统计盘面子数（仅用于显示）
        b_stones, w_stones = self._count_stones(board)
        return final_records, winner, (b_stones, w_stones, len(records), score_lead)

    # ------------------------------------------------------------------
    def _board_to_np(self, board: boards.Board, cur_color: str) -> np.ndarray:
        '''将棋盘转为 (2, 19, 19) 输入格式。'''
        board_ch = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = board.get(r, c)
                if stone == 'b':
                    board_ch[r, c] = 0.5
                elif stone == 'w':
                    board_ch[r, c] = 1.0

        color_code = 1 if cur_color == 'b' else 2
        player_ch = np.full((BOARD_SIZE, BOARD_SIZE),
                            COLOR_MAP[color_code], dtype=np.float32)

        return np.stack([board_ch, player_ch], axis=0)

    # ------------------------------------------------------------------
    def _sample_with_temperature(self, probs: np.ndarray,
                                  tau: float) -> int:
        '''温度采样。

        Args:
            probs: 362 维概率分布
            tau: 温度，越高越随机，越低越贪心
        '''
        if tau <= 0.01:
            return int(np.argmax(probs))

        # log(p) / tau → softmax
        probs = np.clip(probs, 1e-10, None)
        log_probs = np.log(probs)
        scaled = log_probs / tau
        scaled -= scaled.max()  # 数值稳定
        exp_probs = np.exp(scaled)
        exp_probs /= exp_probs.sum()

        return int(np.random.choice(len(probs), p=exp_probs))

    # ------------------------------------------------------------------
    def _count_stones(self, board: boards.Board) -> tuple:
        '''中国规则数目：子数 + 围空。

        对每个空格，用 flood fill 确定被哪种颜色包围。
        双方 pass 后所有盘上子视为活子，无需判断死子。
        '''
        from collections import deque

        size = BOARD_SIZE
        visited = set()
        b_area, w_area = 0, 0

        # 统计子数
        for r in range(size):
            for c in range(size):
                s = board.get(r, c)
                if s == 'b':
                    b_area += 1
                elif s == 'w':
                    w_area += 1

        # 对空格 flood fill 判归属
        for r in range(size):
            for c in range(size):
                if board.get(r, c) is not None or (r, c) in visited:
                    continue

                # flood fill
                q = deque([(r, c)])
                visited.add((r, c))
                region = [(r, c)]
                borders = set()

                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < size and 0 <= nc < size):
                            continue
                        stone = board.get(nr, nc)
                        if stone is None:
                            if (nr, nc) not in visited:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                                region.append((nr, nc))
                        else:
                            borders.add(stone)

                # 判归属
                if borders == {'b'}:
                    b_area += len(region)
                elif borders == {'w'}:
                    w_area += len(region)
                # 双方交界或无边界的空格 → 不算任何一方

        return b_area, w_area
