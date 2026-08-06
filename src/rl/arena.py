'''模型对战评估。

用裸策略网络（与自对弈一致）快速评估新模型相对强度。
'''

import numpy as np
from sgfmill import boards

from common import BOARD_SIZE, GAME_KOMI, PASS_LABEL, _set_strategy_model
from strategy.go_strategy import GoStrategySelector


class Arena:
    '''模型对战评估器。

    用裸策略网络对弈（无搜索），快速评估两模型相对强度。
    与自对弈引擎使用相同的评估方式：只测策略网络决策质量。

    用法::

        arena = Arena(new_model, old_model)
        winrate = arena.evaluate(n_games=100)
    '''

    def __init__(self, new_strategy_model, old_strategy_model):
        self.new_model = new_strategy_model
        self.old_model = old_strategy_model
        self._sg: GoStrategySelector | None = None

    # ------------------------------------------------------------------
    def evaluate(self, n_games: int = 100) -> float:
        '''进行 N 局对战，返回新模型胜率。'''
        self._sg = GoStrategySelector()
        wins = 0

        for i in range(n_games):
            new_is_black = (i % 2 == 0)
            winner = self._play_one_game(new_is_black)
            if winner == 'new':
                wins += 1

            if (i + 1) % 10 == 0:
                print(f'  Arena: {i+1}/{n_games} | '
                      f'新模型胜率: {wins/(i+1):.1%}')

        winrate = wins / n_games
        print(f'Arena 结果: {n_games} 局 | 新模型胜率 {winrate:.1%}')
        return winrate

    # ------------------------------------------------------------------
    def _predict_move(self, board: boards.Board, color: str,
                      use_new: bool, ko_pos=None) -> int:
        '''用指定模型预测一步棋，返回落子索引 (0~361)。

        贪心选择最高概率合法着（Arena 不需要温度，要测纯粹强度）。
        '''
        model = self.new_model if use_new else self.old_model
        _set_strategy_model(model)
        self._sg.model = model
        self._sg.model.eval()         # 训练后 model 可能在 train 模式

        full_probs = self._sg.predict_full_probs(board, color)

        # 劫争禁着点屏蔽（禁止立即提回）
        if ko_pos is not None:
            full_probs = full_probs.copy()
            full_probs[ko_pos[0] * BOARD_SIZE + ko_pos[1]] = 0.0

        # 降低 pass 概率（Arena 应尽量不走弃行）
        full_probs[PASS_LABEL] *= 0.01

        # 贪心选择
        move_idx = int(np.argmax(full_probs))
        return move_idx

    # ------------------------------------------------------------------
    def _play_one_game(self, new_is_black: bool) -> str:
        '''一局对战，返回 'new' 或 'old'。'''
        try:
            board = boards.Board(BOARD_SIZE)
            current_color = 'b'
            consecutive_pass = 0
            ko_pos = None                # 当前劫争禁着点
            max_steps = 400

            for _ in range(max_steps):
                use_new = (
                    (current_color == 'b' and new_is_black) or
                    (current_color == 'w' and not new_is_black)
                )

                try:
                    move_idx = self._predict_move(
                        board, current_color, use_new, ko_pos=ko_pos,
                    )
                except Exception:
                    return self._loser(current_color, new_is_black)

                if move_idx == PASS_LABEL:
                    consecutive_pass += 1
                    if consecutive_pass >= 2:
                        break
                else:
                    consecutive_pass = 0
                    row = move_idx // BOARD_SIZE
                    col = move_idx % BOARD_SIZE
                    try:
                        ko_pos, _ = board.play(row, col, current_color)
                    except ValueError:
                        # 真正非法（自杀）→ 弃行
                        pass

                current_color = 'w' if current_color == 'b' else 'b'

            # 中国规则数目
            b_area, w_area = self._area_score(board)
            b_wins = b_area > (w_area + GAME_KOMI)

            if new_is_black:
                return 'new' if b_wins else 'old'
            else:
                return 'old' if b_wins else 'new'

        except Exception as e:
            print(f'Arena 异常: {e}')
            return 'old'

    # ------------------------------------------------------------------
    def _loser(self, color: str, new_is_black: bool) -> str:
        if color == 'b':
            return 'old' if new_is_black else 'new'
        else:
            return 'new' if new_is_black else 'old'

    # ------------------------------------------------------------------
    def _area_score(self, board: boards.Board) -> tuple:
        '''中国规则数目：子 + 围空。'''
        from collections import deque

        size = BOARD_SIZE
        visited = set()
        b_area, w_area = 0, 0

        for r in range(size):
            for c in range(size):
                s = board.get(r, c)
                if s == 'b':
                    b_area += 1
                elif s == 'w':
                    w_area += 1

        for r in range(size):
            for c in range(size):
                if board.get(r, c) is not None or (r, c) in visited:
                    continue

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

                if borders == {'b'}:
                    b_area += len(region)
                elif borders == {'w'}:
                    w_area += len(region)

        return b_area, w_area
