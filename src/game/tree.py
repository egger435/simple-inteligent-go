'''批量 MiniMax 搜索树 + 蒙特卡洛推演算法实现。'''

import time

import torch
import numpy as np
from sgfmill import boards

from common import (
    TOP_K, GAME_KOMI, USE_OWN_VALUE_NET, idx_to_go_str,
)
from strategy.go_strategy import GoStrategySelector
from value.go_value import GoValuePredictor
from engine.katago import get_katago_engine


# =====================================================================

class BatchMinimaxMCR:
    '''批量 MiniMax 搜索：每层一次 batch 策略网络调用，全部 GPU 并行。'''

    def __init__(self, steps_list, root_state, root_player, curstep,
                 top_k=4, max_depth=7, ko_pos=None,
                 use_own_value_net=None, n_rollouts=None, n_steps=None,
                 verbose=True):
        from common import MC_ROLLOUTS, MC_STEPS, USE_OWN_VALUE_NET
        self.sg_selector = GoStrategySelector()
        self.va_predictor = GoValuePredictor()
        # 实例级参数：默认读全局配置，可覆盖
        self.use_own_value_net = (
            USE_OWN_VALUE_NET if use_own_value_net is None else use_own_value_net
        )
        self.n_rollouts = MC_ROLLOUTS if n_rollouts is None else n_rollouts
        self.n_steps = MC_STEPS if n_steps is None else n_steps
        self.verbose = verbose
        self.root_player = root_player
        self.steps_list = steps_list
        self.top_k = top_k
        self.max_depth = max_depth
        self.root_state = root_state.copy()
        self.root = None
        self.leaves = []

    # ------------------------------------------------------------------
    def build_tree(self):
        '''批量构建：每层一次 predict_full_batch，BFS 全并行。'''
        if self.verbose:
            print(f'[批量树] 构建搜索树 (TOP_K={self.top_k}, depth={self.max_depth})...')
        t0 = time.time()

        # 根节点
        root_board = boards.Board(19)
        root_moves = []
        for color, pos_str in self.steps_list:
            if pos_str == 'pass':
                root_moves.append([color, 'pass'])
                continue
            from common import go_str_to_idx
            r, c = go_str_to_idx(pos_str, have_i=False)
            root_board.play(r, c, color.lower())
            root_moves.append([color, idx_to_go_str((r, c), have_i=False)])
        self.root = _BNode(root_board, None, self.root_player, 0, 0.0)
        self.root.moves = root_moves
        level_nodes = [self.root]
        total_nodes = 1

        for d in range(self.max_depth):
            boards_list = [n.board for n in level_nodes]
            colors_list = [n.color for n in level_nodes]
            n_batch = len(boards_list)

            # 一次批量策略网络调用 → 完整概率
            t1 = time.time()
            all_probs = self.sg_selector.predict_full_batch(
                boards_list, colors_list,
            )  # (N, 362)

            next_level = []
            for node, probs in zip(level_nodes, all_probs):
                order = np.argsort(probs)[::-1]
                created = 0
                for idx in order:
                    if idx == 361:
                        continue
                    if probs[idx] <= 0:
                        continue
                    r, c = idx // 19, idx % 19
                    try:
                        child_board = node.board.copy()
                        child_board.play(r, c, node.color)
                    except ValueError:
                        continue
                    child_color = 'w' if node.color == 'b' else 'b'
                    child = _BNode(child_board, node, child_color, d + 1, 0.0)
                    child.move = (r, c)
                    child.moves = node.moves + [
                        [node.color.upper(), idx_to_go_str((r, c), have_i=False)]
                    ]
                    node.children.append(child)
                    next_level.append(child)
                    created += 1
                    if created >= self.top_k:
                        break

            level_nodes = next_level
            total_nodes += len(next_level)
            t2 = time.time()
            if self.verbose:
                print(f'  深度 {d+1}/{self.max_depth}: {n_batch} 节点 → {len(next_level)} 子节点 | {t2-t1:.1f}s')

            if not level_nodes:
                break

        self.leaves = level_nodes
        if self.verbose:
            print(f'[批量树] 完成: {total_nodes} 节点, {len(self.leaves)} 叶节点 | 总耗时 {time.time()-t0:.1f}s')
        return self.root, self.leaves

    # ------------------------------------------------------------------
    def evaluate_all_leaves(self, n_rollouts=5, n_steps=5):
        '''评估所有叶节点。

        use_own_value_net=true  → 批量 MC 推演 + 自训练价值网络
        use_own_value_net=false → KataGo 流水线批量评估
        '''
        if not self.use_own_value_net:
            self._evaluate_kata()
            return

        device = next(self.va_predictor.model.parameters()).device
        idx = 0 if self.root_player == 'b' else 1
        n_leaves = len(self.leaves)
        t0 = time.time()

        if self.verbose:
            print(f'[MC推演] {n_leaves} 叶 × {n_rollouts} 推演 × {n_steps} 步...')

        # 初始化推演棋盘
        boards = []
        colors = []
        for leaf in self.leaves:
            for _ in range(n_rollouts):
                boards.append(leaf.board.copy())
                colors.append(leaf.color)

        # 逐步推演
        for step in range(n_steps):
            moves = self.sg_selector.predict_batch(boards, colors)
            for j, move in enumerate(moves):
                if move is None:
                    continue
                try:
                    boards[j].play(move[0], move[1], colors[j])
                except ValueError:
                    pass
                colors[j] = 'w' if colors[j] == 'b' else 'b'

        # 批量价值网络评估
        batch = self.va_predictor._preprocess_batch(boards, colors, GAME_KOMI)
        with torch.no_grad():
            probs = self.va_predictor.model(batch.to(device)).cpu().numpy()

        # 每叶节点取平均
        for j, leaf in enumerate(self.leaves):
            start = j * n_rollouts
            end = start + n_rollouts
            leaf.value = float(probs[start:end, idx].mean())

        if self.verbose:
            print(f'[MC推演] 完成 | 总耗时 {time.time()-t0:.1f}s')

    # ------------------------------------------------------------------
    def _evaluate_kata(self):
        '''使用 KataGo 流水线批量评估叶节点。'''
        va_engine = get_katago_engine()
        n_leaves = len(self.leaves)
        t0 = time.time()
        if self.verbose:
            print(f'[KataGo] 批量评估 {n_leaves} 个叶节点...')

        queries = [(self.root_player, leaf.moves) for leaf in self.leaves]
        values = va_engine.get_value_batch(queries)
        for leaf, v in zip(self.leaves, values):
            leaf.value = v if v >= 0 else 0.5

        if self.verbose:
            print(f'[KataGo] 完成 | 总耗时 {time.time()-t0:.1f}s')

    # ------------------------------------------------------------------
    def minimax_backup(self):
        '''MiniMax 回溯。'''
        def _backup(node):
            if not node.children:
                return node.value
            vals = [_backup(c) for c in node.children]
            if node.color == self.root_player:
                node.value = max(vals)
            else:
                node.value = min(vals)
            return node.value
        _backup(self.root)

    # ------------------------------------------------------------------
    def select_best_move(self):
        if not self.root.children:
            return 'pass', 0.0
        best = max(self.root.children, key=lambda c: c.value)
        return best.move, best.value

    # ------------------------------------------------------------------
    def search(self):
        t_start = time.time()
        self.build_tree()
        self.evaluate_all_leaves(n_rollouts=self.n_rollouts, n_steps=self.n_steps)

        self.minimax_backup()

        if self.verbose:
            print('--- Root 子节点 (按 value 降序) ---')
            for i, c in enumerate(sorted(self.root.children, key=lambda x: x.value or 0, reverse=True)):
                if c.move:
                    print(f'  [{i+1}] {idx_to_go_str(c.move)} value={c.value:.4f}')

        best_move, best_value = self.select_best_move()
        if self.verbose:
            print(f'=== 最终选择: {idx_to_go_str(best_move)} | value={best_value:.4f} ===')
        return best_move, best_value


# =====================================================================
# _BNode —— 批量 MiniMax 树的轻量节点
# =====================================================================

class _BNode:
    '''批量树的轻量节点（无 sgfmill 依赖，仅存必要信息）。'''
    __slots__ = ('board', 'parent', 'color', 'depth', 'value',
                 'children', 'move', 'moves')

    def __init__(self, board, parent, color, depth, value):
        self.board = board          # sgfmill Board
        self.parent = parent        # _BNode or None
        self.color = color          # 'b' | 'w'
        self.depth = depth          # int
        self.value = value          # float
        self.children = []          # [_BNode, ...]
        self.move = None            # (row, col) — 父到本节点的落子
        self.moves = None           # KataGo 落子序列 [['B','Q16'],...]

    @property
    def by_move(self):
        return self.move           # 兼容 TreeNode 接口


# =====================================================================
# 统一搜索调度器
# =====================================================================

def search_move(steps_list, board, color, curstep, ko_pos=None,
                algorithm=None, verbose=True,
                top_k=None, max_depth=None,
                use_own_value_net=None,
                n_rollouts=None, n_steps=None,
                simulations=None, c_puct=None, temperature=None,
                expand_width=None, eval_batch_size=None):
    '''按配置选择搜索算法并落子。

    Args:
        algorithm: 'minimax' | 'mcts'，None 则读 config.search_algorithm
        top_k / max_depth: BatchMinimax 参数（默认读 config）
        use_own_value_net: 评估方式（None 读全局）
        n_rollouts / n_steps: BatchMinimax MC 参数
        simulations / c_puct / temperature: MCTS 参数（默认读 config）

    Returns:
        (best_move, best_value)
    '''
    from common import SEARCH_ALGORITHM
    import json

    try:
        cfg = json.load(open('config.json'))
    except Exception:
        cfg = {}
    if algorithm is None:
        algorithm = cfg.get('search_algorithm', SEARCH_ALGORITHM)

    if algorithm == 'mcts':
        from game.mcts import MCTS
        mcts = MCTS(
            steps_list, board, color, curstep,
            simulations=simulations, c_puct=c_puct, temperature=temperature,
            expand_width=expand_width, eval_batch_size=eval_batch_size,
            use_own_value_net=use_own_value_net,
            verbose=verbose,
        )
        return mcts.search()
    else:
        minimax = BatchMinimaxMCR(
            steps_list, board, color, curstep,
            top_k=top_k if top_k is not None else cfg.get('top_k', 4),
            max_depth=max_depth if max_depth is not None
            else cfg.get('max_search_depth', 5),
            use_own_value_net=use_own_value_net,
            n_rollouts=n_rollouts, n_steps=n_steps,
            verbose=verbose,
        )
        return minimax.search()
