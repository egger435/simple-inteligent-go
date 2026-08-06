'''蒙特卡洛树搜索（MCTS）实现。

先验 UCB + 价值网络风格（AlphaGo Zero 思路）：
  - UCB 选择:  Q + c * P * sqrt(ln(N_parent) / N_child)
  - 先验 P 来自策略网络 predict_full_probs()
  - 叶节点评估: 价值网络 或 KataGo（由 use_own_value_net 决定）
  - 落子: 访问次数最多的子节点

与 BatchMinimaxMCR 并存，由 config.search_algorithm 切换。
'''

import threading
import time

import torch
import numpy as np
from sgfmill import boards

from common import (
    BOARD_SIZE, GAME_KOMI, PASS_LABEL,
    USE_OWN_VALUE_NET,
    MCTS_SIMULATIONS, MCTS_C_PUCT, MCTS_TEMPERATURE,
    idx_to_go_str,
)
from strategy.go_strategy import GoStrategySelector
from value.go_value import GoValuePredictor
from engine.katago import get_katago_engine


# =====================================================================
class MCTSNode:
    '''MCTS 树节点。'''

    __slots__ = ('board', 'color', 'parent', 'move', 'prior',
                 'visit_count', 'total_value', 'children', 'is_terminal',
                 'expanded', 'ko_point')

    def __init__(self, board, color, parent=None, move=None, prior=0.0,
                 ko_point=None):
        self.board = board          # sgfmill Board
        self.color = color          # 该轮到谁走 'b'/'w'
        self.parent = parent
        self.move = move            # 从父到本节点的落子
        self.prior = prior          # 策略网络先验概率
        self.visit_count = 0
        self.total_value = 0.0      # 累计价值（root_player 视角）
        self.children = []
        self.is_terminal = False
        self.expanded = False
        self.ko_point = ko_point    # 本节点禁着点（simple ko，轮到 color 走时不可落子）

    # ------------------------------------------------------------------
    @property
    def value(self):
        '''平均价值（供 GUI 展示/选着）。'''
        return self.total_value / self.visit_count if self.visit_count else 0.0

    @property
    def by_move(self):
        return self.move           # 兼容接口


# =====================================================================
class MCTS:
    '''蒙特卡洛树搜索。

    用法::

        mcts = MCTS(steps, board, root_player)
        best_move, best_value = mcts.search()
    '''

    def __init__(self, steps_list, root_state, root_player, curstep,
                 simulations=None, c_puct=None, temperature=None,
                 expand_width=None, eval_batch_size=None,
                 use_own_value_net=None, verbose=None,
                 visualize=False, vis_interval=0.5, ko_pos=None):
        from common import (
            MCTS_SIMULATIONS as _SIM, MCTS_C_PUCT as _C,
            MCTS_TEMPERATURE as _T, MCTS_VISUALIZE, MCTS_VIS_INTERVAL,
            MCTS_EXPAND_WIDTH, MCTS_EVAL_BATCH_SIZE, MCTS_VERBOSE,
        )
        self.sg_selector = GoStrategySelector()
        self.va_predictor = GoValuePredictor()
        self.simulations = simulations if simulations is not None else _SIM
        self.c_puct = c_puct if c_puct is not None else _C
        self.temperature = (
            temperature if temperature is not None else _T
        )
        self.expand_width = (
            expand_width if expand_width is not None else MCTS_EXPAND_WIDTH
        )
        self.eval_batch_size = (
            eval_batch_size if eval_batch_size is not None
            else MCTS_EVAL_BATCH_SIZE
        )
        self.use_own_value_net = (
            USE_OWN_VALUE_NET if use_own_value_net is None else use_own_value_net
        )
        self.verbose = verbose if verbose is not None else MCTS_VERBOSE
        # 可视化：步进 + 快照
        self.visualize = visualize if visualize is not None else MCTS_VISUALIZE
        self.vis_interval = (
            vis_interval if vis_interval is not None else MCTS_VIS_INTERVAL
        )
        self._snapshot = None
        self._snapshot_lock = threading.Lock()
        self.on_step = None     # 每步回调（供 GUI 通知）

        self.root_player = root_player
        self.opp_player = 'b' if root_player == 'w' else 'w'

        # 重建根棋盘（回放真实落子，同时追踪劫争禁着点）
        root_board = boards.Board(BOARD_SIZE)
        ko = None
        for color, pos_str in steps_list:
            if pos_str == 'pass':
                continue
            from common import go_str_to_idx
            r, c = go_str_to_idx(pos_str, have_i=False)
            ko, _ = root_board.play(r, c, color.lower())

        self.root = MCTSNode(root_board, root_player,
                             ko_point=ko_pos if ko_pos is not None else ko)
        self._evaluator_device = (
            next(self.va_predictor.model.parameters()).device
        )
        # KataGo 评估缓存：moves 序列 → 胜率（避免重复查询相同局面）
        self._eval_cache: dict = {}

    # ------------------------------------------------------------------
    def _is_terminal(self, node: MCTSNode) -> bool:
        '''简化终局：连续弃行或棋盘满。'''
        if len(node.board.list_occupied_points()[0]) >= BOARD_SIZE ** 2 - 2:
            return True
        return False

    # ------------------------------------------------------------------
    def _evaluate(self, node: MCTSNode) -> float:
        '''评估叶节点，返回 root_player 视角胜率。'''
        if self.use_own_value_net:
            b_wr, w_wr = self.va_predictor.predict_value(
                node.board, node.color, GAME_KOMI,
            )
            return b_wr if self.root_player == 'b' else w_wr
        else:
            # KataGo 需要落子序列
            moves = self._node_moves(node)
            key = tuple(m for m in moves)   # 可哈希缓存键
            if key in self._eval_cache:
                return self._eval_cache[key]

            b_wr = get_katago_engine().get_value(self.root_player, moves)
            val = b_wr if b_wr >= 0 else 0.5
            self._eval_cache[key] = val
            return val

    # ------------------------------------------------------------------
    def _node_moves(self, node: MCTSNode) -> list:
        '''回溯节点到根的落子序列。'''
        moves = []
        cur = node
        path = []
        while cur.parent is not None:
            path.append((cur.parent.color, cur.move))
            cur = cur.parent
        for color, move in reversed(path):
            moves.append([color.upper(), idx_to_go_str(move, have_i=False)])
        return moves

    # ------------------------------------------------------------------
    def _expand(self, node: MCTSNode):
        '''用策略网络先验展开叶节点。'''
        if node.expanded or node.is_terminal:
            return

        probs = self.sg_selector.predict_full_probs(node.board, node.color)

        # 创建子节点（前 top-K 或全部合法着）
        order = np.argsort(probs)[::-1]
        child_count = 0
        for idx in order:
            if idx == PASS_LABEL:
                continue
            if probs[idx] <= 0:
                continue
            r, c = idx // BOARD_SIZE, idx % BOARD_SIZE
            # 劫争：禁止立即提回（否则会无限循环提劫）
            if node.ko_point is not None and (r, c) == node.ko_point:
                continue
            try:
                child_board = node.board.copy()
                ko_point, _ = child_board.play(r, c, node.color)
            except ValueError:
                continue
            child_color = 'w' if node.color == 'b' else 'b'
            child = MCTSNode(
                child_board, child_color,
                parent=node, move=(r, c), prior=probs[idx],
                ko_point=ko_point,
            )
            node.children.append(child)
            child_count += 1
            if child_count >= self.expand_width:   # 限制展开宽度（可配置）
                break

        node.expanded = True
        if not node.children:
            node.is_terminal = True

    # ------------------------------------------------------------------
    def _best_child(self, node: MCTSNode) -> MCTSNode:
        '''UCB 选择子节点。

        未访问子节点优先（UCB=+inf），保证所有子节点先被探索一遍，
        避免高先验节点垄断模拟。
        '''
        best = None
        best_score = -float('inf')
        log_n = np.log(node.visit_count + 1)

        for child in node.children:
            if child.visit_count == 0:
                # 未访问 → 绝对优先（先探索）
                score = 1e9 + child.prior
            else:
                q = child.total_value / child.visit_count
                score = q + self.c_puct * child.prior * np.sqrt(
                    log_n / child.visit_count,
                )
            if score > best_score:
                best_score = score
                best = child
        return best

    # ------------------------------------------------------------------
    def _select_leaf(self):
        '''选择到叶节点，路径节点虚拟访问（visit_count+1 防重复选择）。

        Returns:
            (node, path)：未展开叶节点 + 从根到它的路径
        '''
        node = self.root
        path = [node]
        while node.expanded and node.children and not node.is_terminal:
            node = self._best_child(node)
            path.append(node)
        # 虚拟访问
        for n in path:
            n.visit_count += 1
        return node, path

    # ------------------------------------------------------------------
    def _backprop(self, path: list, value: float):
        '''回溯（visit_count 已在选择时 +1，这里只加 total_value）。'''
        for n in reversed(path):
            n.total_value += value
            value = 1.0 - value   # 对手视角翻转

    # ------------------------------------------------------------------
    def _terminal_value(self, node: MCTSNode) -> float:
        '''终局价值：盘面数字粗判。'''
        b_count = sum(1 for s, _ in node.board.list_occupied_points() if s == 'b')
        w_count = sum(1 for s, _ in node.board.list_occupied_points() if s == 'w')
        value = 1.0 if (b_count + GAME_KOMI) > w_count else 0.0
        if self.root_player == 'w':
            value = 1.0 - value
        return value

    # ------------------------------------------------------------------
    def _select_best(self):
        '''按访问次数选最优子节点。'''
        if not self.root.children:
            return 'pass', 0.0

        if self.temperature <= 0.01:
            # 贪心：访问次数最多
            best = max(self.root.children, key=lambda c: c.visit_count)
            return best.move, best.value
        else:
            # 温度采样
            visits = np.array([c.visit_count for c in self.root.children],
                              dtype=np.float32)
            probs = visits ** (1.0 / self.temperature)
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.root.children), p=probs)
            child = self.root.children[idx]
            return child.move, child.value

    # ------------------------------------------------------------------
    def _update_snapshot(self):
        '''生成当前搜索状态的轻量快照（线程安全）。'''
        # 根节点子节点分布
        root_visits = []
        for c in self.root.children:
            root_visits.append((
                c.move,
                c.visit_count,
                c.value,
            ))

        # 树结构（depth ≤ 3, visits ≥ 1）
        tree_nodes = []
        counter = [0]

        def _collect(node, depth):
            nid = counter[0]
            counter[0] += 1
            node_rec = {
                'id': nid,
                'depth': depth,
                'move': node.move,
                'visits': node.visit_count,
                'value': node.value,
                'children': [],
            }
            tree_nodes.append(node_rec)
            if depth < 3:
                for child in node.children:
                    if child.visit_count >= 1:
                        child_id = _collect(child, depth + 1)
                        node_rec['children'].append(child_id)
            return nid

        _collect(self.root, 0)

        snapshot = {
            'root_visits': root_visits,
            'tree': tree_nodes,
            'total_sims': self._sim_done,
        }
        with self._snapshot_lock:
            self._snapshot = snapshot

        if self.on_step is not None:
            self.on_step(snapshot)

    # ------------------------------------------------------------------
    def get_snapshot(self):
        '''线程安全读取快照。'''
        with self._snapshot_lock:
            return self._snapshot

    # ------------------------------------------------------------------
    def _evaluate_batch(self, pending: list):
        '''批量评估待评估的叶节点并回溯。

        pending: [(node, path), ...]
        '''
        nodes = [n for n, _ in pending]
        paths = [p for _, p in pending]

        if self.use_own_value_net:
            # 价值网络：单次评估（快）
            values = [self._evaluate(n) for n in nodes]
        else:
            # KataGo：批量流水线一次查询
            queries = [(self.root_player, self._node_moves(n)) for n in nodes]
            vs = get_katago_engine().get_value_batch(queries)
            values = [v if v >= 0 else 0.5 for v in vs]

        for path, val in zip(paths, values):
            self._backprop(path, val)

    # ------------------------------------------------------------------
    def search(self):
        '''执行 MCTS 搜索（叶节点并行批量评估）。'''
        t0 = time.time()
        self._sim_done = 0
        pending = []
        batch = max(1, self.eval_batch_size)

        for i in range(self.simulations):
            node, path = self._select_leaf()

            if self.verbose:
                # 打印当前选中路径（建树过程）
                path_str = ' → '.join(
                    '根' if n.move is None else idx_to_go_str(n.move)
                    for n in path
                )
                print(f'  [MCTS] 模拟{i+1} 选中路径: {path_str}')

            if node.is_terminal:
                # 终局 → 立即回溯
                val = self._terminal_value(node)
                if self.verbose:
                    print(f'  [MCTS] 模拟{i+1} 终局: '
                          f'深度{len(path)} value={val:.3f}')
                self._backprop(path, val)
            else:
                self._expand(node)
                if node.children:
                    if self.verbose:
                        child_str = ', '.join(
                            idx_to_go_str(c.move) for c in node.children[:8]
                        )
                        print(f'  [MCTS] 模拟{i+1} 扩展节点 '
                              f'{"(根)" if node.move is None else idx_to_go_str(node.move)} '
                              f'→ {len(node.children)} 子: [{child_str}]')
                    pending.append((node, path))
                else:
                    node.is_terminal = True
                    val = self._terminal_value(node)
                    if self.verbose:
                        print(f'  [MCTS] 模拟{i+1} 无子→终局 value={val:.3f}')
                    self._backprop(path, val)

            self._sim_done += 1

            # 批量评估：积累够一批 或 最后一轮
            if len(pending) >= batch or i == self.simulations - 1:
                if pending:
                    if self.verbose:
                        print(f'  [MCTS] 批量评估 {len(pending)} 个叶节点'
                              f'（{"KataGo" if not self.use_own_value_net else "价值网络"}）')
                    self._evaluate_batch(pending)
                    pending = []

            if self.visualize:
                self._update_snapshot()
                time.sleep(self.vis_interval)

        best_move, best_value = self._select_best()

        if self.verbose:
            print(f'[MCTS] {self.simulations} 模拟完成 | 总耗时 '
                  f'{time.time()-t0:.1f}s | 根节点 {len(self.root.children)} 子')
            for c in sorted(self.root.children, key=lambda x: x.visit_count,
                            reverse=True)[:5]:
                print(f'  {idx_to_go_str(c.move)} visits={c.visit_count} '
                      f'value={c.value:.4f}')

        return best_move, best_value
