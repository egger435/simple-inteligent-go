'''概率热力图叠加层。

在棋盘上以半透明彩色圆点标注 AI 候选落子及概率分布。
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

from common import BOARD_SIZE, TOP_K, idx_to_go_str


# 蓝→绿→红 colormap（低概率冷色，高概率暖色）
_PROB_CMAP = LinearSegmentedColormap.from_list(
    'prob_cmap', ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'],
)


class ProbOverlay:
    '''概率热力图叠加层。

    在 BoardRenderer 的 Axes 上叠加半透明候选落子可视化。

    用法::

        overlay = ProbOverlay(ax)
        overlay.show(topk_moves)       # 显示 Top-K 候选落子
        overlay.hide()                 # 清除叠加层
    '''

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self._artists: list = []

    # ------------------------------------------------------------------
    def show(self, topk_moves: list, highlight_move: tuple = None, board=None):
        '''显示 Top-K 候选落子。

        Args:
            topk_moves: [((row, col), prob), ...] 候选落子及概率
            highlight_move: 高亮标记的最佳落子 (row, col)
            board: sgfmill Board，用于二次确认位置无子
        '''
        self.hide()

        if not topk_moves:
            return

        probs = np.array([p for _, p in topk_moves])
        max_p, min_p = probs.max(), probs.min()
        p_range = max_p - min_p if max_p > min_p else 1.0

        shown = 0
        for (r, c), prob in topk_moves:
            # 渲染层兜底：已有棋子的位置直接跳过，取下一个
            if board is not None and board.get(r, c) is not None:
                continue
            shown += 1
            if shown > TOP_K:
                break
            # 概率 → 颜色映射
            norm_p = (prob - min_p) / p_range
            color = _PROB_CMAP(norm_p)

            # 概率 → 半径映射 (0.15 ~ 0.40)
            radius = 0.15 + norm_p * 0.25

            circle = Circle(
                (c, r), radius=radius, facecolor=color,
                edgecolor='black', linewidth=0.5,
                alpha=0.55, zorder=4,
            )
            self.ax.add_patch(circle)
            self._artists.append(circle)

            # 概率文字标注
            text = self.ax.text(
                c, r, f'{prob:.1%}',
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='black', zorder=5,
            )
            self._artists.append(text)

        # 最优落子高亮
        if highlight_move:
            r, c = highlight_move
            marker = Circle(
                (c, r), radius=0.44, facecolor='none',
                edgecolor='gold', linewidth=2.5,
                zorder=3,
            )
            self.ax.add_patch(marker)
            self._artists.append(marker)

    # ------------------------------------------------------------------
    def hide(self):
        '''清除所有叠加元素。'''
        for artist in self._artists:
            artist.remove()
        self._artists.clear()


class ValueOverlay:
    '''MiniMax 叶节点评估值叠加层。

    在棋盘上以数值标注各候选落子的 MiniMax 回溯价值。

    用法::

        v_overlay = ValueOverlay(ax)
        v_overlay.show(children_values)   # 显示子节点评估值
        v_overlay.hide()
    '''

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self._artists: list = []

    # ------------------------------------------------------------------
    def show(self, children: list):
        '''显示根节点子节点的 MiniMax 评估值。

        Args:
            children: [TreeNode, ...] 根节点的子节点
        '''
        self.hide()

        values = [c.value for c in children if c.value is not None]
        if not values:
            return
        max_v, min_v = max(values), min(values)
        v_range = max_v - min_v if max_v > min_v else 1.0

        for child in children:
            if child.value is None or child.by_move is None:
                continue
            r, c = child.by_move
            norm_v = (child.value - min_v) / v_range

            # 值越高越绿，越低越红
            color = (1 - norm_v, norm_v, 0, 0.25)
            rect = plt.Rectangle(
                (c - 0.4, r - 0.4), 0.8, 0.8,
                facecolor=color, edgecolor='none',
                zorder=3,
            )
            self.ax.add_patch(rect)
            self._artists.append(rect)

            text = self.ax.text(
                c, r + 0.35, f'{child.value:.3f}',
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='black', zorder=5,
            )
            self._artists.append(text)

    # ------------------------------------------------------------------
    def hide(self):
        '''清除所有叠加元素。'''
        for artist in self._artists:
            artist.remove()
        self._artists.clear()


# =====================================================================
# MCTS 搜索可视化
# =====================================================================

class MCTSBoardOverlay:
    '''MCTS 棋盘热力图：在候选位置显示访问次数 + 价值。'''

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self._artists = []

    def show(self, root_visits: list):
        '''root_visits: [(move, visits, value), ...]'''
        self.hide()
        if not root_visits:
            return

        max_visits = max((v for _, v, _ in root_visits), default=1)
        max_visits = max(max_visits, 1)

        for move, visits, value in root_visits:
            if move is None or visits <= 0:
                continue
            r, c = move
            norm_v = max(0.0, min(1.0, value))
            color = plt.cm.RdYlGn(norm_v)
            radius = 0.12 + 0.3 * (visits / max_visits)

            circle = Circle(
                (c, r), radius=radius, facecolor=color,
                edgecolor='black', linewidth=0.5,
                alpha=0.65, zorder=4,
            )
            self.ax.add_patch(circle)
            self._artists.append(circle)

            text = self.ax.text(
                c, r, f'{visits}',
                ha='center', va='center',
                fontsize=7, fontweight='bold',
                color='black', zorder=5,
            )
            self._artists.append(text)

    def hide(self):
        for artist in self._artists:
            artist.remove()
        self._artists.clear()


class MCTSTreeOverlay:
    '''MCTS 树结构图：在右侧区域画搜索树。'''

    MAX_DEPTH = 3

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self._artists = []

    def show(self, tree_nodes: list):
        self.hide()
        self.ax.clear()
        if not tree_nodes:
            self.ax.text(0.5, 0.5, 'MCTS 搜索树',
                         transform=self.ax.transAxes, ha='center',
                         va='center', fontsize=10, color='#bbb')
            self.ax.axis('off')
            return

        node_map = {t['id']: t for t in tree_nodes}
        depth_nodes = {}
        for t in tree_nodes:
            depth_nodes.setdefault(t['depth'], []).append(t['id'])
        for d in depth_nodes:
            depth_nodes[d].sort(key=lambda nid: node_map[nid]['id'])

        positions = {}
        for d, ids in depth_nodes.items():
            n = len(ids)
            for i, nid in enumerate(ids):
                x = (i + 0.5) / n
                y = 1.0 - (d + 0.5) / (self.MAX_DEPTH + 1)
                positions[nid] = (x, y)

        # 画边（先画，避免盖住节点）
        for t in tree_nodes:
            if not t['children']:
                continue
            px, py = positions[t['id']]
            for cid in t['children']:
                if cid in positions:
                    cx, cy = positions[cid]
                    line = self.ax.plot(
                        [px, cx], [py, cy],
                        color='gray', linewidth=0.5, alpha=0.6, zorder=1,
                    )[0]
                    self._artists.append(line)

        for t in tree_nodes:
            x, y = positions[t['id']]
            visits = t['visits']
            value = t['value'] if t['value'] is not None else 0.5
            radius = 0.015 + 0.03 * np.log1p(visits)

            color = plt.cm.RdYlGn(max(0.0, min(1.0, value)))
            circle = plt.Circle(
                (x, y), radius=radius, facecolor=color,
                edgecolor='gray', linewidth=0.3, zorder=3,
            )
            self.ax.add_patch(circle)
            self._artists.append(circle)

            if t['move'] is not None and visits >= 2:
                label = idx_to_go_str(t['move'])
                text = self.ax.text(
                    x, y, label, ha='center', va='center',
                    fontsize=5, color='black', zorder=4,
                )
                self._artists.append(text)

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.ax.set_title('MCTS 搜索树', fontsize=8)

    def hide(self):
        for artist in self._artists:
            artist.remove()
        self._artists.clear()
