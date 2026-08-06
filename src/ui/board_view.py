'''围棋棋盘渲染模块。

绘制 19×19 棋盘、棋子、星位、坐标标注，处理鼠标点击事件。
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sgfmill import boards

from common import BOARD_SIZE, HAVE_I, idx_to_go_str, go_str_to_idx


class BoardRenderer:
    '''19×19 围棋棋盘渲染器。

    在 matplotlib Axes 上绘制棋盘，处理鼠标点击返回落子坐标。

    用法::

        renderer = BoardRenderer(ax)
        renderer.draw(board)                     # 绘制完整棋盘
        row, col = renderer.wait_for_click()     # 阻塞等待点击
    '''

    # 星位（天元和角星）
    STAR_POINTS = [
        (3, 3), (3, 9), (3, 15),
        (9, 3), (9, 9), (9, 15),
        (15, 3), (15, 9), (15, 15),
    ]

    def __init__(self, ax: plt.Axes, have_i: bool = None):
        self.ax = ax
        self.have_i = have_i if have_i is not None else HAVE_I
        self._last_move: tuple | None = None      # 最后落子高亮
        self._click_result: tuple | None = None   # (row, col) 点击结果
        self._cid: int | None = None              # 事件连接 ID
        self._stone_artists: list = []            # 棋子图形对象
        self._highlight: Circle | None = None     # 高亮圆圈

    # ------------------------------------------------------------------
    def draw(self, board: boards.Board, last_move: tuple = None):
        '''绘制完整棋盘状态。'''
        self._last_move = last_move
        self.ax.clear()
        self._stone_artists.clear()
        self._highlight = None

        self._draw_grid()
        self._draw_star_points()
        self._draw_stones(board)
        self._draw_labels()

        if last_move:
            self._draw_last_move_marker(last_move)

        self.ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
        self.ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.invert_yaxis()

    # ------------------------------------------------------------------
    def _draw_grid(self):
        '''绘制 19×19 网格线。'''
        for i in range(BOARD_SIZE):
            self.ax.axhline(i, color='black', linewidth=0.5, zorder=0)
            self.ax.axvline(i, color='black', linewidth=0.5, zorder=0)

    # ------------------------------------------------------------------
    def _draw_star_points(self):
        '''绘制星位标记点。'''
        for r, c in self.STAR_POINTS:
            circle = Circle(
                (c, r), radius=0.08, color='black',
                fill=True, zorder=1,
            )
            self.ax.add_patch(circle)

    # ------------------------------------------------------------------
    def _draw_stones(self, board: boards.Board):
        '''绘制棋盘上所有棋子。'''
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = board.get(r, c)
                if stone is None:
                    continue
                color = 'black' if stone == 'b' else 'white'
                edge = 'white' if stone == 'b' else 'black'
                circle = Circle(
                    (c, r), radius=0.44, facecolor=color,
                    edgecolor=edge, linewidth=1, zorder=2,
                )
                self.ax.add_patch(circle)
                self._stone_artists.append(circle)

    # ------------------------------------------------------------------
    def _draw_labels(self):
        '''绘制行列坐标标注。'''
        for c in range(BOARD_SIZE):
            label = idx_to_go_str((0, c), self.have_i)[0]  # 列字母
            self.ax.text(c, -0.7, label, ha='center', va='center',
                         fontsize=7, color='black')

        for r in range(BOARD_SIZE):
            label = str(r + 1)
            self.ax.text(-0.7, r, label, ha='center', va='center',
                         fontsize=7, color='black')

    # ------------------------------------------------------------------
    def _draw_last_move_marker(self, pos: tuple):
        '''在最后落子位置绘制高亮标记（红色方块）。'''
        r, c = pos
        marker = Circle(
            (c, r), radius=0.15, facecolor='red',
            edgecolor='none', alpha=0.7, zorder=3,
        )
        self.ax.add_patch(marker)
        self._highlight = marker

    # ------------------------------------------------------------------
    def _on_click(self, event):
        '''鼠标点击回调：将像素坐标转换为棋盘行列索引。'''
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # 取整到最近交叉点
        col = int(round(event.xdata))
        row = int(round(event.ydata))

        # 边界检查
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            self._click_result = (row, col)

    # ------------------------------------------------------------------
    def connect_click(self):
        '''注册鼠标点击事件。'''
        self._cid = self.ax.figure.canvas.mpl_connect(
            'button_press_event', self._on_click,
        )

    # ------------------------------------------------------------------
    def disconnect_click(self):
        '''移除鼠标点击事件。'''
        if self._cid is not None:
            self.ax.figure.canvas.mpl_disconnect(self._cid)
            self._cid = None

    # ------------------------------------------------------------------
    def get_click(self) -> tuple | None:
        '''获取最近一次点击坐标，返回 (row, col) 或 None。'''
        result = self._click_result
        self._click_result = None
        return result
