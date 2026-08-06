'''AI 思考信息面板。

在 matplotlib Axes 上以文字展示对局状态、AI 搜索统计和落子记录。
'''

from collections import deque

import matplotlib.pyplot as plt

from common import HAVE_I, go_str_to_idx, idx_to_go_str


class InfoPanel:
    '''AI 思考信息文字面板。

    在单独的 Axes 上以格式化文字展示对局信息和 AI 搜索统计。

    用法::

        panel = InfoPanel(ax)
        panel.update(
            game_info={'步数': 23, '执棋': 'W', '贴目': 7.5},
            ai_info={'胜率': 0.52, '叶节点': 27, '最佳落子': 'P4'},
            history=[('B', 'Q16'), ('W', 'D4'), ...],
        )
    '''

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.ax.axis('off')
        self._text: plt.Text | None = None

    # ------------------------------------------------------------------
    def update(self, game_info: dict, ai_info: dict,
               history: list = None, status: str = ''):
        '''更新信息面板。'''
        self.ax.clear()
        self.ax.axis('off')

        lines = []

        # ---- 状态指示 ----
        if status:
            lines.append(f'【{status}】')
            lines.append('')

        # ---- 对局信息 ----
        lines.append('── 对局信息 ──')
        for key, val in game_info.items():
            lines.append(f'  {key}: {val}')
        lines.append('')

        # ---- AI 搜索信息 ----
        if ai_info:
            lines.append('── AI 搜索 ──')
            for key, val in ai_info.items():
                lines.append(f'  {key}: {val}')
            lines.append('')

        # ---- 最近落子 ----
        if history:
            lines.append('── 落子记录 ──')
            recent = history[-10:] if len(history) > 10 else history
            for i, (color, pos) in enumerate(recent):
                num = len(history) - len(recent) + i + 1
                stone = '●' if color == 'B' else '○'
                # steps 始终按标准格式（无 I 列）存储，显示时转为用户坐标系
                if pos != 'pass' and HAVE_I:
                    idx = go_str_to_idx(pos, have_i=False)
                    pos = idx_to_go_str(idx, have_i=True)
                lines.append(f'  {num:3d}. {stone} {pos}')

        # ---- 渲染 ----
        text = '\n'.join(lines)
        self._text = self.ax.text(
            0.05, 0.95, text,
            transform=self.ax.transAxes,
            ha='left', va='top',
            fontsize=9,
            linespacing=1.3,
        )

    # ------------------------------------------------------------------
    def show_thinking(self, game_info: dict):
        '''显示「AI 思考中...」状态。'''
        self.update(game_info, {}, status='AI 思考中...')

    # ------------------------------------------------------------------
    def show_waiting(self, game_info: dict, history: list = None):
        '''显示「等待落子」状态。'''
        self.update(game_info, {}, history=history, status='请落子')
