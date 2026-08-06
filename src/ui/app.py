'''围棋 AI 图形界面 —— 主控制器。

整合棋盘渲染、概率叠加、信息面板，管理对局循环。

用法::

    go_player = GoPlayer(ai_player='w')
    gui = GoBoardGUI(go_player)
    gui.run()
'''

import threading
import time
import matplotlib
matplotlib.use('TkAgg')          # 必须在 import pyplot 之前
import matplotlib.pyplot as plt

# ---- 配置中文字体（解决 tkinter 后端 CJK 字形缺失警告） ----
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
    'DejaVu Sans', 'Arial',
]
plt.rcParams['axes.unicode_minus'] = False   # 修复负号显示

from game.player import GoPlayer
from game.tree import BatchMinimaxMCR
from common import GAME_KOMI, idx_to_go_str
from ui.board_view import BoardRenderer
from ui.overlay import (
    ProbOverlay, ValueOverlay, MCTSBoardOverlay, MCTSTreeOverlay,
)
from ui.info_panel import InfoPanel


class GoBoardGUI:
    '''围棋 AI 图形界面。'''

    def __init__(self, go_player: GoPlayer):
        self.go_player = go_player
        self.ai_player = go_player.ai_player
        self.human_player = 'w' if self.ai_player == 'b' else 'b'

        # ---- 事件状态 ----
        self._running = False
        self._ai_needed = False        # AI 需要走棋
        self._ai_running = False       # AI 线程正在运行
        self._ai_done = False          # AI 线程已结束，等待结果处理
        self._last_move: tuple | None = None

        # ---- AI 结果（由 AI 线程写入，主线程读取） ----
        self._ai_result: tuple | None = None
        self._ai_candidates: list | None = None
        self._ai_root_children: list | None = None
        self._ai_error: str | None = None
        self._last_ai_info: dict = {}

        # ---- 胜率历史 ----
        self._winrate_history: list = []   # [(步数, 胜率), ...]

        # ---- MCTS 可视化 ----
        self._mcts_snapshot: dict | None = None   # 后台线程写入的快照
        self._mcts_visualizing = False             # 是否正在可视化

        # ---- matplotlib 对象 ----
        self._fig: plt.Figure | None = None
        self._ax_board: plt.Axes | None = None
        self._ax_info: plt.Axes | None = None
        self._ax_graph: plt.Axes | None = None
        self._board: BoardRenderer | None = None
        self._prob: ProbOverlay | None = None
        self._value: ValueOverlay | None = None
        self._mcts_board: MCTSBoardOverlay | None = None
        self._mcts_tree: MCTSTreeOverlay | None = None
        self._info: InfoPanel | None = None

    # =================================================================
    # 初始化
    # =================================================================

    def _create_figure(self):
        '''创建 matplotlib 图形。'''
        self._fig = plt.figure(
            'Kata-EgGO',
            figsize=(13, 9),
            facecolor='#f5f0e8',
        )
        # 棋盘（左）
        self._ax_board = self._fig.add_axes(
            [0.04, 0.04, 0.62, 0.92],
            facecolor='#dcb35c',
        )
        # 信息面板（右上）
        self._ax_info = self._fig.add_axes(
            [0.69, 0.42, 0.29, 0.54],
            facecolor='white',
        )
        # 胜率走势图（右下）
        self._ax_graph = self._fig.add_axes(
            [0.69, 0.04, 0.29, 0.34],
            facecolor='white',
        )

        # 事件绑定
        self._fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._fig.canvas.mpl_connect('key_press_event', self._on_key)

        # 子模块
        self._board = BoardRenderer(self._ax_board)
        self._prob = ProbOverlay(self._ax_board)
        self._value = ValueOverlay(self._ax_board)
        self._mcts_board = MCTSBoardOverlay(self._ax_board)
        self._mcts_tree = MCTSTreeOverlay(self._ax_graph)
        self._info = InfoPanel(self._ax_info)

    # =================================================================
    # 渲染
    # =================================================================

    def _draw_winrate_graph(self):
        '''在右下角绘制 AI 胜率走势图。'''
        ax = self._ax_graph
        ax.clear()

        if len(self._winrate_history) < 1:
            # 空图提示
            ax.text(0.5, 0.5, 'AI 胜率走势',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='#bbb')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return

        steps, values = zip(*self._winrate_history)
        line_color = '#1a1a1a' if self.ai_player == 'b' else '#888888'

        ax.plot(steps, values, 'o-', color=line_color, linewidth=1.5,
                markersize=4, markerfacecolor=line_color, markeredgecolor='white',
                markeredgewidth=0.5)

        # 参考线 0.5
        ax.axhline(y=0.5, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.6)

        # 标注
        ax.set_title('AI 胜率走势', fontsize=9, fontweight='bold', pad=4)
        ax.set_ylabel('胜率', fontsize=7)
        ax.set_xlabel('步数', fontsize=7)
        ax.set_ylim(-0.02, 1.02)

        # x 轴自动范围
        if len(steps) >= 2:
            ax.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
        else:
            ax.set_xlim(steps[0] - 0.5, steps[0] + 0.5)

        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda y, _: f'{y:.0%}'))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------------
    def _render(self):
        '''更新整个界面。'''
        # 棋盘
        self._board.draw(self.go_player.board, self._last_move)

        # 概率叠加（AI 思考时展示）
        if self._ai_running and self._ai_candidates:
            self._prob.show(self._ai_candidates, board=self.go_player.board)
        else:
            self._prob.hide()

        # 价值叠加（AI 走完之后展示评估值）
        if not self._ai_running and self._ai_root_children:
            children_with_val = [
                c for c in self._ai_root_children if c.value is not None
            ]
            if children_with_val:
                self._value.show(children_with_val)
            else:
                self._value.hide()
        else:
            self._value.hide()

        # MCTS 可视化叠加（搜索过程中）
        if self._mcts_visualizing and self._mcts_snapshot:
            self._mcts_board.show(self._mcts_snapshot['root_visits'])
            self._mcts_tree.show(self._mcts_snapshot['tree'])
            self._mcts_tree.ax.figure.canvas.draw_idle()
        else:
            self._mcts_board.hide()
            self._mcts_tree.hide()

        # 信息面板
        game_info = {
            '人类执棋': '黑 ●' if self.human_player == 'b' else '白 ○',
            'AI 执棋': '黑 ●' if self.ai_player == 'b' else '白 ○',
            '步数': self.go_player.curstep,
            '当前方': '黑 ●' if self.go_player.current_color == 'b' else '白 ○',
            '贴目': f'{GAME_KOMI}',
        }
        if self._mcts_visualizing and self._mcts_snapshot:
            game_info['MCTS 模拟'] = f"{self._mcts_snapshot['total_sims']}"

        if self._ai_running:
            status = 'AI 思考中...'
        elif self._ai_needed:
            status = ''
        else:
            status = '请落子（点击棋盘）'

        if self._ai_error:
            status = f'⚠ {self._ai_error}'

        self._info.update(game_info, self._last_ai_info,
                          history=self.go_player.steps, status=status)

        # ---- 胜率走势图（MCTS 可视化时树图占用该区域） ----
        if not (self._mcts_visualizing and self._mcts_snapshot):
            self._draw_winrate_graph()

        self._fig.canvas.draw_idle()

    # =================================================================
    # 事件处理
    # =================================================================

    def _on_click(self, event):
        '''鼠标点击 → 人类落子。'''
        if not self._running:
            return
        if self._ai_needed or self._ai_running:
            return        # AI 回合，不处理
        if event.inaxes != self._ax_board:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < 19 and 0 <= col < 19):
            return

        # 检查落子合法性
        if self.go_player.board.get(row, col) is not None:
            print(f'该位置已有棋子')
            return

        try:
            self.go_player.play_move((row, col))
            self._last_move = (row, col)
            self._ai_needed = True
        except Exception as e:
            print(f'落子失败: {e}')

    # ------------------------------------------------------------------
    def _on_key(self, event):
        '''键盘快捷键。'''
        if not self._running:
            return
        key = event.key.lower() if event.key else ''

        if key == 'p':                           # 弃行
            if not self._ai_needed:
                self.go_player.play_move_str('pass')
                self._ai_needed = True
                self._last_move = None

        elif key == 'b':                         # 悔棋
            if not self._ai_running:
                self.go_player.back_state()
                self._ai_needed = False
                self._last_move = None
                self._ai_candidates = None
                self._ai_root_children = None
                self._last_ai_info = {}
                # 移除最近一条胜率记录
                if self._winrate_history:
                    self._winrate_history.pop()

        elif key == 'q' or key == 'escape':      # 退出
            self._running = False
            plt.close(self._fig)

    # =================================================================
    # AI 思考
    # =================================================================

    def _ai_thread_func(self):
        '''后台线程：按配置选择搜索算法（minimax / mcts）。'''
        try:
            import json
            cfg = json.load(open('config.json'))
            algorithm = cfg.get('search_algorithm', 'minimax')

            if algorithm == 'mcts':
                from game.mcts import MCTS
                self._mcts_visualizing = cfg.get('mcts_visualize', False)
                searcher = MCTS(
                    self.go_player.steps,
                    self.go_player.board,
                    self.go_player.current_color,
                    self.go_player.curstep,
                    verbose=cfg.get('mcts_verbose', False),
                    visualize=self._mcts_visualizing,
                    vis_interval=cfg.get('mcts_vis_interval', 0.5),
                )
                # 后台线程每步模拟后回调，更新 GUI 可见的快照
                searcher.on_step = lambda snap: self._on_mcts_step(snap)
            else:
                self._mcts_visualizing = False
                self._mcts_snapshot = None
                searcher = BatchMinimaxMCR(
                    self.go_player.steps,
                    self.go_player.board,
                    self.go_player.current_color,
                    self.go_player.curstep,
                    top_k=cfg.get('top_k', 4),
                    max_depth=cfg.get('max_search_depth', 5),
                    verbose=False,
                )

            best_move, best_value = searcher.search()

            self._ai_result = (best_move, best_value)
            self._ai_candidates = getattr(searcher, 'root_candidates', None)
            self._ai_root_children = (
                searcher.root.children if searcher.root else []
            )
            self._ai_error = None

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._ai_error = str(e)
            self._ai_result = None

        self._mcts_visualizing = False
        self._ai_done = True

    # ------------------------------------------------------------------
    def _on_mcts_step(self, snapshot: dict):
        '''后台线程回调：更新 MCTS 快照（主线程轮询渲染）。'''
        self._mcts_snapshot = snapshot

    # ------------------------------------------------------------------
    def _start_ai(self):
        '''启动 AI 线程。'''
        self._ai_running = True
        self._ai_done = False
        self._ai_result = None
        threading.Thread(target=self._ai_thread_func, daemon=True).start()

    # ------------------------------------------------------------------
    def _apply_ai_result(self):
        '''AI 思考完成 → 落子 + 更新信息。'''
        if self._ai_error:
            print(f'AI 错误: {self._ai_error}')
            self._ai_needed = False
            self._ai_running = False
            return

        best_move, best_value = self._ai_result

        if best_move == 'pass':
            print('AI 弃行')
            self.go_player.play_move_str('pass')
            self._last_move = None
        else:
            print(f'AI 落子: {idx_to_go_str(best_move)} | value: {best_value:.4f}')
            self.go_player.play_move(best_move)
            self._last_move = best_move

        # 记录胜率历史（每步一个点）
        self._winrate_history.append((self.go_player.curstep, best_value))

        self._last_ai_info = {
            '最佳落子': idx_to_go_str(best_move) if best_move != 'pass' else 'pass',
            '评估值': f'{best_value:.4f}',
            '候选数': str(len(self._ai_root_children)) if self._ai_root_children else '-',
        }

        self._ai_needed = False
        self._ai_running = False

    # =================================================================
    # 主循环
    # =================================================================

    def run(self):
        '''启动 GUI。'''
        plt.ion()                  # 交互模式（非阻塞）
        self._create_figure()
        plt.show(block=False)      # 显示窗口
        self._running = True

        # AI 执黑 → 先走
        if self.go_player.ai_player == 'b' and self.go_player.curstep == 0:
            self._ai_needed = True

        print('GUI 已启动 | 点击棋盘落子 | p=弃行 b=悔棋 q=退出')

        while self._running:
            # ---- 处理 AI 完成 ----
            if self._ai_done:
                self._apply_ai_result()
                self._ai_done = False

            # ---- 启动 AI ----
            if self._ai_needed and not self._ai_running:
                self._start_ai()

            # ---- 渲染 ----
            self._render()

            # ---- 事件刷新 ----
            try:
                self._fig.canvas.flush_events()
            except Exception:
                break

            time.sleep(0.06)

            if not plt.fignum_exists(self._fig.number):
                self._running = False

        print('GUI 已关闭')
