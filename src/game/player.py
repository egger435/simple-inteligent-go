'''围棋对局状态管理。'''

from sgfmill import boards, ascii_boards

from common import BOARD_SIZE, HAVE_I, idx_to_go_str, go_str_to_idx


class GoPlayer:
    '''围棋对局控制器。

    管理棋盘状态、行棋方、落子历史和劫争信息。
    '''

    def __init__(self, ai_player: str = 'b'):
        self.init_player = 'b'
        self.ai_player = ai_player
        self.board = boards.Board(BOARD_SIZE)
        self.current_color = 'b'

        self.last_board = self.board.copy()
        self.last_color = self.current_color

        self.curstep = 0
        self.steps = []       # 落子记录 [[color, pos_str], ...]
        self.ko_pos = None    # 当前劫争位

    # ------------------------------------------------------------------
    def play_move_str(self, go_str: str) -> None:
        '''执行落子（字符坐标，如 "J8"）。'''
        if go_str == 'pass':
            print(f'{self.current_color.upper()} 弃行')
        else:
            pos = go_str_to_idx(go_str)
            self.last_board = self.board.copy()
            self.last_color = self.current_color
            self.ko_pos, _ = self.board.play(pos[0], pos[1], self.current_color)
            print(f'{self.current_color.upper()} 落子：{idx_to_go_str((pos[0], pos[1]), HAVE_I)}')
            self.steps.append([
                self.current_color.upper(),
                idx_to_go_str((pos[0], pos[1]), False),
            ])

        self.current_color = 'w' if self.current_color == 'b' else 'b'
        self.curstep += 1

    # ------------------------------------------------------------------
    def play_move(self, go_pos: tuple) -> None:
        '''执行落子（元组坐标 (row, col)），返回劫争位。'''
        self.ko_pos, _ = self.board.play(go_pos[0], go_pos[1], self.current_color)
        self.steps.append([
            self.current_color.upper(),
            idx_to_go_str((go_pos[0], go_pos[1]), False),
        ])
        self.current_color = 'w' if self.current_color == 'b' else 'b'
        self.curstep += 1

    # ------------------------------------------------------------------
    def print_board(self) -> None:
        print('当前棋盘状态')
        print(ascii_boards.render_board(self.board, HAVE_I))

    # ------------------------------------------------------------------
    def back_state(self) -> None:
        '''悔棋：恢复到上一个棋盘状态。'''
        self.board = self.last_board.copy()
        self.current_color = self.last_color
