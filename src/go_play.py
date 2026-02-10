import numpy as np
import torch
from sgfmill import boards, ascii_boards
import strategy.go_strategy as go_sg
from common import *
                
class GoPlayer:
    def __init__(self, init_player='b'):
        # 加载模型
        self.model = STRATEGY_MODEL.to(DEVICE)
        checkpoint = torch.load(STRATEGY_MODEL_PATH, map_location=DEVICE)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()  
        
        # 初始化棋盘
        self.board = boards.Board(BOARD_SIZE)
        self.current_color = init_player

        # 记录上一个状态
        self.last_board = self.board.copy()
        self.last_color = self.current_color

    def strategy_model_select(self):
        """策略选择得到top_k个候选"""
        return go_sg.model_pred(self.board, self.current_color)

    def play_move_str(self, go_str:str):
        """执行落子 字符坐标"""
        if go_str == "pass":
            print(f"{self.current_color.upper()} 弃行")
        else:
            pos = go_str_to_idx(go_str)
            self.last_board = self.board.copy()
            self.last_color = self.current_color
            self.board.play(pos[0], pos[1], self.current_color)
            print(f"{self.current_color.upper()} 落子：({pos[0]}, {pos[1]})")
        
        # 切换行棋方
        self.current_color = 'w' if self.current_color == 'b' else 'b'
    
    def play_move(self, go_pos:tuple):
        '''执行落子 元组坐标'''
        self.board.play(go_pos[0], go_pos[1], self.current_color)
        self.current_color = 'w' if self.current_color == 'b' else 'b'

    def print_board(self):
        print("当前棋盘状态")
        print(ascii_boards.render_board(self.board, HAVE_I))

    def back_state(self):
        '''悔棋'''
        self.board = self.last_board.copy()
        self.current_color = self.last_color

def vs_ai():
    # 初始化
    player = GoPlayer()
    print("* 落子输入格式：列字符+行坐标 J8 | 输入 pass 弃行 | 输入 back 悔棋")
    
    while True:
        # 人类落子
        player.print_board()
        human_input = input("落子位置: ")
        if human_input == "pass":
            player.play_move_str("pass")
        elif human_input == 'back':
            player.back_state()
            continue
        else:
            player.play_move_str(human_input)
        
        # AI落子
        ai_candidates = player.strategy_model_select()  
        if ai_candidates[0][0] == 'pass':
            print("AI选择弃行")
            player.play_move_str('pass')
        else:
            print(f'AI候选落子位置:')
            for candidate in ai_candidates:
                print(f'  落子: {idx_to_go_str(candidate[0])} | 概率: {candidate[1]:.4f}')
            ai_pos, ai_prob = ai_candidates[0]
            print(f"AI 选择落子：{idx_to_go_str(ai_pos)} 概率：{ai_prob:.4f}")
            player.play_move(ai_pos)

if __name__ == "__main__":
    vs_ai()