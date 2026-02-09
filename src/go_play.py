import numpy as np
import torch
from sgfmill import boards, ascii_boards 
from models.go_cnn import GoCNN  

MODEL_PATH = "output\go_model_step_GoCNN_v2.pth"  

BOARD_SIZE = 19                  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLOR_MAP = {1: 0.0, 2: 1.0}     
PASS_LABEL = 361                 


class GoPlayer:
    def __init__(self, model_path, init_player='b'):
        # 加载模型
        self.model = GoCNN().to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()  
        
        # 初始化棋盘
        self.board = boards.Board(BOARD_SIZE)
        self.current_color = init_player

    def board_to_model_input(self):
        """将当前棋盘状态转为模型输入格式"""
        # 生成棋盘矩阵
        board_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = self.board.get(r, c)
                if stone == 'b':
                    board_matrix[r, c] = 1.0
                elif stone == 'w':
                    board_matrix[r, c] = 2.0
        board_matrix = board_matrix / 2.0    # 归一化 
        
        # 生成行棋方通道
        color_code = 1 if self.current_color == 'b' else 2
        player_channel = np.full((BOARD_SIZE, BOARD_SIZE), COLOR_MAP[color_code], dtype=np.float32)
        
        # 得到模型输入
        model_input = np.stack([board_matrix, player_channel], axis=0)
        model_input = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return model_input

    def model_predict(self, top_k=3):
        """模型预测最优落子位置, 得到top_k个候选"""
        with torch.no_grad():
            input_tensor = self.board_to_model_input()
            output = self.model(input_tensor)
            prob = torch.softmax(output, dim=1)  # 转为概率分布
        
        # 解析预测结果
        prob_np = prob.cpu().numpy()[0]
        sorted_indices = np.argsort(prob_np)[::-1]  # 根据概率排序
        #print(sorted_indices)
        
        candidates = []
        for idx in sorted_indices:
            if idx == PASS_LABEL:
                candidates.append(("pass", prob_np[idx]))
                continue
            # 转换为棋盘坐标
            r = idx // BOARD_SIZE
            c = idx % BOARD_SIZE
            if self.is_legal((r, c)):
                candidates.append(((r, c), prob_np[idx]))
            if len(candidates) >= top_k:
                break

        return candidates

    def is_legal(self, pos):
        '''检查合法落子'''
        occupied_points = self.board.list_occupied_points()
        for points in occupied_points:
            if pos == points[1]:
                return False
        return True

    def play_move_str(self, go_str:str):
        """执行落子 字符坐标"""
        if go_str == "pass":
            print(f"{self.current_color.upper()} 弃行")
        else:
            pos = go_str_to_idx(go_str)
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
        print(ascii_boards.render_board(self.board))

def go_str_to_idx(go_str):
    '''将棋盘字符坐标输入转化为索引坐标'''
    col_char = go_str[0].upper()
    row_num = int(go_str[1:])

    col_map = {'A':0, 'B':1, 'C':2,  'D':3,  'E':4,  'F':5,  'G':6,  'H':7,
               'I':8, 'J':9, 'K':10, 'L':11, 'M':12, 'N':13, 'O':14, 'P':15,
               'Q':16,'R':17,'S':18}
    
    if col_char not in col_map:
        raise ValueError(f"无效的列字母: {col_char}")
    if not (1 <= row_num <= 19):
        raise ValueError(f"无效行号: {row_num}")
    
    col_idx = col_map[col_char]
    row_idx = row_num - 1  
    
    return (row_idx, col_idx)

def idx_to_go_str(idx):
    '''将索引坐标转化为字符坐标'''
    r, c = idx
    col_map_rev = {0:'A', 1:'B',  2:'C',  3:'D',  4:'E',  5:'F',  6:'G',  7:'H',
                   8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P',
                   16:'Q',17:'R',18:'S'}
    rownum = r + 1
    colchr = col_map_rev[c]
    return f'{colchr}{rownum}'

def human_vs_ai():
    # 初始化
    player = GoPlayer(MODEL_PATH)
    print("* 落子输入格式：列字符+行坐标 J8 | 输入 pass 表示弃行")
    print("* quit 结束对弈\n")
    
    while True:
        # 人类落子
        player.print_board()
        human_input = input("落子位置: ")
        if human_input == "pass":
            player.play_move_str("pass")
        else:
            player.play_move_str(human_input)
        
        # AI落子
        ai_candidates = player.model_predict(top_k=3)  # 取最优解
        if not ai_candidates:
            print("AI选择弃行")
            player.play_move_str("pass")
        else:
            print(f'AI候选落子位置:')
            for candidate in ai_candidates:
                print(f'  落子: {idx_to_go_str(candidate[0])} | 概率: {candidate[1]:.4f}')
            ai_pos, ai_prob = ai_candidates[0]
            print(f"AI 选择落子：{idx_to_go_str(ai_pos)}（概率：{ai_prob:.4f}）")
            player.play_move(ai_pos)

if __name__ == "__main__":
    human_vs_ai()