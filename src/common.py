import models.strategy_net as sg

# common
DEVICE     = 'cuda'            # 运行设备
PASS_LABEL = 361               # 弃行标签, 不修改
BOARD_SIZE = 19                # 棋盘大小, 不修改
COLOR_MAP  = {1: 0.0, 2: 1.0}  # 行棋方数值映射, 不修改
HAVE_I     = False             # 棋盘坐标是否包含I

# 策略选择相关
STRATEGY_MODEL_PATH = "output\go_strategy_model.pth"
STRATEGY_MODEL      = sg.GoCNN_p()  # 策略选择网络模型选择
TOP_K               = 3             # 最优候选落子位置个数

def go_str_to_idx(go_str):
    '''将棋盘字符坐标输入转化为索引坐标'''
    col_char = go_str[0].upper()
    row_num = int(go_str[1:])

    col_map_I = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7,
                 'I':8, 'J':9, 'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,
                 'Q':16,'R':17,'S':18}
    col_map   = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7,
                 'J':8, 'K':9, 'L':10,'M':11,'N':12,'O':13,'P':14,'Q':15,
                 'R':16,'S':17,'T':18}
    
    col_idx = col_map_I[col_char] if HAVE_I else col_map[col_char]
    row_idx = row_num - 1  
    
    return (row_idx, col_idx)

def idx_to_go_str(idx):
    '''将索引坐标转化为字符坐标'''
    r, c = idx
    col_map_I_rev = {0:'A', 1:'B', 2:'C',  3:'D',  4:'E',  5:'F',  6:'G',  7:'H',
                     8:'I', 9:'J',10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P',
                    16:'Q',17:'R',18:'S'}
    col_map_rev   = {0:'A', 1:'B', 2:'C',  3:'D',  4:'E',  5:'F',  6:'G',  7:'H',
                     8:'J', 9:'K',10:'L', 11:'M', 12:'N', 13:'O', 14:'P', 15:'Q',
                    16:'R',17:'S',18:'T'}
    rownum = r + 1
    colchr = col_map_I_rev[c] if HAVE_I else col_map_rev[c]
    return f'{colchr}{rownum}'