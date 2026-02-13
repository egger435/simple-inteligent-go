import os
from sgfmill import sgf, boards
import models.strategy_net as sg
import models.value_net as va
import numpy as np

# common =========================================================================================
DEVICE     = 'cuda'
'''运行设备'''            
PASS_LABEL = 361
'''弃行标签, 不修改'''
BOARD_SIZE = 19 
'''棋盘大小, 不修改'''               
COLOR_MAP  = {1: 0.0, 2: 1.0}
'''行棋方数值映射, 不修改'''
HAVE_I     = False
'''棋盘坐标是否包含I'''
GAME_KOMI  = 0.0
'''棋局贴目数量'''
# ================================================================================================



# 数据集生成相关 ==================================================================================
SGF_DATASET_DIR      = 'data\gogod_commentary_sgfs\gogod_commentary'
'''原始sgf数据位置'''

# 策略网络数据集====================================
SG_DATASET_SAVE_PATH = 'E:/go_dataset/strategy_net'  
'''策略网络数据集保存位置'''
SG_CHUNK_SAMPLE_NUM  = 500000  
'''每50万个样本写入一个策略网络npz文件'''

# 终局价值网络数据集================================
VA_DATASET_SAVE_PATH      = 'E:/go_dataset/value_net_dataset.npz'  
'''价值网络数据集保存位置'''
# ================================================================================================



# 模型训练相关 ====================================================================================
LOG_PATH            = 'D:/01_EGGER/program/python/simple-inteligent-go/output/log_resnet.txt'
'''训练日志的保存位置'''

# 策略网络模型训练===================================
SG_CHUNK_DIR        = 'E:/go_dataset/strategy_net' 
'''分片数据集位置'''
SG_VALID_CHUNK_DIR  = 'E:/go_dataset/strategy_net/valid_chunk.npz' 
'''策略网络训练验证集位置'''
SG_SAVE_MODEL_PATH  = 'D:/01_EGGER/program/python/simple-inteligent-go/output/go_strategy_model.pth'
'''策略网络模型数据保存位置'''
SG_BATCH_SIZE       = 32
SG_EPOCHS_PER_CHUNK = 3
SG_LEARNING_RATE    = 1e-4
SG_WEIGHT_DECAY     = 1e-5
SG_GRADIENT_ACCUMULATION_STEPS = 2

# 价值网络模型训练===================================
VA_DATASET_PATH     = 'E:/go_dataset/value_net_dataset.npz'
'''价值网络数据集位置'''
VA_SAVE_MODEL_PATH  = 'D:/01_EGGER/program/python/simple-inteligent-go/output/go_final_val_model_1_2.pth'
'''价值网络模型数据保存位置'''
VA_BATCH_SIZE       = 128
VA_LEARNING_RATE    = 1e-3
VA_WEIGHT_DECAY     = 1e-4
VA_PATIENCE         = 25
# ================================================================================================



# 策略选择相关 ====================================================================================
STRATEGY_MODEL_PATH = "output\go_strategy_model.pth"
'''策略网络模型数据位置'''
STRATEGY_MODEL      = sg.GoCNN_p().to(DEVICE)
'''策略选择网络模型选择'''
TOP_K               = 3   
'''最优候选落子位置个数'''
# ================================================================================================



# 价值判断相关 ====================================================================================
VALUE_MODEL_PATH = 'output\go_final_val_model_1_2.pth'
'''终局价值网络模型数据位置'''
VALUE_MODEL      = va.GoValueNet().to(DEVICE)
'''终局价值网络模型选择'''
# ================================================================================================



# 工具函数 ========================================================================================
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

def read_go_sgf(sgf_content):
    '''
    根据sgf文件字节流解析对局信息和每步落子并生成落子记录列表

    :param sgf_content: sgf文件内容字节流

    :return:
        game_info:    对局信息字典

        step_records: 每步落子记录列表
    '''
    # 实例化对局
    game = sgf.Sgf_game.from_bytes(sgf_content)

    # 获取对局信息
    main_sequence = list(game.get_main_sequence())  # 落子步骤主序列
    board_size    = game.get_size()
    black_player  = game.get_player_name('b')
    white_player  = game.get_player_name('w')
    komi          = game.get_komi()  # 贴目
    winner        = game.get_winner()

    game_info = {
        'board_size':   board_size,
        'black_player': black_player,
        'white_player': white_player,
        'komi':         komi,
        'winner':       winner
    }

    # 初始化棋盘
    board = boards.Board(board_size)
    step_records = []

    # 生成对弈序列和棋盘矩阵
    for idx, node in enumerate(main_sequence, 1):  # 跳过根节点
        move = node.get_move()
        if not move:  # 空操作判断
            step_records.append({
                'step':         idx,   # 步数
                'color_code':   None,  # 行棋方颜色代码 1黑棋, 2白棋
                'pos':          None,  # 落子坐标 元组
                'pos_str':      None,  # 落子坐标字符串
                'board_matrix': None,  # 当前棋盘矩阵
            })
            continue
        color, pos = move
        color_code = 1 if color == 'b' else 2  
        pos_str = f'({pos[0]}, {pos[1]})' if pos else 'pass'  # 落子坐标字符串
        if pos:
            board.play(pos[0], pos[1], color)

        # 将当前棋盘状态转换为矩阵
        board_matrix = np.zeros((board_size, board_size), dtype=int)
        for r in range(board_size):
            for c in range(board_size):
                stone = board.get(r, c)
                if stone == 'b':
                    board_matrix[r, c] = 1
                elif stone == 'w':
                    board_matrix[r, c] = 2
        
        # 记录每步信息
        step_records.append({
            'step':         idx,
            'color_code':   color_code,
            'pos':          pos,
            'pos_str':      pos_str,
            'board_matrix': board_matrix,
        })

    return game_info, step_records

def get_final_from_sgf(sgf_content):
    '''
    根据sgf文件字节流得到终局信息, 返回终局矩阵(包含贴目)和终局价值列表
    '''
    game = sgf.Sgf_game.from_bytes(sgf_content)
    main_sequence = list(game.get_main_sequence())

    winner = game.get_winner()
    komi = game.get_komi()

    board = boards.Board(BOARD_SIZE)
    for idx, node in enumerate(main_sequence, 1):
        move = node.get_move()
        if not move:
            continue
        color, pos = move
        if pos:
            board.play(pos[0], pos[1], color)

    final_board_matrix  = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = board.get(r, c)
            if stone == 'b':
                final_board_matrix[r, c] = 0.5
            elif stone == 'w':
                final_board_matrix[r, c] = 1.0
    
    # 生成贴目通道
    komi_matrix = np.full(shape=(BOARD_SIZE, BOARD_SIZE), fill_value=komi, dtype=np.float32)
    board_with_komi_matrix = np.stack([final_board_matrix, komi_matrix], axis=0)
    
    if winner == 'b':
        final_val = [1.0, 0.0]
    elif winner == 'w':
        final_val = [0.0, 1.0]
    else:
        final_val = None

    return board_with_komi_matrix, final_val

def get_final_board_from_sgf(sgf_content):
    '''根据sgf文件字节流读取终局信息, 返回终局棋盘和贴目数以及胜利方'''
    game = sgf.Sgf_game.from_bytes(sgf_content)
    main_sequence = list(game.get_main_sequence())

    winner = game.get_winner()
    komi = game.get_komi()

    board = boards.Board(BOARD_SIZE)
    for idx, node in enumerate(main_sequence, 1):
        move = node.get_move()
        if not move:
            continue
        color, pos = move
        if pos:
            board.play(pos[0], pos[1], color)
    
    return board, komi, winner

def link_next_move(step_records):
    '''
    将每一步落子信息和下一步落子信息链接

    :param step_records: 每步落子记录列表

    :return: 
        board_map: 落子信息映射元组 (当前落子信息, 简化版下一步落子信息)
    '''
    board_map = []
    for i in range(len(step_records) - 1):
        cur_step = step_records[i]
        next_step = step_records[i + 1]
        simple_next_step = {                 
            'color_code': next_step['color_code'],
            'pos':        next_step['pos']
        }
        board_map.append((cur_step, simple_next_step))
    return board_map

def np_input_to_board(np_input: np.ndarray):
    '''将numpy矩阵转化为棋盘和行棋方输出'''
    board = boards.Board(BOARD_SIZE)
    np_board = np_input[0]  # 训练数据第一个通道为棋盘通道
    np_player = np_input[1] # 第二个数据为下一个行棋方

    # 得到棋盘
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            val = np_board[r, c]
            if val == 0.0:
                continue
            if val == 1.0:
                color = 'b'
            elif val == 2.0:
                color = 'w'
            board.play(r, c, color)
    
    # 得到下一个行棋方
    player = 'b' if np_player[0, 0] == 0.0 else 'w'

    return board, player

def get_sorted_chunk_files(chunk_dir):
    '''获取有序文件列表'''
    chunk_files = []
    for file in os.listdir(chunk_dir):
        if file.startswith('go_dataset') and file.endswith('.npz'):
            chunk_idx = int(file.split('_')[-1].split('.')[0])
            chunk_files.append((chunk_idx, os.path.join(chunk_dir, file)))
    chunk_files.sort(key=lambda x: x[0])
    return [f[1] for f in chunk_files]  # 返回路径

if __name__ == '__main__':
    sgf_path = "D:\\02_EdgeDownload\\varied_selfplay_commentary_sgfs\\varied_selfplay_commentary\\60\\985.sgf"
    with open(sgf_path, 'rb') as f:
        sgf_content = f.read()
    np_board, val = get_final_from_sgf(sgf_content)
    print(np_board, val)
    
