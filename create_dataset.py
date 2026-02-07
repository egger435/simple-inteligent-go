import numpy as np
from sgfmill import sgf, boards
import os
import h5py

# 总处理文件数：76771
# 有效训练样本数：16133673
# 数据保存路径：E:\go_dataset\go_dataset.npz

sgf_dataset_dir = "dataset/gogod_commentary_sgfs/gogod_commentary"
SAVE_PATH = "E:/go_dataset/go_dataset.npz"
COLOR_MAP = {1: 0.0, 2: 1.0}  # 训练模型时候将行棋方编码转换为0和1

def read_go_sgf(sgf_content):
    '''
    根据sgf文件字节流解析对局信息和每步落子并生成落子记录列表

    :param sgf_content: sgf文件内容字节流

    :return:
        game_info:    对局信息字典

        step_records: 每步落子记录列表
    '''
    # 实例化对局并获取根节点
    game = sgf.Sgf_game.from_bytes(sgf_content)
    root_node = game.get_root()

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
    for idx, node in enumerate(main_sequence[1:], 1):  # 跳过根节点
        move = node.get_move()
        if not move:  # 空操作判断
            step_records.append({
                'step':         idx,   # 步数
                'color_code':   None,  # 行棋方颜色代码 1黑棋, 2白棋
                'pos':          None,  # 落子坐标 元组
                'pos_str':      None,  # 落子坐标字符串
                'board_matrix': None   # 当前棋盘矩阵
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
            'board_matrix': board_matrix
        })

    return game_info, step_records

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

if __name__ == "__main__":
    processed_files = 0
    valid_samples = 0

    # 创建HDF5文件 可扩展数据集
    with h5py.File(SAVE_PATH, 'w') as f:
        inputs_dset = f.create_dataset(
            'inputs',
            shape    = (0, 2, 19, 19),  
            maxshape = (None, 2, 19, 19),  # 可无限扩展
            dtype    = np.float32,
            chunks   = (1000, 2, 19, 19)   # 分块大小1000
        )
        labels_dset = f.create_dataset(
            'labels',
            shape    = (0,),
            maxshape = (None,),
            dtype    = np.int64,
            chunks   = (1000,)
        )

        # 遍历SGF文件，流式写入
        for root, dirs, files in os.walk(sgf_dataset_dir):
            for file in files:
                if not file.endswith('.sgf'):
                    continue
                sgf_path = os.path.join(root, file)
                
                # 每个SGF的样本的临时列表
                temp_boards = []
                temp_players = []
                temp_labels = []
                
                try:
                    with open(sgf_path, 'rb') as f_sgf:
                        sgf_content = f_sgf.read()
                    
                    game_info, step_records = read_go_sgf(sgf_content)
                    board_map = link_next_move(step_records)

                    # 生成当前SGF的样本
                    for cur_step, next_step in board_map:
                        if cur_step['board_matrix'] is None:
                            continue

                        # 训练输入
                        input_board = cur_step['board_matrix'].astype(np.float32)
                        next_player = next_step['color_code']
                        if next_player is None:
                            continue
                        next_player = COLOR_MAP.get(next_player)
                        
                        # 生成训练输出: 下一个落子位置 转化为一维索引 弃行为-1
                        next_pos = next_step['pos']
                        if next_pos is not None:
                            label_idx = next_pos[0] * game_info['board_size'] + next_pos[1]
                        else:
                            label_idx = -1
                        
                        # 加入临时列表
                        temp_boards.append(input_board)
                        temp_players.append(next_player)
                        temp_labels.append(label_idx)
                        valid_samples += 1

                    processed_files += 1
                    print(f"第{processed_files}个棋谱: {sgf_path}处理完成, 当前有效样本: {valid_samples}个")

                    # 写入HDF5
                    if len(temp_boards) > 0:
                        # 转换为numpy数组
                        temp_boards_np = np.array(temp_boards, dtype=np.float32)
                        temp_players_np = np.array(temp_players, dtype=np.float32)[:, np.newaxis, np.newaxis]
                        temp_players_np = np.tile(temp_players_np, (1, 19, 19))
                        temp_inputs_np = np.stack([temp_boards_np, temp_players_np], axis=1)
                        temp_labels_np = np.array(temp_labels, dtype=np.int64)
                        
                        # 扩展数据集并写入
                        current_size = inputs_dset.shape[0]
                        new_size = current_size + len(temp_inputs_np)
                        inputs_dset.resize((new_size, 2, 19, 19))
                        inputs_dset[current_size:new_size] = temp_inputs_np
                        labels_dset.resize((new_size,))
                        labels_dset[current_size:new_size] = temp_labels_np
                        
                        # 释放内存
                        temp_boards.clear()
                        temp_players.clear()
                        temp_labels.clear()
                        del temp_boards_np, temp_players_np, temp_inputs_np, temp_labels_np  

                except Exception as e:
                    print(f"处理文件{sgf_path}时出错: {e}")
                    temp_boards.clear()
                    temp_players.clear()
                    temp_labels.clear()
                    continue

    print(f"\n===== 训练数据生成完成 =====")
    print(f"总处理文件数：{processed_files}")
    print(f"有效训练样本数：{valid_samples}")
    print(f"数据保存路径：{os.path.abspath(SAVE_PATH)}")             