import numpy as np
import os
from sgfmill import boards
import strategy.go_strategy as go_sg
import value.go_value as go_val
from common import *

def generate_strategy_model_dataset():
    '''
    策略网络训练数据生成:
        总处理棋谱数:   76,771
        有效训练样本数: 16,133,673
    '''
    processed_files = valid_samples = chunk_index = 0
    
    # 全局临时列表
    global_boards  = []  # 棋盘通道
    global_players = []  # 下一位行棋方
    global_labels  = []  # 下一手的标签

    # 流式解析sgf并拆分保存
    for root, dirs, files in os.walk(SGF_DATASET_DIR):
        for file in files:
            if not file.endswith('.sgf'):
                continue
            sgf_path = os.path.join(root, file)
            
            try:
                with open(sgf_path, 'rb') as f_sgf:
                    sgf_content = f_sgf.read()
                
                game_info, step_records = read_go_sgf(sgf_content)
                board_map = link_next_move(step_records)

                # 生成当前SGF的样本
                for cur_step, next_step in board_map:
                    if cur_step['board_matrix'] is None or next_step['color_code'] is None:
                        continue

                    # 训练输入：棋盘矩阵 + 行棋方
                    input_board = cur_step['board_matrix'].astype(np.float32)
                    next_player = COLOR_MAP[next_step['color_code']]
                    
                    # 策略选择网络训练输出：落子位置（一维索引），弃行=-1
                    next_pos = next_step['pos']
                    if next_pos is not None:
                        label_idx = next_pos[0] * game_info['board_size'] + next_pos[1]
                    else:
                        label_idx = -1
                    
                    # 加入全局临时列表
                    global_boards.append(input_board)
                    global_players.append(next_player)
                    global_labels.append(label_idx)
                    valid_samples += 1

                processed_files += 1
                print(f"第{processed_files}个棋谱: {sgf_path} | 累计有效样本: {valid_samples}")

                # 当全局样本数达到阈值，保存为一个npz文件
                if len(global_boards) >= SG_CHUNK_SAMPLE_NUM:
                    chunk_path = os.path.join(SG_DATASET_SAVE_PATH, f'go_dataset_chunk_{chunk_index}.npz')
                    boards_np = np.array(global_boards, dtype=np.float32)
                    players_np = np.array(global_players, dtype=np.float32)[:, np.newaxis, np.newaxis]
                    players_np = np.tile(players_np, (1, 19, 19))
                    inputs_np = np.stack([boards_np, players_np], axis=1)
                    labels_np = np.array(global_labels, dtype=np.int64)
                    
                    np.savez(
                        chunk_path,
                        inputs=inputs_np,
                        labels=labels_np,
                    )
                    print(f"保存拆分文件：{chunk_path} | 样本数：{len(inputs_np)}")
                    
                    # 重置全局列表，释放内存
                    global_boards.clear()
                    global_players.clear()
                    global_labels.clear()
                    del boards_np, players_np, inputs_np, labels_np
                    chunk_index += 1

            except Exception as e:
                print(f"处理文件{sgf_path}出错: {e}")
                continue

    # 处理最后一批样本
    if len(global_boards) > 0:
        chunk_path = os.path.join(SG_DATASET_SAVE_PATH, f'go_dataset_chunk_{chunk_index}.npz')
        boards_np = np.array(global_boards, dtype=np.float32)
        players_np = np.array(global_players, dtype=np.float32)[:, np.newaxis, np.newaxis]
        players_np = np.tile(players_np, (1, 19, 19))
        inputs_np = np.stack([boards_np, players_np], axis=1)
        labels_np = np.array(global_labels, dtype=np.int64)
        
        np.savez(chunk_path, inputs=inputs_np, labels=labels_np)
        print(f"保存最后一批拆分文件：{chunk_path} | 样本数：{len(inputs_np)}")
        
        global_boards.clear()
        global_players.clear()
        global_labels.clear()
        del boards_np, players_np, inputs_np, labels_np

    print(f"\n===== 策略网络训练数据生成完成 =====")
    print(f"总处理SGF文件数 {processed_files}")
    print(f"总有效训练样本数 {valid_samples}")
    print(f"拆分文件总数 {chunk_index + 1}")
    print(f"文件保存路径 {os.path.abspath(SG_DATASET_SAVE_PATH)}")

def generate_final_val_model_dataset():
    '''
    终局价值网络数据生成
    '''
    processed_files = 0

    # 全局临时列表
    global_boards = []  # 终局棋盘
    global_vals   = []  # 终局价值

    for root, dirs, files in os.walk(SGF_DATASET_DIR):
        for file in files:
            if not file.endswith('.sgf'):
                continue
            sgf_path = os.path.join(root, file)

            try:
                with open(sgf_path, 'rb') as f_sgf:
                    sgf_content = f_sgf.read()

                np_final_board, fianl_val = get_final_from_sgf(sgf_content)
                if fianl_val is None:
                    continue
                global_boards.append(np_final_board)
                global_vals.append(fianl_val)

                processed_files += 1
                print(f"第{processed_files}个棋谱: {sgf_path} 处理完成")
            
            except Exception as e:
                print(f"处理文件 {sgf_path} 出错: {e}")
                continue
    
    boards_np = np.array(global_boards, dtype=np.float32)
    values_np = np.array(global_vals, dtype=np.float32)

    np.savez_compressed(
        VA_DATASET_SAVE_PATH,
        inputs=boards_np,
        values=values_np
    )

    print(f"保存终局价值网络数据集：{VA_DATASET_SAVE_PATH} | 样本数：{len(boards_np)}")


if __name__ == "__main__":
    generate_final_val_model_dataset()