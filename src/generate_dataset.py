import numpy as np
from sgfmill import sgf, boards
import os
import math

# 总处理文件数：76771
# 有效训练样本数：16133673
# 数据保存路径：E:\go_dataset\go_dataset.npz

sgf_dataset_dir  = 'data\gogod_commentary_sgfs'
SAVE_ROOT_PATH   = 'E:/go_dataset'
CHUNK_SAMPLE_NUM = 500000    # 每50万个样本写入一个npz文件

COLOR_MAP = {1: 0.0, 2: 1.0}  # 训练模型时候将行棋方编码转换为0(黑棋)和1(白棋)

def calculate_board_value(board: boards.Board) -> tuple[float, float]:
    '''根据规则计算双方局面价值'''
    size = board.side
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _find_all_groups(side: str):
        '''找到某一方所有棋块'''
        visited = set()  
        all_groups = []  # 存储所有棋块
        
        for r in range(size):
            for c in range(size):
                pos = (r, c)
                if pos not in visited and board.get(r, c) == side:
                    # BFS找连通的棋块
                    group = set()
                    queue = [pos]
                    visited.add(pos)
                    
                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        group.add((curr_r, curr_c))
                        
                        for dr, dc in dirs:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < size and 0 <= nc < size:
                                n_pos = (nr, nc)
                                # 相邻是同色且未访问 加入当前棋块
                                if n_pos not in visited and board.get(nr, nc) == side:
                                    visited.add(n_pos)
                                    queue.append(n_pos)
                    all_groups.append(group)
        return all_groups

    def _cal_group_liberties(group):
        '''计算棋块的气数'''
        liberties = set()
        for (r, c) in group:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if board.get(nr, nc) is None:
                        liberties.add((nr, nc))
        return len(liberties)

    def _cal_territory(side: str) -> int:
        """计算某一方活棋块围起来的空"""
        visited_empty = set()  
        territory = 0  

        # 遍历所有空点，判断是否属于该方的围空
        for r in range(size):
            for c in range(size):
                pos = (r, c)
                if pos not in visited_empty and board.get(r, c) is None:
                    # BFS遍历该空点的连通区域
                    empty_area = set()
                    queue = [pos]
                    visited_empty.add(pos)
                    belongs_to = None  # 该空区属于哪一方 side/opponent/None

                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        empty_area.add((curr_r, curr_c))
                        # 检查该空点的相邻棋子
                        adjacent_sides = set()
                        for dr, dc in dirs:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < size and 0 <= nc < size:
                                stone = board.get(nr, nc)
                                if stone is not None:
                                    adjacent_sides.add(stone)

                        # 判断该空区归属
                        if len(adjacent_sides) == 1:
                            single_side = adjacent_sides.pop()
                            if belongs_to is None:
                                belongs_to = single_side
                            elif belongs_to != single_side:
                                belongs_to = None  # 混合相邻，公共空
                        else:
                            belongs_to = None  # 无相邻棋子/多色相邻，公共空

                        # 继续遍历连通的空点
                        for dr, dc in dirs:
                            nr, nc = curr_r + dr, curr_c + dc
                            n_pos = (nr, nc)
                            if 0 <= nr < size and 0 <= nc < size and n_pos not in visited_empty and board.get(nr, nc) is None:
                                visited_empty.add(n_pos)
                                queue.append(n_pos)

                    # 统计归属当前方的空区大小
                    if belongs_to == side:
                        territory += len(empty_area)

        return territory

    def _cal_single_value(side: str) -> float:
        '''计算某一方的局面价值'''
        opponent = 'w' if side == 'b' else 'b'

        # 找到双方所有棋块
        my_groups = _find_all_groups(side)
        opp_groups = _find_all_groups(opponent)

        # 计算相关指标
        my_total_liberties = sum(_cal_group_liberties(g) for g in my_groups)
        opp_total_liberties = sum(_cal_group_liberties(g) for g in opp_groups)
        my_alive_groups = sum(len(g) for g in my_groups if _cal_group_liberties(g) >= 2)
        opp_dead_groups = sum(len(g) for g in opp_groups if _cal_group_liberties(g) < 2)

        my_territory = _cal_territory(side)
        opp_territory = _cal_territory(opponent)

        # 加权
        total_score = (
            + 2.0 * my_territory
            + 1.0 * my_total_liberties
            + 0.8 * my_alive_groups
            + 1.0 * opp_dead_groups
            - 1.5 * opp_territory
            - 1.0 * opp_total_liberties
        )

        return total_score
    
    b_value = _cal_single_value('b')
    w_value = _cal_single_value('w')

    # 结合相对价值信息
    value_sum = b_value + w_value
    if value_sum == 0:
        value_sum = 0.001
    b_value = b_value / value_sum
    w_value = w_value / value_sum

    # 反正切归一化
    b_value = math.atan(b_value) / (math.pi / 2)
    w_value = math.atan(w_value) / (math.pi / 2)

    return (b_value, w_value)

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
    for idx, node in enumerate(main_sequence, 1):  # 跳过根节点
        move = node.get_move()
        if not move:  # 空操作判断
            step_records.append({
                'step':         idx,   # 步数
                'color_code':   None,  # 行棋方颜色代码 1黑棋, 2白棋
                'pos':          None,  # 落子坐标 元组
                'pos_str':      None,  # 落子坐标字符串
                'board_matrix': None,  # 当前棋盘矩阵
                'value':        None   # 当前双方局面价值
            })
            continue
        color, pos = move
        color_code = 1 if color == 'b' else 2  
        pos_str = f'({pos[0]}, {pos[1]})' if pos else 'pass'  # 落子坐标字符串
        if pos:
            board.play(pos[0], pos[1], color)
        
        # 计算双方局面价值
        board_value = np.array(calculate_board_value(board), dtype=np.float32)

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
            'value':        board_value
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
    chunk_index = 0

    # 全局临时列表
    global_boards  = []  # 棋盘通道
    global_players = []  # 下一位行棋方
    global_labels  = []  # 下一手的标签
    global_values  = []  # 局面价值

    # 流式解析sgf并拆分保存
    for root, dirs, files in os.walk(sgf_dataset_dir):
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

                    # 局面价值网络训练输出: 当前黑白双方局面价值
                    board_value = cur_step['value'].astype(np.float32)
                    
                    # 加入全局临时列表
                    global_boards.append(input_board)
                    global_players.append(next_player)
                    global_labels.append(label_idx)
                    global_values.append(board_value)
                    valid_samples += 1

                processed_files += 1
                print(f"第{processed_files}个棋谱: {sgf_path} | 累计有效样本: {valid_samples}")

                # 当全局样本数达到阈值，保存为一个npz文件
                if len(global_boards) >= CHUNK_SAMPLE_NUM:
                    chunk_path = os.path.join(SAVE_ROOT_PATH, f'go_dataset_chunk_{chunk_index}.npz')
                    boards_np = np.array(global_boards, dtype=np.float32)
                    players_np = np.array(global_players, dtype=np.float32)[:, np.newaxis, np.newaxis]
                    players_np = np.tile(players_np, (1, 19, 19))
                    inputs_np = np.stack([boards_np, players_np], axis=1)
                    labels_np = np.array(global_labels, dtype=np.int64)
                    values_np = np.array(global_values, dtype=np.float32)
                    
                    np.savez(
                        chunk_path,
                        inputs=inputs_np,
                        labels=labels_np,
                        values=values_np
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
        chunk_path = os.path.join(SAVE_ROOT_PATH, f'go_dataset_chunk_{chunk_index}.npz')
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

    print(f"\n===== 训练数据生成完成 =====")
    print(f"总处理SGF文件数：{processed_files}")
    print(f"总有效训练样本数：{valid_samples}")
    print(f"拆分文件总数：{chunk_index + 1}")
    print(f"文件保存路径：{os.path.abspath(SAVE_ROOT_PATH)}")            