import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from sgfmill import boards
import math
from common import *
import strategy.go_strategy as go_sg
import random as rd

def calculate_board_value(board: boards.Board) -> tuple[float, float]:
    '''根据规则计算双方局面价值'''
    size = board.side
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _cal_manhattan_dis(pos1: tuple, pos2: tuple):
        '''计算棋盘上两个点之间的曼哈顿距离'''
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos1[1])
    
    def _cal_dis(pos1: tuple, pos2: tuple):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _cal_points(side: str):
        '''计算某一方的空点归属'''
        points_cnt = 0
        occ_poinmts, emptys =  board.list_occupied_points()
        for (er, ec) in emptys:
            b_min_dis = w_min_dis = 999  # 到黑棋白棋的最短距离
            for (color, (o_r, o_c)) in occ_poinmts:
                m_dis = _cal_dis((er, ec), (o_r, o_c))
                if m_dis >= 2:
                    continue
                if color == 'b':
                    if m_dis < b_min_dis:
                        b_min_dis = m_dis
                elif color == 'w':
                    if m_dis < w_min_dis:
                        w_min_dis = m_dis             
            if side == 'b':
                if b_min_dis < w_min_dis:
                    points_cnt += 1
            elif side == 'w':
                if w_min_dis < b_min_dis:
                    points_cnt += 1

        return points_cnt

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

        # if side == 'b':
        #     my_territory -= 6.5

        my_points = _cal_points(side)
        opp_points = _cal_points(opponent)

        if side == 'b':
          my_points -= 5 * 1.5

        # 加权
        total_score = (
            + 2.0 * my_territory
            + 0.5 * my_total_liberties
            + 0.5 * my_alive_groups
            + 0.5 * opp_dead_groups
            - 1.5 * opp_territory
            - 0.6 * opp_total_liberties
            + 1.5 * my_points
            - 0.5 * opp_points
        )

        return total_score
    
    b_value = _cal_single_value('b')
    w_value = _cal_single_value('w')

    # 结合相对价值信息
    value_sum = b_value + w_value
    if value_sum == 0:
        value_sum = 0.0001
    b_value = b_value / value_sum
    w_value = w_value / value_sum

    # 反正切归一化
    b_value = math.atan(b_value) / (math.pi / 2)
    w_value = math.atan(w_value) / (math.pi / 2)

    return (b_value, w_value)

def get_rollout_value(board: boards.Board, cur_color: chr, deepth):
    '''输出棋盘的一轮推演后的局面价值 下一个行棋方(cur_color)'''
    # 进行1次蒙特卡洛推演
    for i in range(deepth):
        candidates = go_sg.model_pred(board, cur_color)
        candidate = rd.choice(candidates)[0]
        if candidate == 'pass':
            continue
        nr, nc = candidate
        board.play(nr, nc, cur_color)
        cur_color = 'b' if cur_color == 'w' else 'w'
    return calculate_board_value(board)

def monte_carlo_rollout_value(board: boards.Board, cur_color: chr, times, deepth):
    '''输出棋盘经过蒙特卡洛推演后的平均局面价值 下一个行棋方(cur_color)'''
    b_total_value = w_total_value = 0
    for i in range(times):
        init_board = board.copy()
        b_rollout_value, w_rollout_value = get_rollout_value(init_board, cur_color, deepth)
        b_total_value += b_rollout_value
        w_total_value += w_rollout_value
        del init_board
    b_MCR_value = b_total_value / times
    w_MCR_value = w_total_value / times
    return (b_MCR_value, w_MCR_value)

if __name__ == '__main__':
    pass