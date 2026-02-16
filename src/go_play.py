import torch
from sgfmill import boards, ascii_boards
from strategy.go_strategy import GoStrategySelector
from value.go_value import GoValuePredictor
from collections import deque
from common import *

class TreeNode:
    '''MiniMax 树节点'''
    def __init__(self, board_state: boards.Board, cur_player, deepth=0, parent=None, by_move=None):
        self.board_state = board_state.copy()  # 当前棋盘状态
        self.cur_player  = cur_player          # 当前落子方
        self.deepth      = deepth              # 节点深度
        self.parent      = parent              # 父节点
        self.by_move     = by_move             # 通过何种落子到达该节点
        self.children    = []                  # 子节点列表
        self.value       = None                # 价值
    
    def is_leaf(self):
        return self.deepth >= MAX_SEARCH_DEEPTH
    
    def is_fully_expanded(self):
        '''是否已经扩展所有top_k个子节点'''
        return len(self.children) >= TOP_K
    
class MinimaxMCR:
    '''Minimax搜索树和蒙特卡洛推演算法实现类'''
    def __init__(self, root_state: boards.Board, root_player: chr, curstep: int):
        self.sg_selector = GoStrategySelector()  # 策略选择器
        self.va_predictor = GoValuePredictor()   # 价值选择器
        self.root_state = root_state.copy()
        self.root_player = root_player           # 根节点行棋方
        self.opp_player = 'b' if self.root_player == 'w' else 'w'
        self.cursteps = curstep                  # 双方走的步数
        self.root = None
        self.leaves = []
    
    def build_tree(self):
        '''从根节点开始按照MAX_SEARCH_DEEPTH构建搜索树'''
        self.root = TreeNode(self.root_state, self.root_player, deepth=0)
        queue = deque([self.root])

        while queue:
            node: TreeNode = queue.popleft()

            if node.is_leaf():
                self.leaves.append(node)
                continue

            topk_moves = self.sg_selector.predict(node.board_state, node.cur_player)
            for move, prob in topk_moves:
                if move == 'pass':
                    continue
                node.board_state.play(move[0], move[1], node.cur_player)
                new_player = 'b' if node.cur_player == 'w' else 'w'
                child = TreeNode(
                    node.board_state,
                    new_player,
                    deepth=node.deepth + 1,
                    parent=node,
                    by_move=move
                )
                node.children.append(child)
                queue.append(child)
        return self.root, self.leaves
    
    def rollout_leaf(self, leaf_node: TreeNode):
        '''从叶节点开始推演, 返回终局价值 根节点cur_player视角'''
        if self.cursteps <= MC_START_THRESHOLD:
            val_tuple = self.va_predictor.get_rollout_value(
                leaf_node.board_state,
                self.opp_player,
                GAME_KOMI,
                rollout_deepth=280-self.cursteps
            )
        else:
            val_tuple, win_rate_tuple = (
                self.va_predictor.get_monte_carlo_rollout_value(
                    leaf_node.board_state,
                    self.opp_player,
                    GAME_KOMI,
                    rollout_deepth=280-self.cursteps,
                    times=MC_SIMULATIONS
                )
            )
        print(f'叶节点: {leaf_node} | value: ({val_tuple[0]:.4f}, {val_tuple[1]:.4f})')
        return val_tuple[0] if self.root_player == 'b' else val_tuple[1]
    
    def evaluate_all_leaves(self):
        '''预测所有叶节点的价值'''
        for leaf in self.leaves:
            leaf.value = self.rollout_leaf(leaf)
    
    def minimax_backup(self):
        '''从叶节点开始, 按Minimax规则回溯更新所有节点的价值'''
        def post_order(node: TreeNode):
            '''后序遍历 先处理子节点'''
            if node.is_leaf():
                return node.value
            child_values = [post_order(child) for child in node.children]

            if node.cur_player == self.root.cur_player:
                # 与根节点行棋方相同, max层
                node.value = max(child_values)
            else:
                node.value = min(child_values)
            
            return node.value
        
        post_order(self.root)
    
    def select_best_move(self):
        '''选择价值最大的落子'''
        if not self.root.children:
            return 'pass'
        best_child = max(self.root.children, key=lambda c: c.value)
        return best_child.by_move, best_child.value
    
    def search(self):
        '''算法完整流程'''
        self.build_tree()
        print(f"minimax树构建完成 共 {len(self.leaves)} 个叶节点")

        self.evaluate_all_leaves()
        self.minimax_backup()
        best_move, best_value = self.select_best_move()
        print(f"最优落子: {best_move} | value: {best_value:.4f}")

        return best_move, best_value

class GoPlayer:
    def __init__(self, ai_player='b'):
        self.init_player = 'b'
        self.ai_player = ai_player
        # 初始化棋盘
        self.board = boards.Board(BOARD_SIZE)
        self.current_color = 'b'

        # 记录上一个状态
        self.last_board = self.board.copy()
        self.last_color = self.current_color

        self.curstep = 0

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
        self.curstep += 1
    
    def play_move(self, go_pos:tuple):
        '''执行落子 元组坐标'''
        self.board.play(go_pos[0], go_pos[1], self.current_color)
        self.current_color = 'w' if self.current_color == 'b' else 'b'
        self.curstep += 1

    def print_board(self):
        print("当前棋盘状态")
        print(ascii_boards.render_board(self.board, HAVE_I))

    def back_state(self):
        '''悔棋'''
        self.board = self.last_board.copy()
        self.current_color = self.last_color

def ai_play(board: boards.Board, color, curstep):
    '''ai行棋'''
    minimax_search = MinimaxMCR(board, color, curstep)
    best_move, best_value = minimax_search.search()
    print(f'ai选择落子: {idx_to_go_str(best_move)} | value: {best_value:.4f}')
    return best_move

def vs_ai():
    go_player = GoPlayer('w')
    print("* 落子输入格式：列字符+行坐标 J8 | 输入 pass 弃行 | 输入 back 悔棋")
    
    if go_player.ai_player == 'b':  # 若黑 ai先下
        move = ai_play(go_player.board, 'b', go_player.curstep)
        go_player.play_move(move)

    while True:
        # 人类落子
        go_player.print_board()
        human_input = input("落子位置: ")
        if human_input == "pass":
            go_player.play_move_str("pass")
        elif human_input == 'back':
            go_player.back_state()
            continue
        else:
            go_player.play_move_str(human_input)
        
        # AI落子
        ai_move = ai_play(go_player.board, go_player.current_color, go_player.curstep)
        if ai_move == 'pass':
            print("AI选择弃行")
            go_player.play_move_str('pass')
        else:
            go_player.play_move(ai_move)

if __name__ == "__main__":
    vs_ai()