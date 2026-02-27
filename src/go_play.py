import subprocess
import sys
import time
import json
from sgfmill import boards, ascii_boards
from strategy.go_strategy import GoStrategySelector
from value.go_value import GoValuePredictor
from collections import deque
from common import *
import pyautogui
import cv2
import numpy as np

class KataGoEngine:
    '''katago交互类'''
    def __init__(self):
        self.process = None
        self.request_id = 0
        self._start_engine()

    def _start_engine(self):
        try:
            self.process = subprocess.Popen(
                [
                    KATA_EXE_PATH,
                    'analysis',
                    '-model', KATA_MODEL_PATH,
                    '-config', KATA_CONFIG_PATH
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            ready = False
            for _ in range(300):
                line = self.process.stdout.readline()
                if not line:
                    continue
                if 'ready to begin handling requests' in line:
                    ready = True
                    break
            
            if not ready:
                raise RuntimeError('katago启动超时')
        
        except Exception as e:
            print(f'启动失败: {e}')
            sys.exit(1)
    
    def restart(self):
        self.close()
        self._start_engine()
        print('重启KataGo...')
    
    def close(self):
        if self.process:
            print('关闭KataGo...')
            self.process.terminate()
            self.process.wait()
    
    def get_value(self, root_player, moves):
        '''根据当前落子列表给出下一个行棋方的胜率'''
        self.request_id += 1
        request_id = str(self.request_id)
        request = {
            'id': request_id,
            'boardXSize': BOARD_SIZE,
            'boardYSize': BOARD_SIZE,
            'initialStones': [],
            'moves': moves,
            'rules': 'chinese',
            'komi': GAME_KOMI,
            'visits': 10,
            'includePolicy': False,
            'includeOwnership': False,
            'includeMovesOwnership': False
        }

        # 写入请求
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()

        response = None
        start_read = time.time()
        while time.time() - start_read < 5:  # 5秒超时
            line = self.process.stdout.readline()
            if not line:
                continue
            
            kata_message = line.strip()
            print(f'[KataGo] {kata_message}')
            if ('error' or 'Error' in kata_message) and 'rawStWrError' not in kata_message:
                print(f"KataGo返回错误: {kata_message}")
                self.restart()
                return -2
            try:
                resp = json.loads(line)
                if 'error' in resp:
                    print(f"KataGo返回错误: {resp['error']}")
                    self.restart()
                    return -2
                if 'rootInfo' not in resp:
                    continue
                response = resp
                break
            except json.JSONDecodeError:
                continue
        
        if not response:
            print('KataGo响应超时')
            return -1
        
        # 解析结果
        player = 'b' if len(moves) % 2 == 0 else 'w'  # 下一个行棋方
        winrate = response['rootInfo']['winrate']
        print(f'我方胜率: {100*(1-winrate):.2f}%\n')

        return winrate if root_player == player else (1-winrate)

class TreeNode:
    '''MiniMax 树节点'''
    def __init__(self, board_state, cur_player:str, ko_pos = None, deepth=0, 
                 parent=None, by_move=None, is_root=False, root_moves=None):
        self.board_state: boards.Board = board_state.copy()  # 当前棋盘状态
        self.cur_player  = cur_player          # 当前落子方
        self.ko_pos      = ko_pos              # 劫争位
        self.deepth      = deepth              # 节点深度
        self.parent: TreeNode = parent         # 父节点
        self.by_move     = by_move             # 通过何种落子到达该节点
        self.children    = []                  # 子节点列表
        self.value       = None                # 价值

        # 根节点属性
        self.is_root     = is_root
        self.root_moves  = root_moves          # 当前根节点所有落子序列

        # 叶节点属性
        self.moves       = None                # 到达这个叶节点的所有落子序列
    
    def is_leaf(self):
        return self.deepth >= MAX_SEARCH_DEEPTH

    def get_moves_to_leaf(self):
        # 得到到达这个叶节点的所有落子序列
        t_moves = []
        cur_node = self
        while not cur_node.is_root:
            t_moves.append([
                cur_node.parent.cur_player.upper(), 
                idx_to_go_str((cur_node.by_move[0], cur_node.by_move[1]), False)
            ])
            cur_node = cur_node.parent
        t_moves = t_moves[::-1]
        self.moves = cur_node.root_moves + t_moves
        del t_moves

    
    def is_fully_expanded(self):
        '''是否已经扩展所有top_k个子节点'''
        return len(self.children) >= TOP_K
    
class MinimaxMCR:
    '''Minimax搜索树和蒙特卡洛推演算法实现类'''
    def __init__(self, steps_list,  # 落子序列
                 root_state: boards.Board, 
                 root_player: chr, 
                 curstep: int,
                 ko_pos = None  # 根节点劫争位
                 ):
        self.sg_selector = GoStrategySelector()  # 策略选择器
        self.va_predictor = GoValuePredictor()   # 价值选择器
        self.va_engine = KataGoEngine()          # katago价值判断器
        self.root_state = root_state.copy()      # 根节点棋盘状态
        self.root_player = root_player           # 根节点行棋方
        self.opp_player = 'b' if self.root_player == 'w' else 'w'
        self.cursteps = curstep                  # 双方走的步数
        self.steps_list = steps_list             # 双方当前局面落子记录
        self.root = None                         # 搜索树根节点
        self.ko_pos = ko_pos                     # 根节点劫争位
        self.leaves = []                         # 所有叶节点列表

    def build_tree(self):
        '''从根节点开始按照MAX_SEARCH_DEEPTH构建搜索树'''
        self.root = TreeNode(
            self.root_state, 
            self.root_player, 
            ko_pos=self.ko_pos,
            deepth=0, 
            is_root=True, 
            root_moves=self.steps_list
        )
        queue = deque([self.root])

        def _is_legal(pos, node: TreeNode):
            '''
            判断该情况下落子是否合法
                落子位置是上一次劫争位
                落子位置已经有棋子
                落子位置导致自己棋子被提
            '''
            pos = tuple(pos)
            # 判断是否已经有棋子
            if node.board_state.get(pos[0], pos[1]) is not None:
                return False
            
            # 判断是否上一次劫争位
            # print(f'判断合法性: 落子 {pos} | 当前节点劫争位: {node.ko_pos}')
            if node.ko_pos is not None and pos == node.ko_pos:
                return False
            
            # 判断落子后是否自己棋子被提
            test_board = node.board_state.copy()
            _, to_be_captured = test_board.play(pos[0], pos[1], node.cur_player)
            if pos in to_be_captured:
                return False

            return True

        while queue:
            node: TreeNode = queue.popleft()

            if node.is_leaf():  # 遇到叶节点，反推构建落子步骤
                self.leaves.append(node)
                node.get_moves_to_leaf()
                continue

            topk_moves = self.sg_selector.predict(node.board_state, node.cur_player)
            for move, prob in topk_moves:
                if move == 'pass':
                    continue
                if not _is_legal(move, node):
                    continue
                child_board = node.board_state.copy()
                ko_pos, _ = child_board.play(move[0], move[1], node.cur_player)
                new_player = 'b' if node.cur_player == 'w' else 'w'
                child = TreeNode(
                    child_board,
                    new_player,
                    ko_pos=ko_pos,
                    deepth=node.deepth + 1,
                    parent=node,
                    by_move=move
                )
                node.children.append(child)
                queue.append(child)
        return self.root, self.leaves
    
    def _rollout_leaf(self, leaf_node: TreeNode):
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
    
    def rollout_leaf_kata(self, leaf_node: TreeNode):
        '''使用katago计算叶节点的局面价值'''
        print(f'叶节点: {leaf_node} | 落子序列: {leaf_node.moves}')
        return self.va_engine.get_value(self.root_player, leaf_node.moves)
    
    def evaluate_all_leaves(self):
        '''预测所有叶节点的价值'''
        for leaf in self.leaves:
            leaf.value = self.rollout_leaf_kata(leaf)
    
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
        self.steps = []  # 记录落子信息
        self.ko_pos = None # 记录劫争位

    def play_move_str(self, go_str:str):
        """执行落子 字符坐标"""
        if go_str == "pass":
            print(f"{self.current_color.upper()} 弃行")
        else:
            pos = go_str_to_idx(go_str)
            self.last_board = self.board.copy()
            self.last_color = self.current_color
            self.ko_pos, _ = self.board.play(pos[0], pos[1], self.current_color)
            print(f"{self.current_color.upper()} 落子：({pos[0]}, {pos[1]})")
            self.steps.append([self.current_color.upper(), idx_to_go_str((pos[0], pos[1]), False)])
        
        # 切换行棋方
        self.current_color = 'w' if self.current_color == 'b' else 'b'
        self.curstep += 1
    
    def play_move(self, go_pos:tuple):
        '''执行落子 元组坐标, 返回劫争位'''
        self.ko_pos, _ = self.board.play(go_pos[0], go_pos[1], self.current_color)
        self.steps.append([self.current_color.upper(), idx_to_go_str((go_pos[0], go_pos[1]), False)])
        self.current_color = 'w' if self.current_color == 'b' else 'b'
        self.curstep += 1

    def print_board(self):
        print("当前棋盘状态")
        print(ascii_boards.render_board(self.board, HAVE_I))

    def back_state(self):
        '''悔棋'''
        self.board = self.last_board.copy()
        self.current_color = self.last_color

class GoBoardDector:
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.cur_board_list = []
        self.last_board_list = []

    def get_board_img(self):
        '''对当前棋盘截图得到灰度图'''
        pil_img = pyautogui.screenshot(region=BOARD_REGION)
        gray_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        return gray_img
    
    def show_board_img(self, title, img, wait_ms=1000):
        cv2.imshow(title, img)
        key = cv2.waitKey(wait_ms)
        if wait_ms == 0:
            cv2.destroyAllWindows()
        return key
    
    def do_move(self, pos):
        self.cur_board_list.append(pos)
        pix_pos = (META_PLAY_POS[0] + pos[1] * GAP, META_PLAY_POS[1] - pos[0] * GAP)
        for _ in range(5):
            pyautogui.click(pix_pos[0], pix_pos[1])
            time.sleep(0.1)
        for _ in range(5):
            pyautogui.click(pix_pos[0], pix_pos[1])
            time.sleep(0.1)
        print('准备保存当前棋盘状态')
        self._save_last_board()

    def _analyze_black(self):
        '''分析得到黑棋位置'''
        _, b_thresh = cv2.threshold(  # 二值化提取黑棋特征图
            self.cur_img,
            BLACK_THRESH,
            255,
            cv2.THRESH_BINARY
        )

        # self.show_board_img('黑棋二值化特征图' ,b_thresh, 0)

        # 遍历棋盘格子位置 判断是否有黑棋
        for r in range(self.board_size):
            for c in range(self.board_size):
                pix_pos = (META_ANALYSYS_POS[0] + r * GAP, META_ANALYSYS_POS[1] + c * GAP)
                if b_thresh[pix_pos[0], pix_pos[1]] == 0:
                    self.cur_board_list.append((18 - r, c))
    
    def _analyze_white(self):
        '''分析得到白棋位置'''
        _, w_thresh = cv2.threshold(
            self.cur_img,
            WHITE_THRESH,
            255,
            cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        w_thresh = cv2.morphologyEx(w_thresh, cv2.MORPH_CLOSE, kernel)

        

        for r in range(self.board_size):
            for c in range(self.board_size):
                # 区域采样
                roi_ok = False
                roi = w_thresh[
                    META_ANALYSYS_POS[0] + r * GAP - WHITE_BIAS : META_ANALYSYS_POS[0] + r * GAP + WHITE_BIAS,
                    META_ANALYSYS_POS[1] + c * GAP - WHITE_BIAS : META_ANALYSYS_POS[1] + c * GAP + WHITE_BIAS
                ]
                roi_ok = np.any(roi == 255)
                if roi_ok:
                    self.cur_board_list.append((18 - r, c))
                    # go_str = idx_to_go_str((18 - r, c), False)
                    # print(f"【识别】白棋：{go_str} (r={r}, c={c})")
        
        # self.show_board_img('白棋二值化特征图' ,w_thresh, 0)
    
    def analyze_cur_board(self):
        '''对棋盘截图, 分析得到当前棋盘状态'''
        self.cur_img = self.get_board_img()

        # 分析得到黑棋白棋位置
        self._analyze_black()
        self._analyze_white()

    def _save_last_board(self):
        '''保存当前棋盘状态为上一个状态'''
        self.last_board_list = self.cur_board_list.copy()
        self.cur_board_list = []
    
    def cal_new_stone_pos(self):
        '''计算新落子位置'''
        for pos in self.cur_board_list:
            if pos not in self.last_board_list:
                return pos
        return None

def ai_play(steps_list, board: boards.Board, color, curstep, ko_pos=None):
    '''ai行棋'''
    minimax_search = MinimaxMCR(steps_list, board, color, curstep, ko_pos)
    if curstep % 10 == 0:  # 每10手重启一次katago
        minimax_search.va_engine.restart()
    best_move, best_value = minimax_search.search()
    print(f'ai选择落子: {idx_to_go_str(best_move)} | value: {best_value:.4f}')
    return best_move

def vs_ai():
    go_player = GoPlayer('b')
    print("* 落子输入格式：列字符+行坐标 J8 ")
    
    if go_player.ai_player == 'b':  # 若黑 ai先下
        move = ai_play(
            go_player.steps, 
            go_player.board, 
            'b', 
            go_player.curstep, 
            ko_pos=None
        )
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
            print(f'当前落子序列: {go_player.steps}')
        
        # AI落子
        ai_move = ai_play(
            go_player.steps, 
            go_player.board, 
            go_player.current_color, 
            go_player.curstep, 
            go_player.ko_pos
        )
        if ai_move == 'pass':
            print("AI选择弃行")
            go_player.play_move_str('pass')
        else:
            go_player.play_move(ai_move)

def vs_ai_auto():
    go_player = GoPlayer('w')
    board_dector = GoBoardDector()
    board_dector.analyze_cur_board()
    print('模型加载完成')

    if go_player.ai_player == 'b':
        move = ai_play(
            go_player.steps, 
            go_player.board, 
            'b', 
            go_player.curstep, 
            ko_pos=None
        )
        go_player.play_move(move)
        board_dector.do_move(move)

    while True:
        # 人类落子
        board_dector.analyze_cur_board()
        new_stone_pos = board_dector.cal_new_stone_pos()
        if new_stone_pos is None:
            time.sleep(0.5)
            continue

        print(f'\n新落子: {idx_to_go_str(new_stone_pos)}')
        go_player.play_move_str(idx_to_go_str(new_stone_pos))
        go_player.print_board()
        print(f'当前落子序列: {go_player.steps}')

        # AI落子
        ai_move = ai_play(
            go_player.steps, 
            go_player.board, 
            go_player.current_color, 
            go_player.curstep,
            go_player.ko_pos
        )
        go_player.play_move(ai_move)
        board_dector.do_move(ai_move)

if __name__ == "__main__":
    vs_ai_auto()