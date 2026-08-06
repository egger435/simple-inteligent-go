'''AI 对战测试：任意两种模式互搏。

模式:
  A: Minimax + 自训练价值网络 (top_k=4, depth=5, MC 5×5)
  B: Minimax + KataGo (top_k=3, depth=3)
  C: MCTS + 自训练价值网络 (100 模拟)
  D: MCTS + KataGo (100 模拟)

用法:
  python test_battle.py                  # 默认 A vs B
  python test_battle.py --left C --right D   # MCTS+Katago vs MCTS+Katago
'''

import sys, os, time, argparse
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from sgfmill import boards

from common import BOARD_SIZE, GAME_KOMI, idx_to_go_str
from game.tree import search_move
from engine.katago import get_katago_engine

# ===== 参数 =====
N_GAMES = 10
MAX_STEPS = 100
KOMI = 7.5

# 模式定义
MODES = {
    'A': dict(
        label='Minimax+价值网络',
        algorithm='minimax', use_own_value_net=True,
        n_rollouts=5, n_steps=5, top_k=4, max_depth=5,
    ),
    'B': dict(
        label='Minimax+KataGo',
        algorithm='minimax', use_own_value_net=False,
        top_k=3, max_depth=3,
    ),
    'C': dict(
        label='MCTS+价值网络',
        algorithm='mcts', use_own_value_net=True,
        simulations=100, c_puct=1.4,
    ),
    'D': dict(
        label='MCTS+KataGo',
        algorithm='mcts', use_own_value_net=False,
        simulations=100, c_puct=1.4,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description='AI 对战测试')
    p.add_argument('--left', '-l', choices=list(MODES), default='A',
                   help='左方模式 (默认 A)')
    p.add_argument('--right', '-r', choices=list(MODES), default='B',
                   help='右方模式 (默认 B)')
    p.add_argument('--games', '-n', type=int, default=N_GAMES,
                   help=f'对局数 (默认 {N_GAMES})')
    p.add_argument('--steps', '-s', type=int, default=MAX_STEPS,
                   help=f'每局最多手数 (默认 {MAX_STEPS})')
    return p.parse_args()


# =====================================================================
def ai_move(color, steps, board, params):
    '''走一步棋，返回 (move_or_pass, value)。'''
    if params['algorithm'] == 'mcts':
        return search_move(
            steps, board, color, len(steps),
            algorithm='mcts',
            use_own_value_net=params.get('use_own_value_net', True),
            simulations=params.get('simulations', 100),
            c_puct=params.get('c_puct', 1.4),
            verbose=False,
        )
    else:
        return search_move(
            steps, board, color, len(steps),
            algorithm='minimax',
            use_own_value_net=params.get('use_own_value_net'),
            n_rollouts=params.get('n_rollouts'),
            n_steps=params.get('n_steps'),
            top_k=params['top_k'],
            max_depth=params['max_depth'],
            verbose=False,
        )


def play_one_game(kata, left_params, right_params,
                  left_key, right_key, left_is_black, game_idx, max_steps):
    '''进行一局对战，返回 (left_won, right_won, b_winrate)。'''
    board = boards.Board(BOARD_SIZE)
    steps = []
    current = 'b'
    left_color = 'b' if left_is_black else 'w'
    right_color = 'w' if left_is_black else 'b'

    for step in range(max_steps):
        if current == left_color:
            params, who = left_params, left_key
        else:
            params, who = right_params, right_key

        move, value = ai_move(current, steps, board, params)

        # 打印对局进度
        print(f'  第{game_idx+1}场 | {who}方 | 第{step+1}手 | '
              f'{idx_to_go_str(move) if move != "pass" else "pass"} | '
              f'胜率: {value*100:.1f}%', flush=True)

        if move == 'pass':
            steps.append([current.upper(), 'pass'])
        else:
            try:
                board.play(move[0], move[1], current)
            except ValueError:
                steps.append([current.upper(), 'pass'])
                current = 'w' if current == 'b' else 'b'
                continue
            steps.append([current.upper(), idx_to_go_str(move, have_i=False)])

        current = 'w' if current == 'b' else 'b'

    # 终局判定（KataGo 静默批量）
    b_winrate = kata.get_value_batch([('b', steps)])[0]
    if b_winrate < 0:
        b_winrate = 0.5

    black_won = b_winrate > 0.5
    if left_is_black:
        left_won, right_won = black_won, not black_won
    else:
        left_won, right_won = not black_won, black_won

    return left_won, right_won, b_winrate


# =====================================================================
def main():
    args = parse_args()
    left_key = args.left
    right_key = args.right
    left_params = MODES[left_key]
    right_params = MODES[right_key]

    print('=' * 60)
    print(f'AI 对战测试: {left_key}({left_params["label"]}) vs '
          f'{right_key}({right_params["label"]})')
    print('=' * 60)
    print(f'  对局数: {args.games} | 每场最多 {args.steps} 手')
    print()

    kata = get_katago_engine()
    left_wins = right_wins = draws = 0

    for game_idx in range(args.games):
        left_is_black = (game_idx % 2 == 0)   # 黑白交替
        t0 = time.time()

        lw, rw, b_winrate = play_one_game(
            kata, left_params, right_params,
            left_key, right_key, left_is_black, game_idx, args.steps,
        )

        if lw and not rw:
            winner = left_key
            left_wins += 1
        elif rw and not lw:
            winner = right_key
            right_wins += 1
        else:
            winner = '平局'
            draws += 1

        elapsed = time.time() - t0
        print(f'  局 {game_idx+1}/{args.games} | '
              f'{left_key} 执{"黑" if left_is_black else "白"} | '
              f'胜者: {winner} | 黑胜率: {b_winrate:.2%} | {elapsed:.0f}s')

    print()
    print('=' * 60)
    print(f'  最终: {left_key} {left_wins} 胜 | {right_key} {right_wins} 胜 | '
          f'{draws} 平')
    print('=' * 60)

    kata.close()


if __name__ == '__main__':
    main()
