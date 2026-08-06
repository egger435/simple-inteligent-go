'''双模式 AI 的 ELO 棋力评测。

用 KataGo GTP -level 作基准对手，测两种模式的胜率，推导 ELO。

A 模式: 自训练价值网络 (top_k=4, depth=5, MC 5×5)
B 模式: KataGo 叶节点评估 (top_k=3, depth=3)

用法: python eval_elo.py
'''

import sys, os, math, time
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from sgfmill import boards

from common import BOARD_SIZE, idx_to_go_str, go_str_to_idx
from game.tree import search_move
from engine.katago_gtp import KataGoGTPClient

# ===== 参数 =====
N_GAMES = 10            # 每档每模式对局数
MAX_STEPS = 60         # 每局最多手数
# KataGo 基准档位（用 maxVisits 控制棋力，visits 越多越强）
# visits 档位 → 参考棋力 ELO（经验值，随模型版本浮动）
VISIT_LEVELS = [1, 10, 100]     # 弱 / 中 / 强
LEVEL_ELO = {1: 800, 10: 1300, 100: 1800}

MODES = {
    'A_value': {
        'label': '自训练价值网络',
        'algorithm': 'minimax',
        'use_own_value_net': True,
        'top_k': 4, 'max_depth': 5,
        'n_rollouts': 5, 'n_steps': 5,
    },
    'B_kata': {
        'label': 'KataGo评估',
        'algorithm': 'minimax',
        'use_own_value_net': False,
        'top_k': 3, 'max_depth': 3,
    },
    'C_mcts': {
        'label': 'MCTS',
        'algorithm': 'mcts',
        'use_own_value_net': True,
        'simulations': 100, 'c_puct': 1.4,
    },
}


# =====================================================================
def ai_move(color, steps, board, params):
    '''AI 走一步，返回 (move_or_pass, value)。'''
    algorithm = params.get('algorithm', 'minimax')
    return search_move(
        steps, board, color, len(steps),
        algorithm=algorithm,
        use_own_value_net=params.get('use_own_value_net'),
        top_k=params.get('top_k'),
        max_depth=params.get('max_depth'),
        n_rollouts=params.get('n_rollouts'),
        n_steps=params.get('n_steps'),
        simulations=params.get('simulations'),
        c_puct=params.get('c_puct'),
        verbose=False,
    )


def play_one_game(params, opponent, ai_is_black, max_steps,
                  mode_label='', game_idx=0, visits=0):
    '''AI vs GTP 对手一局，返回 'ai' / 'opp'。逐步打印每手。'''
    board = boards.Board(BOARD_SIZE)
    current = 'b'
    ai_color = 'b' if ai_is_black else 'w'
    opp_color = 'w' if ai_is_black else 'b'
    steps = []
    consecutive_pass = 0

    for step in range(max_steps):
        if current == ai_color:
            # AI 走棋
            move, value = ai_move(current, steps, board, params)
            if move == 'pass':
                move_str = 'pass'
            else:
                move_str = idx_to_go_str(move, have_i=False)
                try:
                    board.play(move[0], move[1], current)
                except ValueError:
                    move_str = 'pass'
            # 同步给对手
            opponent.play(current, move_str)
            steps.append([current.upper(), move_str])
            # 打印
            color_cn = '黑' if current == 'b' else '白'
            print(f'    [模式{mode_label} vs visits={visits} 局{game_idx}] '
                  f'{color_cn}方 AI 第{step+1}手 {move_str} | 胜率 {value*100:.1f}%',
                  flush=True)
        else:
            # 对手走棋
            move_str = opponent.genmove(current)
            if move_str == 'resign':
                # 对手认输 → 当前方 AI 胜
                color_cn = '黑' if current == 'b' else '白'
                print(f'    [模式{mode_label} vs visits={visits} 局{game_idx}] '
                      f'{color_cn}方对手认输')
                return 'ai' if current == ai_color else 'opp'
            if move_str == 'pass':
                pass   # 弃行
            else:
                try:
                    r, c = go_str_to_idx(move_str, have_i=False)
                    board.play(r, c, current)
                except (ValueError, KeyError):
                    move_str = 'pass'
            steps.append([current.upper(), move_str])
            # 打印
            color_cn = '黑' if current == 'b' else '白'
            print(f'    [模式{mode_label} vs visits={visits} 局{game_idx}] '
                  f'{color_cn}方对手 第{step+1}手 {move_str}',
                  flush=True)

        if move_str == 'pass':
            consecutive_pass += 1
        else:
            consecutive_pass = 0

        if consecutive_pass >= 2:
            break

        current = 'w' if current == 'b' else 'b'

    # 终局判定
    score = opponent.final_score()  # 如 'B+3.5'
    black_won = score.startswith('B')
    if ai_is_black:
        return 'ai' if black_won else 'opp'
    else:
        return 'opp' if black_won else 'ai'


# =====================================================================
def elo_delta(winrate: float) -> float:
    '''胜率 → ELO 分差。'''
    if winrate <= 0:
        return -400
    if winrate >= 1:
        return 400
    return 400 * math.log10(winrate / (1 - winrate))


def evaluate_mode(mode_key, params, visits):
    '''某模式 vs 某 visits 档，对 N 局，返回胜率。'''
    opponent = KataGoGTPClient(visits=visits)
    wins = 0
    for g in range(N_GAMES):
        opponent.clear_board()   # 每局重置棋盘，避免继承上局残局
        ai_is_black = (g % 2 == 0)
        result = play_one_game(params, opponent, ai_is_black, MAX_STEPS,
                               mode_label=mode_key, game_idx=g + 1,
                               visits=visits)
        if result == 'ai':
            wins += 1
        print(f'    局 {g+1}/{N_GAMES} | AI 执{"黑" if ai_is_black else "白"} | '
              f'胜者: {result}', flush=True)
    opponent.close()
    return wins / N_GAMES


# =====================================================================
def main():
    print('=' * 60)
    print('双模式 AI ELO 棋力评测')
    print('=' * 60)
    print(f'  每档每模式 {N_GAMES} 局 | 每局最多 {MAX_STEPS} 手')
    print()

    for mode_key, params in MODES.items():
        print(f'--- {mode_key} ({params["label"]}) ---')
        for visits in VISIT_LEVELS:
            winrate = evaluate_mode(mode_key, params, visits)
            d_elo = elo_delta(winrate)
            abs_elo = LEVEL_ELO[visits] + d_elo
            print(f'  vs visits={visits} (≈{LEVEL_ELO[visits]}): '
                  f'胜率 {winrate:.1%} | ΔELO {d_elo:+.0f} | AI≈{abs_elo:.0f}')
        print()

    print('=' * 60)
    print('评测完成')
    print('=' * 60)


if __name__ == '__main__':
    main()
