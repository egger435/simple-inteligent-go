'''对局循环逻辑。'''

from common import HAVE_I, go_str_to_idx, idx_to_go_str
from game.tree import search_move


def ai_play(steps_list: list, board, color: str, curstep: int,
            ko_pos=None):
    '''AI 行棋：按配置选择搜索算法（minimax / mcts），返回最优落子。'''
    best_move, best_value = search_move(
        steps_list, board, color, curstep, ko_pos=ko_pos,
    )
    print(f'AI 选择落子: {idx_to_go_str(best_move)} | value: {best_value:.4f}')
    return best_move


# =====================================================================
# 手动对局
# =====================================================================

def vs_ai(go_player) -> None:
    '''命令行手动对局循环。'''
    print('* 落子输入格式：列字符+行坐标  J8')

    if go_player.ai_player == 'b':
        move = ai_play(
            go_player.steps, go_player.board, 'b',
            go_player.curstep, ko_pos=None,
        )
        go_player.play_move(move)

    while True:
        go_player.print_board()
        human_input = input('落子位置: ')
        if human_input == 'pass':
            go_player.play_move_str('pass')
        elif human_input == 'back':
            go_player.back_state()
            continue
        else:
            try:
                go_player.play_move_str(human_input)
            except ValueError as e:
                # 劫争禁着点 / 已有棋子等非法落子，重新输入
                print(f'✗ {e}')
                continue
            # 以用户坐标系统展示落子序列
            display_steps = []
            for color, pos_str in go_player.steps:
                if pos_str == 'pass':
                    display_steps.append([color, pos_str])
                else:
                    idx = go_str_to_idx(pos_str, have_i=False)
                    display_steps.append([color, idx_to_go_str(idx, HAVE_I)])
            print(f'当前落子序列: {display_steps}')

        ai_move = ai_play(
            go_player.steps, go_player.board,
            go_player.current_color, go_player.curstep,
            go_player.ko_pos,
        )
        if ai_move == 'pass':
            print('AI 选择弃行')
            go_player.play_move_str('pass')
        else:
            go_player.play_move(ai_move)
