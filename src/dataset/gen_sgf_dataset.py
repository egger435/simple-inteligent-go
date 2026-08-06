'''从带胜率标注的 SGF 中提取价值网络训练数据。

用法::

    python dataset/gen_sgf_dataset.py --dir data/varied_selfplay_commentary_sgfs -o data/sgf_val.npz
    python dataset/gen_sgf_dataset.py --dir data/gogod_commentary_sgfs -o data/gogod_val.npz
'''

import sys, os, re, argparse, time
import numpy as np
from sgfmill import sgf, boards

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from common import BOARD_SIZE, GAME_KOMI, COLOR_MAP


def parse_args():
    p = argparse.ArgumentParser(description='从 SGF 提取价值网络训练数据')
    p.add_argument('--dir', required=True, help='SGF 目录')
    p.add_argument('-o', '--output', default='data/sgf_val.npz')
    p.add_argument('--max', type=int, default=0, help='最多处理文件数 (0=全部)')
    p.add_argument('--sample', type=int, default=3, help='每 N 手采一个样本')
    return p.parse_args()


def board_to_np(b: boards.Board, cur_color: str) -> np.ndarray:
    board_ch = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            s = b.get(r, c)
            if s == 'b':    board_ch[r, c] = 0.5
            elif s == 'w':  board_ch[r, c] = 1.0
    color_val = 0.0 if cur_color == 'b' else 1.0
    color_ch = np.full((BOARD_SIZE, BOARD_SIZE), color_val, dtype=np.float32)
    return np.stack([board_ch, color_ch], axis=0)


def extract_one_sgf(filepath: str, sample_interval: int):
    '''提取一个 SGF 中的所有 (局面, 胜率) 对。'''
    with open(filepath, 'rb') as f:
        content = f.read()
    game = sgf.Sgf_game.from_bytes(content)
    board = boards.Board(BOARD_SIZE)
    data = []

    for node in game.get_main_sequence():
        move = node.get_move()
        comment = node.get('C') or ''

        # 解析 C[0.493\n8042] → black_winrate
        wr = None
        if comment:
            m = re.match(r'([\d.]+)', comment.strip())
            if m:
                wr = float(m.group(1))

        # 采样（落子前局面 + 胜率）
        if wr is not None and move is not None:
            color, (row, col) = move
            # 局面 = 落子前的棋盘 + 此时的胜率
            # sgfmill 的 get_move() 返回 (color, pos)，落子已执行
            # 注释中的胜率通常对应落子后的分析
            # 我们用落子前的盘面 + 当前行棋方
            if board.get(row, col) is not None:
                # 该位置已有子（非法 SGF）→ 跳过
                continue

            # 构建输入: 落子前的棋盘 + 轮到谁走
            state = board_to_np(board, color)
            # 标签: wr 是黑方胜率
            data.append((state, wr))

            # 落子
            board.play(row, col, color)

        elif move is None and wr is not None:
            # pass 或终局注释
            pass

    return data


def main():
    args = parse_args()
    print(f'SGF 目录: {args.dir}')
    print(f'采样间隔: 每 {args.sample} 手')
    print()

    print('  边扫描边提取...', flush=True)

    all_inputs, all_labels = [], []
    errors = 0
    processed = 0
    max_files = args.max if args.max > 0 else None
    t_start = time.time()

    for root, dirs, files in os.walk(args.dir):
        for f in files:
            if not f.endswith('.sgf'):
                continue
            fp = os.path.join(root, f)

            # 每个文件处理前先打印，方便定位卡住的文件
            if processed == 0:
                sys.stdout.write(f'\r  处理第 1 个: {os.path.basename(fp)}...')
                sys.stdout.flush()

            try:
                recs = extract_one_sgf(fp, args.sample)
                for state, wr in recs:
                    all_inputs.append(state)
                    all_labels.append(np.array([wr, 1 - wr], dtype=np.float32))
            except Exception:
                errors += 1
                continue

            processed += 1

            if processed % 10 == 0 or processed == 1:
                elapsed = time.time() - t_start
                sys.stdout.write(
                    f'\r  已处理 {processed} 个文件 | '
                    f'{len(all_inputs):,} 样本 | '
                    f'{elapsed:.0f}s'
                )
                sys.stdout.flush()

            if max_files and processed >= max_files:
                break

        if max_files and processed >= max_files:
            break

    sys.stdout.write(f'\r  完成 {processed} 文件 | {len(all_inputs):,} 样本 | 错误 {errors}    \n')

    inputs_arr = np.stack(all_inputs).astype(np.float32)
    labels_arr = np.stack(all_labels).astype(np.float32)

    b_wr = labels_arr[:, 0]
    print(f'黑方平均胜率: {b_wr.mean():.3f}  std: {b_wr.std():.3f}')
    print(f'黑胜(>0.5) 占比: {(b_wr > 0.5).mean():.1%}')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(args.output, inputs=inputs_arr, values=labels_arr)
    print(f'保存: {args.output} ({inputs_arr.shape[0]:,} 样本)')


if __name__ == '__main__':
    main()
