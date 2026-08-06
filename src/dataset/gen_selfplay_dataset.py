'''自对弈 + KataGo 标注数据生成流水线。

纯策略网络自对弈（无搜索），每 N 手采一个局面，
由 KataGo 标注 [黑胜率, 白胜率]，输出 .npz 数据集供价值网络训练。

用法::

    python -m dataset.gen_selfplay_dataset
    python -m dataset.gen_selfplay_dataset --games 200 --device cpu
'''

import sys
import os
import time
import argparse
import numpy as np
from sgfmill import boards

# ---- 路径 ----
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))


def parse_args():
    p = argparse.ArgumentParser(description='自对弈 + KataGo 标注数据生成')
    p.add_argument('--games', '-ng', type=int, default=None)
    p.add_argument('--interval', '-i', type=int, default=None,
                   help='采样间隔（每 N 手）')
    p.add_argument('--temperature', '-t', type=float, default=None)
    p.add_argument('--visits', '-v', type=int, default=None,
                   help='KataGo 标注 visits')
    p.add_argument('--output', '-o', type=str, default=None)
    p.add_argument('--device', '-d', choices=['cpu', 'cuda'], default=None)
    return p.parse_args()


args = parse_args()

import common
if args.device is not None:
    common.config.device = args.device
if args.games is not None:
    common.config.sp_games = args.games
if args.interval is not None:
    common.config.sp_sample_interval = args.interval
if args.temperature is not None:
    common.config.sp_temperature = args.temperature
if args.visits is not None:
    common.config.sp_katago_visits = args.visits
if args.output is not None:
    common.config.sp_output_path = args.output
common._sync_module_vars()

from common import (
    DEVICE, BOARD_SIZE, PASS_LABEL, COLOR_MAP, GAME_KOMI,
    SP_GAMES, SP_SAMPLE_INTERVAL, SP_TEMPERATURE, SP_KATAGO_VISITS,
    SP_OUTPUT_PATH, idx_to_go_str,
)
from strategy.go_strategy import GoStrategySelector
from engine.katago import get_katago_engine


# =====================================================================
def board_to_np(b: boards.Board, cur_color: str) -> np.ndarray:
    '''局面 → (3, 19, 19) numpy。

    Channel 0: 棋子（空=0, 黑=0.5, 白=1.0）
    Channel 1: 行棋方（黑=0.0, 白=1.0）
    Channel 2: 贴目
    '''
    board_ch = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            s = b.get(r, c)
            if s == 'b':
                board_ch[r, c] = 0.5
            elif s == 'w':
                board_ch[r, c] = 1.0
    color_val = 0.0 if cur_color == 'b' else 1.0
    color_ch = np.full((BOARD_SIZE, BOARD_SIZE), color_val, dtype=np.float32)
    return np.stack([board_ch, color_ch], axis=0)


def sample_move(probs: np.ndarray, tau: float) -> int:
    '''温度采样落子索引。'''
    if probs[PASS_LABEL] >= 0.999:
        return PASS_LABEL
    p = np.clip(probs, 1e-10, None)
    lp = np.log(p) / tau
    lp -= lp.max()
    ep = np.exp(lp)
    ep /= ep.sum()
    return int(np.random.choice(len(probs), p=ep))


# =====================================================================
def main():
    print('=' * 55)
    print('自对弈 + KataGo 标注数据生成')
    print('=' * 55)
    print(f'  局数: {SP_GAMES}')
    print(f'  间隔: 每 {SP_SAMPLE_INTERVAL} 手')
    print(f'  温度: {SP_TEMPERATURE}')
    print(f'  标注 visits: {SP_KATAGO_VISITS}')
    print(f'  输出: {SP_OUTPUT_PATH}')
    print()

    sg = GoStrategySelector()
    kata = get_katago_engine()

    all_inputs = []
    all_labels = []
    total_samples = 0
    total_errors = 0

    for game_i in range(SP_GAMES):
        t0 = time.time()
        board = boards.Board(BOARD_SIZE)
        current_color = 'b'
        kata_moves = []          # KataGo 格式
        sampled = []             # [(state_np, moves_snapshot), ...]
        consecutive_pass = 0

        for step in range(400):
            # ---- 采样：落子前 ----
            if step % SP_SAMPLE_INTERVAL == 0:
                state = board_to_np(board, current_color)
                sampled.append((state, list(kata_moves)))

            # ---- 策略网络预测 ----
            full_probs = sg.predict_full_probs(board, current_color)
            move_idx = sample_move(full_probs, SP_TEMPERATURE)

            # ---- 落子 ----
            if move_idx == PASS_LABEL:
                consecutive_pass += 1
                kata_moves.append([current_color.upper(), 'pass'])
            else:
                consecutive_pass = 0
                row, col = move_idx // BOARD_SIZE, move_idx % BOARD_SIZE
                try:
                    board.play(row, col, current_color)
                    kata_moves.append([
                        current_color.upper(),
                        idx_to_go_str((row, col), have_i=False),
                    ])
                except ValueError:
                    kata_moves.append([current_color.upper(), 'pass'])

            current_color = 'w' if current_color == 'b' else 'b'

            if consecutive_pass >= 2:
                break

        # 终局后也采最后一个局面
        final_state = board_to_np(board, current_color)
        sampled.append((final_state, list(kata_moves)))

        # ---- KataGo 标注 ----
        for state, moves_snap in sampled:
            if len(moves_snap) == 0:
                # 空局面 → 均等
                all_inputs.append(state)
                all_labels.append(np.array([0.5, 0.5], dtype=np.float32))
                total_samples += 1
                continue

            b_wr = kata.get_value('b', moves_snap)
            if b_wr < 0:
                total_errors += 1
                continue
            all_inputs.append(state)
            all_labels.append(np.array([b_wr, 1 - b_wr], dtype=np.float32))
            total_samples += 1

        elapsed = time.time() - t0
        pct = (game_i + 1) / SP_GAMES
        sys.stdout.write(
            f'\r  局 {game_i+1}/{SP_GAMES} ({pct:.0%}) | '
            f'累计 {total_samples} 样本 | {elapsed:.0f}s'
        )
        sys.stdout.flush()

    sys.stdout.write('\n')
    print(f'标注错误/跳过: {total_errors}')
    print(f'有效样本: {total_samples}')

    # ---- 保存 ----
    inputs_arr = np.stack(all_inputs).astype(np.float32)
    labels_arr = np.stack(all_labels).astype(np.float32)
    out_dir = os.path.dirname(SP_OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(SP_OUTPUT_PATH, inputs=inputs_arr, values=labels_arr)
    print(f'保存: {inputs_arr.shape}, {labels_arr.shape} → {SP_OUTPUT_PATH}')


if __name__ == '__main__':
    main()
