'''价值网络数据回炉：用 A 方自己的对局微调价值网络。

流程:
  ① A 方（自训练价值网络）实战自对弈 N 局
  ② 每局每 K 手采样一个局面
  ③ KataGo 给每个局面打胜率标签
  ④ D4 旋转对称增强
  ⑤ 在现有价值网络上微调

用法: python selftune_value_net.py
'''

import sys, os, time, argparse
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sgfmill import boards

from common import BOARD_SIZE, GAME_KOMI, idx_to_go_str, go_str_to_idx
from game.tree import BatchMinimaxMCR
from engine.katago import get_katago_engine
from models import value_net as va_net

# ===== 参数 =====
N_GAMES = 20           # 自对弈局数（100 局全跑约 10 小时，先 20 局验证）
MAX_STEPS = 150        # 每局最多手数
SAMPLE_EVERY = 5       # 每 5 手采一个局面
KOMI = 7.5
# A 方搜索参数
TOP_K = 4
DEPTH = 5
N_ROLLOUTS = 5
N_STEPS = 5
# 微调参数
LR = 1e-4              # 小学习率微调
EPOCHS = 20
BATCH_SIZE = 128
BASE_MODEL = r'output_models\go_final_val_model_1_2.pth'
SAVE_PATH = r'output_models\go_final_val_model_selftune.pth'


# =====================================================================
def board_to_np(b: boards.Board, cur_color: str) -> np.ndarray:
    '''2 通道：Channel0 棋子(空=0,黑=0.5,白=1.0)，Channel1 行棋方。'''
    board_ch = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    occupied, _ = b.list_occupied_points()
    for stone, (r, c) in occupied:
        board_ch[r, c] = 0.5 if stone == 'b' else 1.0
    color_val = 0.0 if cur_color == 'b' else 1.0
    color_ch = np.full((BOARD_SIZE, BOARD_SIZE), color_val, dtype=np.float32)
    return np.stack([board_ch, color_ch], axis=0)


def d4_augment(inputs: np.ndarray, labels: np.ndarray):
    '''D4 旋转对称增强 ×8。'''
    aug_x, aug_y = [inputs], [labels]
    for k in (1, 2, 3):
        aug_x.append(np.rot90(inputs, k=k, axes=(2, 3)))
        aug_y.append(labels)
    flipped = np.flip(inputs, axis=3)
    aug_x.append(flipped)
    aug_y.append(labels)
    for k in (1, 2, 3):
        aug_x.append(np.rot90(flipped, k=k, axes=(2, 3)))
        aug_y.append(labels)
    return np.concatenate(aug_x), np.concatenate(aug_y)


# =====================================================================
def self_play_collect(kata, n_games, max_steps, sample_every):
    '''A 方自对弈 + 采集局面。返回 (states, kata_moves_list)。'''
    print('=' * 55)
    print(f'① A 方自对弈 {n_games} 局')
    print('=' * 55)
    all_states = []
    all_kata_moves = []

    for g in range(n_games):
        board = boards.Board(BOARD_SIZE)
        current = 'b'
        steps = []           # KataGo 格式落子记录 [[color, pos], ...]
        sampled = []         # [(state, moves_snapshot), ...]

        for step in range(max_steps):
            # 落子前采样
            if step % sample_every == 0:
                sampled.append((board_to_np(board, current), list(steps)))

            # A 方走棋
            minimax = BatchMinimaxMCR(
                steps, board, current, len(steps),
                top_k=TOP_K, max_depth=DEPTH,
                use_own_value_net=True,
                n_rollouts=N_ROLLOUTS, n_steps=N_STEPS,
                verbose=False,
            )
            move, value = minimax.search()

            # 打印每一手
            move_str = idx_to_go_str(move) if move != 'pass' else 'pass'
            color_cn = '黑' if current == 'b' else '白'
            print(f'  第{g+1}局 | {color_cn}方 | 第{step+1}手 | '
                  f'{move_str} | 预估胜率 {value*100:.1f}%', flush=True)

            if move == 'pass':
                steps.append([current.upper(), 'pass'])
            else:
                try:
                    board.play(move[0], move[1], current)
                except ValueError:
                    steps.append([current.upper(), 'pass'])
                    current = 'w' if current == 'b' else 'b'
                    continue
                steps.append([current.upper(),
                              idx_to_go_str(move, have_i=False)])

            current = 'w' if current == 'b' else 'b'

        # 终局局面
        sampled.append((board_to_np(board, current), list(steps)))

        for state, moves_snap in sampled:
            all_states.append(state)
            all_kata_moves.append(moves_snap)

        pct = (g + 1) / n_games
        print(f'  局 {g+1}/{n_games} ({pct:.0%}) | 累计 {len(all_states)} 局面',
              flush=True)

    return all_states, all_kata_moves


# =====================================================================
def katago_label(kata, states, kata_moves):
    '''KataGo 打标签（静默批量）。返回 labels [黑胜率, 白胜率]。'''
    print()
    print('=' * 55)
    print(f'② KataGo 打标签 ({len(states)} 局面)')
    print('=' * 55)
    labels = np.zeros((len(states), 2), dtype=np.float32)
    labels[:, 0] = 0.5
    labels[:, 1] = 0.5   # 默认均势
    errors = 0
    PIPE = 50   # 流水线批量大小

    # 分批流水线查询
    for start in range(0, len(states), PIPE):
        end = min(start + PIPE, len(states))
        queries = []
        valid_idx = []
        for i in range(start, end):
            if kata_moves[i]:
                queries.append(('b', kata_moves[i]))
                valid_idx.append(i)

        if queries:
            values = kata.get_value_batch(queries)
            for i, v in zip(valid_idx, values):
                if v < 0:
                    errors += 1
                else:
                    labels[i] = [v, 1 - v]

        pct = end / len(states)
        print(f'  标注 {end}/{len(states)} ({pct:.0%}) | 错误 {errors}',
              flush=True)

    print(f'  完成: {len(labels)} 标签 | 错误 {errors}')
    return labels


# =====================================================================
def finetune(inputs, labels):
    '''微调价值网络。'''
    print()
    print('=' * 55)
    print(f'⑤ 微调价值网络 ({EPOCHS} epochs)')
    print('=' * 55)

    # D4 增强
    inputs, labels = d4_augment(inputs, labels)
    print(f'  D4 增强后: {len(inputs):,} 样本')

    # 切分
    n_val = max(2000, len(inputs) // 10)
    indices = np.random.RandomState(42).permutation(len(inputs))
    inputs, labels = inputs[indices], labels[indices]
    train_x, val_x = inputs[:-n_val], inputs[-n_val:]
    train_y, val_y = labels[:-n_val], labels[-n_val:]
    print(f'  训练: {len(train_x):,} | 验证: {len(val_x):,}')

    class DS(Dataset):
        def __init__(self, x, y): self.x, self.y = x, y
        def __len__(self): return len(self.x)
        def __getitem__(self, i):
            return torch.tensor(self.x[i]), torch.tensor(self.y[i])

    train_loader = DataLoader(DS(train_x, train_y), BATCH_SIZE,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(DS(val_x, val_y), BATCH_SIZE, shuffle=False,
                            pin_memory=True)

    # 加载现有价值网络
    model = va_net.GoValueNet(dropout_rate=0.5)
    state_dict = torch.load(BASE_MODEL, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'  加载基座模型: {BASE_MODEL}')

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )

    best_val = float('inf')
    patience = 0
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = val_mae = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += loss_fn(pred, by).item() * bx.size(0)
                val_mae += (pred - by).abs().mean(1).sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        scheduler.step()
        status = ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            status = ' ✓'
            patience = 0
        else:
            patience += 1

        print(f'Epoch {epoch:2d}/{EPOCHS} | train={train_loss:.4f} | '
              f'val={val_loss:.4f} | mae={val_mae:.3f}{status}')

        if patience >= 15:
            print('  早停')
            break

    print(f'  完成 | 最佳 val_loss={best_val:.4f} | 模型: {SAVE_PATH}')


# =====================================================================
def main():
    kata = get_katago_engine()

    # ① 自对弈
    states, kata_moves = self_play_collect(kata, N_GAMES, MAX_STEPS, SAMPLE_EVERY)

    # ② KataGo 打标签
    labels = katago_label(kata, states, kata_moves)

    # ③+④ 保存中间数据（增强在训练时做）
    os.makedirs('data', exist_ok=True)
    npz_path = 'data/selftune_dataset.npz'
    np.savez_compressed(npz_path,
                        inputs=np.stack(states), values=labels)
    print(f'\n  数据已保存: {npz_path} ({len(states):,} 样本)')

    # ⑤ 微调
    finetune(np.stack(states), labels)

    kata.close()


if __name__ == '__main__':
    main()
