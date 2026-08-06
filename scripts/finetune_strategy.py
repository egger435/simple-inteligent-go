'''策略网络微调 —— 在现有模型上用强 AI SGF 继续训练。

用法: python finetune_strategy.py
'''

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import re, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sgfmill import sgf, boards
from common import BOARD_SIZE, DEVICE, STRATEGY_MODEL_PATH, COLOR_MAP

# ===== 配置 =====
SGF_DIR = 'data/varied_selfplay_commentary_sgfs'
MAX_SAMPLES = 300000
SAMPLE_INTERVAL = 2       # 每 2 手取一个样本
LR = 1e-4                  # 微调用较小学习率
EPOCHS = 30
BATCH_SIZE = 128

# =====================================================================
def board_to_np(b, cur_color):
    '''局面 → (2, 19, 19)。'''
    board_ch = np.zeros((19, 19), dtype=np.float32)
    for r in range(19):
        for c in range(19):
            s = b.get(r, c)
            if s == 'b':    board_ch[r, c] = 1.0
            elif s == 'w':  board_ch[r, c] = 2.0
    board_ch /= 2.0
    color_val = 0.0 if cur_color == 'b' else 1.0
    color_ch = np.full((19, 19), color_val, dtype=np.float32)
    return np.stack([board_ch, color_ch], axis=0)


# =====================================================================
print('=' * 55)
print('策略网络微调 (Fine-tuning)')
print('=' * 55)
print(f'基座模型: {STRATEGY_MODEL_PATH}')
print(f'数据源: {SGF_DIR}')
print(f'学习率: {LR}')
print()

# ---- Step 1: 提取训练数据 ----
print('提取训练数据...', flush=True)
print('  正在扫描目录...', end=' ', flush=True)
_first_file = False
t_start = time.time()
inputs_arr = np.empty((MAX_SAMPLES, 2, 19, 19), dtype=np.float32)
labels_arr = np.empty(MAX_SAMPLES, dtype=np.int64)
count = 0
processed = 0
errors = 0

for root, dirs, files in os.walk(SGF_DIR):
    for f in files:
        if not f.endswith('.sgf'):
            continue
        if not _first_file:
            _first_file = True
            print('  开始处理...', flush=True)
        fp = os.path.join(root, f)
        processed += 1
        try:
            with open(fp, 'rb') as fh:
                raw = fh.read()
            game = sgf.Sgf_game.from_bytes(raw)
            board = boards.Board(19)
            move_seq = 0   # 本局落子序号（独立于全局采样计数）
            for node in game.get_main_sequence():
                move = node.get_move()
                if move is None or move[1] is None:
                    continue
                color, (row, col) = move
                if board.get(row, col) is not None:
                    continue
                if move_seq % SAMPLE_INTERVAL == 0 and count < MAX_SAMPLES:
                    state = board_to_np(board, color)
                    move_idx = row * 19 + col
                    inputs_arr[count] = state
                    labels_arr[count] = move_idx
                    count += 1
                board.play(row, col, color)
                move_seq += 1
                if count >= MAX_SAMPLES:
                    break
        except Exception:
            errors += 1

        if processed % 100 == 0 or count >= MAX_SAMPLES:
            pct = count / MAX_SAMPLES
            elapsed = time.time() - t_start
            print(f'  文件 {processed:,} | {count:,}/{MAX_SAMPLES:,} ({pct:.0%}) | 错误 {errors} | {elapsed:.0f}s', flush=True)

        if count >= MAX_SAMPLES:
            break
    if count >= MAX_SAMPLES:
        break

# 截断
inputs_arr = inputs_arr[:count]
labels_arr = labels_arr[:count]
print(f'提取完成: {count:,} 样本 | 错误 {errors}')
print()

# ---- Step 2: 加载现有模型 ----
from config import config
model = config.create_strategy_model()
model.to(DEVICE)
model.train()
print(f'模型已加载: {STRATEGY_MODEL_PATH}')
print(f'参数: {sum(p.numel() for p in model.parameters()):,}')
print()

# ---- Step 3: 微调 ----
n_val = min(5000, count // 10)
indices = np.random.RandomState(42).permutation(count)
inputs_arr, labels_arr = inputs_arr[indices], labels_arr[indices]
train_x, val_x = inputs_arr[:-n_val], inputs_arr[-n_val:]
train_y, val_y = labels_arr[:-n_val], labels_arr[-n_val:]
print(f'训练: {len(train_x):,} | 验证: {len(val_x):,}')
print()

class DS(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])

train_loader = DataLoader(DS(train_x, train_y), BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
val_loader = DataLoader(DS(val_x, val_y), BATCH_SIZE, shuffle=False, pin_memory=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

save_path = STRATEGY_MODEL_PATH.replace('.pth', '_ft.pth')
best_val = float('inf')

print('=' * 55)
print(f'开始微调 ({EPOCHS} epochs)')
print('=' * 55)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(bx), by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * bx.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = val_acc = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            pred = model(bx)
            val_loss += loss_fn(pred, by).item() * bx.size(0)
            val_acc += (pred.argmax(1) == by).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    scheduler.step()
    status = ''
    if val_loss < best_val:
        best_val = val_loss
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        status = ' ✓ saved'

    print(f'Epoch {epoch:2d}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f} | acc={val_acc:.1%}{status}')

print(f'\n完成 | 模型: {save_path}')
print(f'使用方式: 将 config.json 中 strategy_model_path 改为 "{save_path}"')
