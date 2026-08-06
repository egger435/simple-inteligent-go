'''KataGo 知识蒸馏 —— 用 KataGo 的策略分布训练策略网络。

从 SGF 提取局面 → KataGo 查询 362 维策略概率 → KL 散度训练。

用法: python distill_strategy.py
'''

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sgfmill import sgf, boards
from common import BOARD_SIZE, DEVICE, STRATEGY_MODEL_PATH, COLOR_MAP

# ===== 配置 =====
SGF_DIR = 'data/varied_selfplay_commentary_sgfs'
MAX_SAMPLES = 50000      # 蒸馏样本数（Katago 查询耗时，不宜太多）
SAMPLE_INTERVAL = 5       # 每 5 手采一个
LR = 1e-4
EPOCHS = 40
BATCH_SIZE = 64

# =====================================================================
def board_to_np(b, cur_color):
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
print('KataGo 知识蒸馏')
print('=' * 55)
print(f'基座模型: {STRATEGY_MODEL_PATH}')
print(f'蒸馏样本: {MAX_SAMPLES:,}')
print()

# ---- Step 1: 从 SGF 提取局面 ----
print('Step 1: 提取局面...')
positions = []  # [(board_np, cur_color, kata_moves), ...]
processed = 0
errors = 0
_first = True

for root, dirs, files in os.walk(SGF_DIR):
    for f in files:
        if not f.endswith('.sgf'):
            continue
        if _first:
            _first = False
            print('  开始扫描...', flush=True)
        fp = os.path.join(root, f)
        processed += 1
        try:
            with open(fp, 'rb') as fh:
                game = sgf.Sgf_game.from_bytes(fh.read())
        except Exception:
            errors += 1
            continue

        board = boards.Board(19)
        kata_moves = []
        move_seq = 0
        for node in game.get_main_sequence():
            move = node.get_move()
            if move is None or move[1] is None:
                continue
            color, (row, col) = move
            if board.get(row, col) is not None:
                continue

            if move_seq % SAMPLE_INTERVAL == 0:
                from common import idx_to_go_str
                state = board_to_np(board, color)
                positions.append((state, color, list(kata_moves)))

            # 落子
            board.play(row, col, color)
            kata_moves.append([color.upper(), idx_to_go_str((row, col), have_i=False)])
            move_seq += 1

            if len(positions) >= MAX_SAMPLES:
                break

        if processed % 200 == 0 or len(positions) >= MAX_SAMPLES:
            pct = len(positions) / MAX_SAMPLES
            print(f'  文件 {processed:,} | {len(positions):,}/{MAX_SAMPLES:,} 局面 ({pct:.0%}) | 错误 {errors}', flush=True)

        if len(positions) >= MAX_SAMPLES:
            break
    if len(positions) >= MAX_SAMPLES:
        break

print(f'  提取 {len(positions):,} 个局面')
print()

# ---- Step 2: Katago 查询策略分布（流水线批量） ----
print('Step 2: Katago 查询策略分布...')
from engine.katago import get_katago_engine
kata = get_katago_engine()

PIPE_SIZE = 100   # 每次流水线发送的请求数
all_states = []
all_policies = np.zeros((MAX_SAMPLES, 362), dtype=np.float32)
valid_count = 0
t_start = time.time()

# 按批次流水线处理：连续发 PIPE_SIZE 个请求 → 连续读 PIPE_SIZE 个响应
pos_idx = 0
while pos_idx < len(positions):
    batch_end = min(pos_idx + PIPE_SIZE, len(positions))
    batch_states = []
    req_ids = set()

    # ---- 发送整个批次 ----
    for i in range(pos_idx, batch_end):
        state, color, moves = positions[i]
        kata.request_id += 1
        rid = str(kata.request_id)
        req_ids.add(rid)
        request = {
            'id': rid,
            'boardXSize': 19, 'boardYSize': 19,
            'initialStones': [], 'moves': moves,
            'rules': 'chinese', 'komi': 7.5,
            'visits': 1,           # 策略分布 1 visit 即可
            'includePolicy': True,
            'includeOwnership': False,
            'includeMovesOwnership': False,
        }
        kata.process.stdin.write(json.dumps(request) + '\n')
        batch_states.append(state)
    kata.process.stdin.flush()

    # ---- 读取整个批次（按请求数收响应） ----
    received = 0
    t0 = time.time()
    while received < len(batch_states) and time.time() - t0 < 30:
        line = kata._read_line(0.5)
        if not line:
            continue
        try:
            r = json.loads(line)
            if 'error' in r:
                continue
            if 'policy' in r and valid_count < MAX_SAMPLES:
                all_policies[valid_count] = np.array(r['policy'], dtype=np.float32)
                all_states.append(batch_states[received])
                valid_count += 1
                received += 1
        except json.JSONDecodeError:
            continue

    pos_idx = batch_end

    elapsed = time.time() - t_start
    eta = elapsed / valid_count * (MAX_SAMPLES - valid_count) if valid_count > 0 else 0
    pct = valid_count / MAX_SAMPLES
    print(f'  {valid_count:,}/{MAX_SAMPLES:,} ({pct:.0%}) | {elapsed:.0f}s | 预计剩余 {eta:.0f}s', flush=True)

print(f'  完成: {valid_count:,} 有效样本')
print()

# ---- Step 3: 蒸馏训练 ----
print('Step 3: 蒸馏训练...')
all_states = np.stack(all_states[:valid_count])
all_policies = all_policies[:valid_count]

n_val = min(2000, valid_count // 10)
indices = np.random.RandomState(42).permutation(valid_count)
all_states, all_policies = all_states[indices], all_policies[indices]
train_x, val_x = all_states[:-n_val], all_states[-n_val:]
train_y, val_y = all_policies[:-n_val], all_policies[-n_val:]
print(f'  训练: {len(train_x):,} | 验证: {len(val_x):,}')
print()

class DS(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])

train_loader = DataLoader(DS(train_x, train_y), BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
val_loader = DataLoader(DS(val_x, val_y), BATCH_SIZE, shuffle=False, pin_memory=True)

from config import config
model = config.create_strategy_model().to(DEVICE)
model.train()
print(f'模型参数: {sum(p.numel() for p in model.parameters()):,}')
print()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# KL 散度损失
def distillation_loss(student_logits, teacher_logits):
    student_log_prob = F.log_softmax(student_logits, dim=1)
    teacher_prob = F.softmax(teacher_logits, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')

save_path = STRATEGY_MODEL_PATH.replace('.pth', '_distill.pth')
best_val = float('inf')

print('=' * 55)
print(f'开始蒸馏 ({EPOCHS} epochs)')
print('=' * 55)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        loss = distillation_loss(model(bx), by)
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
            val_loss += distillation_loss(pred, by).item() * bx.size(0)
            val_acc += (pred.argmax(1) == by.argmax(1)).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    scheduler.step()
    status = ''
    if val_loss < best_val:
        best_val = val_loss
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        status = ' ✓'

    print(f'Epoch {epoch:2d}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f} | acc={val_acc:.1%}{status}')

print(f'\n完成 | 模型: {save_path}')
