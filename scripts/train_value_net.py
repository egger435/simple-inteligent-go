'''价值网络训练入口 — 使用自对弈 + KataGo 标注数据。

用法::

    python train_value_net.py
    python train_value_net.py --data data/val.npz --epochs 50 --device cuda
'''

import sys
import os
import argparse

# ---- 路径 ----
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- CLI ----
parser = argparse.ArgumentParser(description='价值网络训练')
parser.add_argument('--data', type=str, default=None,
                    help='训练数据 .npz 路径')
parser.add_argument('--epochs', type=int, default=None,
                    help='训练轮数（默认: 50）')
parser.add_argument('--batch-size', type=int, default=None,
                    help='Batch size（默认: 128）')
parser.add_argument('--lr', type=float, default=None,
                    help='学习率（默认: 1e-3）')
parser.add_argument('--save', type=str, default=None,
                    help='模型保存路径')
parser.add_argument('--device', '-d', choices=['cpu', 'cuda'], default=None)
args = parser.parse_args()

# ---- 导入 common 并覆写参数 ----
import common
if args.device is not None:
    common.config.device = args.device
if args.data is not None:
    common.config.va_dataset_path = args.data
if args.save is not None:
    common.config.va_save_model_path = args.save
common._sync_module_vars()

from common import DEVICE, BOARD_SIZE, GAME_KOMI
from models import value_net as va_net


# =====================================================================
class VaDataset(Dataset):
    '''价值网络数据集（兼容软标签）。'''

    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        self.inputs = inputs.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])


# =====================================================================
def _d4_augment(inputs: np.ndarray, labels: np.ndarray):
    '''D4 对称群增强：4 个旋转 × 2 种翻转 = 8 倍样本。

    Args:
        inputs: (N, 2, 19, 19) 局面张量
        labels: (N, 2) 胜率标签

    Returns:
        augmented inputs, labels (8N samples)
    '''
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

    return (np.concatenate(aug_x, axis=0).astype(np.float32),
            np.concatenate(aug_y, axis=0).astype(np.float32))


# =====================================================================
def main():
    # 从 common 取最终值（可能已被 CLI 覆盖）
    data_path = common.config.va_dataset_path
    save_path = common.config.va_save_model_path
    batch_size = common.config.va_batch_size if args.batch_size is None else args.batch_size
    lr = common.config.va_learning_rate if args.lr is None else args.lr
    epochs = 50 if args.epochs is None else args.epochs

    print('=' * 55)
    print('价值网络训练')
    print('=' * 55)
    print(f'  数据: {data_path}')
    print(f'  设备: {DEVICE}')
    print(f'  Batch: {batch_size}')
    print(f'  学习率: {lr}')
    print(f'  目标轮数: {epochs}')
    print(f'  保存: {save_path}')
    print()

    # ---- 加载数据 ----
    data = np.load(data_path)
    all_x = data['inputs']
    all_y = data['values']

    print(f'  原始样本: {len(all_x):,}')

    # 标签分布（增强前）
    b_winrates = all_y[:, 0]
    print(f'  黑方平均胜率: {b_winrates.mean():.3f}')
    print(f'  黑方胜率 std: {b_winrates.std():.3f}')
    print(f'  黑胜(>0.5) 占比: {(b_winrates > 0.5).mean():.1%}')
    print()

    # ---- 先切分再增强（避免同一局面不同朝向泄漏） ----
    n_val = max(5000, len(all_x) // 10)    # 至少 5000，或 10%
    indices = np.random.RandomState(42).permutation(len(all_x))
    all_x, all_y = all_x[indices], all_y[indices]

    raw_val_x, raw_val_y = all_x[-n_val:], all_y[-n_val:]
    raw_train_x, raw_train_y = all_x[:-n_val], all_y[:-n_val]

    train_x, train_y = _d4_augment(raw_train_x, raw_train_y)
    val_x, val_y = _d4_augment(raw_val_x, raw_val_y)
    print(f'  训练集: {len(train_x):,} (增强后)')
    print(f'  验证集: {len(val_x):,} (增强后)')
    print()

    train_ds = VaDataset(train_x, train_y)
    val_ds = VaDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, pin_memory=True)

    # ---- 模型 ----
    model = va_net.GoValueNet(dropout_rate=0.5).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  模型: {n_params:,} 参数')
    print()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 20  # 放宽早停

    # ---- 训练循环 ----
    for epoch in range(1, epochs + 1):
        # == 训练 ==
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_ds)

        # == 验证 ==
        model.eval()
        val_loss = 0.0
        val_mae = 0.0       # 胜率平均绝对误差（比 acc 更有意义）
        val_acc = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                pred = model(batch_x)
                val_loss += loss_fn(pred, batch_y).item() * batch_x.size(0)
                val_mae += (pred - batch_y).abs().mean(1).sum().item()
                val_acc += (pred.argmax(1) == batch_y.argmax(1)).sum().item()
        val_loss /= len(val_ds)
        val_mae /= len(val_ds)
        val_acc /= len(val_ds)

        scheduler.step()

        # 打印
        lr_now = optimizer.param_groups[0]['lr']
        status = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            status = ' ✓ saved'
            patience_counter = 0
        else:
            patience_counter += 1

        print(f'Epoch {epoch:3d}/{epochs} | '
              f'train={train_loss:.4f} | val_loss={val_loss:.4f} | '
              f'mae={val_mae:.3f} | acc={val_acc:.1%} | '
              f'lr={lr_now:.2e}{status}')

        if patience_counter >= patience_limit:
            print(f'\n早停：{patience_limit} 轮未改善')
            break

    print(f'\n训练完成 | 最佳 val_loss: {best_val_loss:.4f} | 模型: {save_path}')


if __name__ == '__main__':
    main()
