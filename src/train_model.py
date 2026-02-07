import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from models.go_cnn import GoCNN

CHUNK_DIR        = 'E:/go_dataset'
SAVE_MODEL_PATH  = 'output/go_model_step.pth'
BATCH_SIZE       = 32
EPOCHS_PER_CHUNK = 2
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 1e-5
GRADIENT_ACCUMULATION_STEPS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class GoChunkDataset(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path, mmap_mode='r')
        self.inputs = self.data['inputs']
        self.labels = self.data['labels']
        self.pass_label = 361  # 弃行

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        '''
        获取该Chunk的第index个样本
        
        :return: x, y张量元组
        '''
        x = self.inputs[index].astype(np.float32)
        x[0] = x[0] / 2.0   # 棋子数值归一化
        y = self.labels[index].astype(np.int64)
        if y == -1:
            y = self.pass_label
        return torch.tensor(x), torch.tensor(y)
    
    def close(self):
        if hasattr(self, 'data'):
            self.data.close()
            del self.data, self.inputs, self.labels
        gc.collect()

def get_sorted_chunk_files(chunk_dir):
    '''获取有序文件列表'''
    chunk_files = []
    for file in os.listdir(chunk_dir):
        if file.endswith('.npz'):
            chunk_idx = int(file.split('_')[-1].split('.')[0])
            chunk_files.append((chunk_idx, os.path.join(chunk_dir, file)))
    chunk_files.sort(key=lambda x: x[0])
    return [f[1] for f in chunk_files]  # 返回路径

def train_per_chunk():
    print(f'初始化模型 {DEVICE}')
    model = GoCNN().to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

     # 学习率调度：基于总训练步数（所有chunk累积）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)  # 每8个chunk减半
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()  # 混合精度训练

    # 获取数据集
    chunk_files = get_sorted_chunk_files(CHUNK_DIR)
    total_chunks = len(chunk_files)
    print(f'共 {total_chunks} 个拆分数据集:')
    for i, f in enumerate(chunk_files):
        print(f' [{i}] {os.path.basename(f)}')

    # 对每个chunk进行训练
    best_loss = float('inf')
    total_batches = 0

    for chunk_idx, chunk_path in enumerate(chunk_files):
        chunk_name = os.path.basename(chunk_path)
        print('='*50)
        print(f'训练第{chunk_idx+1}/{total_chunks}个数据集: {chunk_name}')

        try:
            dataset = GoChunkDataset(chunk_path)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                pin_memory=True,
                drop_last=True
            )
            print(f'加载完成  样本数: {len(dataset)}  批次数量: {len(dataloader)}')

            # 开始训练
            model.train()
            chunk_total_loss = 0.0

            for epoch in range(EPOCHS_PER_CHUNK):
                print(f'第 {chunk_idx+1} 个数据集 - Epoch {epoch+1}')
                for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                    batch_x = batch_x.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)

                    # 前向传播
                    with autocast():
                        pred = model(batch_x)
                        loss = loss_fn(pred, batch_y)
                        loss = loss / GRADIENT_ACCUMULATION_STEPS
                    
                    # 反向传播
                    scaler.scale(loss).backward()

                    # 更新参数
                    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        total_batches += 1

                    # 打印进度
                    chunk_total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                    if batch_idx % 100 == 0:
                        print(f"  数据集编号: {chunk_idx+1} | Batch {batch_idx:>4} | Loss: {chunk_total_loss:.4f}")
            
            # 当前chunk训练完成
            chunk_avg_loss = chunk_total_loss / len(dataloader)
            print(f'第 {chunk_idx+1} 个数据集训练完成   平均Loss: {chunk_avg_loss:.4f}')

            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f'当前学习率：{current_lr:.8f}')

            # 保存阶段性模型
            if chunk_avg_loss < best_loss:
                best_loss = chunk_avg_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'trained_chunks': chunk_idx + 1
                }, SAVE_MODEL_PATH) 
                print(f'阶段模型已保存')
        
        except Exception as e:
            print(f'训练出错: {e}')
            continue

        finally:  # 释放资源
            if 'dataset' in locals():
                dataset.close()
                del dataset
            if 'dataloader' in locals():
                del dataloader
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            print(f'第 {chunk_idx+1} 个数据集资源已释放')

    print(f'{total_chunks} 个数据集训练完成')

if __name__ == "__main__":
    train_per_chunk()