import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import models.strategy_net as sg_net
from common import *

SG_TRAIN_MODEL = sg_net.GoCNN_p()  # 训练策略网络使用的模型

class SgChunkDataset(Dataset):
    '''策略网络分片数据集'''
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

class SgModelTrain():
    '''策略网络训练类'''
    def __init__(self, model=SG_TRAIN_MODEL):
        self.model = model.to(DEVICE)
        self.chunk_files = get_sorted_chunk_files(SG_CHUNK_DIR)
        total_chunks = len(self.chunk_files)
        print(f'共 {total_chunks} 个拆分数据集:')
        for i, f in enumerate(self.chunk_files):
            print(f' [{i}] {os.path.basename(f)}')
        self.best_loss = float('inf')
        self.total_batches = 0
    
    def validate_model(self):
        '''在验证集上评估模型, 返回预测准确率'''
        self.model.eval()
        valid_dataset = SgChunkDataset(SG_VALID_CHUNK_DIR)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=32,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

        total_correct = total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                pred = self.model(batch_x)

                pred_idx = torch.argmax(pred, dim=1)
                total_correct += (pred_idx == batch_y).sum().item()
                total_samples += batch_x.size(0)
        
        valid_acc = total_correct / total_samples

        valid_dataset.close()
        del valid_dataset, valid_loader
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        self.model.train()  # 重新进入train模式进行下一chunk训练
        return valid_acc
    
    

def validate_model(model, valid_path=SG_VALID_CHUNK_DIR, batch_size=32):
    '''在验证集上评估模型, 返回预测准确率'''
    model.eval()
    valid_dataset = SgChunkDataset(valid_path)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    total_correct = total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            pred = model(batch_x)

            pred_idx = torch.argmax(pred, dim=1)
            total_correct += (pred_idx == batch_y).sum().item()
            total_samples += batch_x.size(0)
    
    valid_acc = total_correct / total_samples

    valid_dataset.close()
    del valid_dataset, valid_loader
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    model.train()
    return valid_acc

def train_sg_model_per_chunk():
    print(f'初始化模型 {DEVICE}')
    model = SG_TRAIN_MODEL.to(DEVICE)
    # # 随机梯度下降
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=LEARNING_RATE,
    #     momentum=0.9,
    #     weight_decay=WEIGHT_DECAY
    # )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=SG_LEARNING_RATE,
        weight_decay=SG_WEIGHT_DECAY
    )

     # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)  # 每8个chunk减半
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()  # 混合精度训练

    # 获取数据集
    chunk_files = get_sorted_chunk_files(SG_CHUNK_DIR)
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
            dataset = SgChunkDataset(chunk_path)
            dataloader = DataLoader(
                dataset,
                batch_size=SG_BATCH_SIZE,
                shuffle=True,
                pin_memory=True,
                drop_last=True
            )
            print(f'加载完成  样本数: {len(dataset)}  批次数量: {len(dataloader)}')

            # 开始训练
            model.train()
            chunk_total_loss = 0.0

            for epoch in range(SG_EPOCHS_PER_CHUNK):
                epoch_loss = 0.0
                print(f'第 {chunk_idx+1} 个数据集 - Epoch {epoch+1}')
                for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                    batch_x = batch_x.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)

                    # 前向传播
                    with autocast():
                        pred = model(batch_x)
                        loss = loss_fn(pred, batch_y)
                        loss = loss / SG_GRADIENT_ACCUMULATION_STEPS
                    
                    # 反向传播
                    scaler.scale(loss).backward()

                    # 更新参数
                    if (batch_idx + 1) % SG_GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        total_batches += 1

                    epoch_loss += loss.item() * SG_GRADIENT_ACCUMULATION_STEPS
                    chunk_total_loss += epoch_loss

                    # 打印进度
                    if batch_idx % 100 == 0:
                        avg_batch_loss = epoch_loss / (batch_idx + 1)
                        print(f"  数据集编号: {chunk_idx+1} | batch: {batch_idx:4} | Epoch: {epoch+1} | Epoch Loss: {epoch_loss:.4f} | Avg Loss: {avg_batch_loss:.4f}")
            
            # 当前chunk训练完成
            chunk_avg_loss = chunk_total_loss / (len(dataloader) * SG_EPOCHS_PER_CHUNK)
            print(f'第 {chunk_idx+1} 个数据集训练完成   平均Loss: {chunk_avg_loss:.6f}')

            # 验证集评估
            print('\n进行验证集评估...')
            valid_acc = validate_model(model)
            print(f'验证集准确率: {100*valid_acc:.2f}%')

            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f'当前学习率：{current_lr:.8f}')

            log_info = (
                f"===== Chunk {chunk_idx+1}/{total_chunks} =====\n"
                f"训练集：{chunk_name}\n"
                f"训练平均Loss: {chunk_avg_loss:.6f}\n"
                f"验证准确率: {100*valid_acc:.2f}%\n"
                f"当前学习率: {current_lr:.8f}\n"
            )

            with open(LOG_PATH, 'a') as f:
                f.write(log_info)

            # 保存阶段性模型
            if chunk_avg_loss < best_loss:
                best_loss = chunk_avg_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'trained_chunks': chunk_idx + 1
                }, SG_SAVE_MODEL_PATH) 
                print(f'阶段模型已保存')
        
        except Exception as e:
            print(f'训练出错: {e}')
            with open(LOG_PATH, 'a') as f:
                print(f'训练出错: {e}', file=f)
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

    print(f'\n {total_chunks} 个数据集训练完成 | 最优Loss: {best_loss:.6f}')
    with open(LOG_PATH, 'a') as f:
        f.write(f'训练完成 | 总Chunks: {total_chunks} | 最优Loss: {best_loss:.6f}\n')

if __name__ == "__main__":
    train_sg_model_per_chunk()