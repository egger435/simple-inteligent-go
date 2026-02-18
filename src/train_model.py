import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import models.value_net as va_net
from common import *

SG_TRAIN_MODEL = sg.GoCNN_t()     # 训练策略网络使用的模型
VA_TRAIN_MODEL = va_net.GoValueNet()  # 训练价值网络使用的模型

class SgChunkDataset(Dataset):
    '''策略网络分片数据集'''
    def __init__(self, npz_path, is_train):
        self.data = np.load(npz_path, mmap_mode='r')
        self.inputs = self.data['inputs']
        self.labels = self.data['labels']
        self.pass_label = 361  # 弃行
        self.board_size = 19
        self.is_train = is_train

    def __len__(self):
        return len(self.inputs)
    
    def _augment(self, x, y):
        '''数据增强'''
        if y == self.pass_label:
            return x, y
        
        board = x[0].copy()
        i = y // BOARD_SIZE
        j = y % BOARD_SIZE

        trans_type = np.random.randint(0, 8)

        if trans_type == 1:
            board = np.rot90(board, k=1)
            i, j = j, self.board_size - 1 - i
        elif trans_type == 2:
            board = np.rot90(board, k=2)
            i, j = self.board_size - 1 - i, self.board_size - 1 - j
        elif trans_type == 3:
            board = np.rot90(board, k=3)
            i, j = self.board_size - 1 - j, i
        elif trans_type == 4:
            board = np.fliplr(board)
            j = self.board_size - 1 - j
        elif trans_type == 5:
            board = np.fliplr(board)
            board = np.rot90(board, k=1)
            i, j = self.board_size - 1 - j, self.board_size - 1 - i
        elif trans_type == 6:
            board = np.fliplr(board)
            board = np.rot90(board, k=2)
            i, j = self.board_size - 1 - i, j
        elif trans_type == 7:
            board = np.fliplr(board)
            board = np.rot90(board, k=3)
            i, j = j, i
        
        x_aug = x.copy()
        x_aug[0] = board
        y_aug = i * BOARD_SIZE + j

        return x_aug, y_aug

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
        
        if self.is_train:
            x, y = self._augment(x, y)
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
        self.total_chunks = len(self.chunk_files)
        print(f'共 {self.total_chunks} 个拆分数据集:')
        for i, f in enumerate(self.chunk_files):
            print(f' [{i}] {os.path.basename(f)}')

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=SG_LEARNING_RATE,
            weight_decay=SG_WEIGHT_DECAY
        )
        self.scaler = GradScaler()  # 混合精度训练
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(self.chunk_files),  # 每个全局Epoch（遍历所有Chunk）为一个周期
            T_mult=1,                       # 周期长度不变
            eta_min=2e-6,                   # 最小学习率
            verbose=False
        )

        self.best_loss = float('inf')
        self.total_batches = 0
        self.best_vali_acc = 0.0
        
    def validate_model(self):
        '''在验证集上评估模型, 返回预测准确率'''
        self.model.eval()
        valid_dataset = SgChunkDataset(SG_VALID_CHUNK_DIR, False)
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
    
    def train_per_chunk(self):
        '''每个chunk训练一次并评估模型'''
        # 学习率调度
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=16, gamma=0.85) 
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-6)
        
        scaler = GradScaler()  # 混合精度训练

        for chunk_idx, chunk_path in enumerate(self.chunk_files):
            chunk_name = os.path.basename(chunk_path)
            print('='*50)
            print(f'训练第{chunk_idx+1}/{self.total_chunks}个数据集: {chunk_name}')

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

                self.model.train()
                chunk_total_loss = 0.0

                for epoch in range(SG_EPOCHS_PER_CHUNK):
                    epoch_loss = batch_loss = 0.0
                    print(f'第 {chunk_idx+1} 个数据集 - Epoch {epoch+1}')
                    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                        batch_x = batch_x.to(DEVICE)
                        batch_y = batch_y.to(DEVICE)

                        # 前向传播
                        with autocast():
                            pred = self.model(batch_x)
                            loss = self.loss_fn(pred, batch_y)
                            loss = loss / SG_GRADIENT_ACCUMULATION_STEPS
                        
                        # 反向传播
                        scaler.scale(loss).backward()

                        # 更新参数
                        if (batch_idx + 1) % SG_GRADIENT_ACCUMULATION_STEPS == 0:
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad()
                            self.total_batches += 1
                        
                        batch_loss = loss.item() * SG_GRADIENT_ACCUMULATION_STEPS
                        epoch_loss += batch_loss

                        # 打印进度
                        if batch_idx % 100 == 0:
                            avg_batch_loss = epoch_loss / (batch_idx + 1)
                            print(f"  数据集编号: {chunk_idx+1} | batch: {batch_idx:4} | Epoch: {epoch+1} | Epoch Loss: {epoch_loss:.4f} | Avg Loss: {avg_batch_loss:.4f}")
                    chunk_total_loss += epoch_loss

                # 当前chunk训练完成
                chunk_avg_loss = chunk_total_loss / (len(dataloader) * SG_EPOCHS_PER_CHUNK)
                print(f'第 {chunk_idx+1} 个数据集训练完成   平均Loss: {chunk_avg_loss:.6f}')

                # 验证集评估
                print('\n进行验证集评估...')
                valid_acc = self.validate_model()
                print(f'验证集准确率: {100*valid_acc:.2f}%')

                # 保存最佳模型
                if valid_acc > self.best_vali_acc:
                    self.best_vali_acc = valid_acc
                    print(f'保存最佳模型 (Best Acc: {100*self.best_vali_acc:.2f}%)')
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_vali_acc': self.best_vali_acc,
                        'trained_chunks': chunk_idx + 1,
                        'total_batches': self.total_batches
                    }, SG_BEST_MODEL_PATH)  # 单独保存一个best模型

                # 更新学习率
                scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f'当前学习率：{current_lr:.8f}')

                log_info = (
                    f"===== Chunk {chunk_idx+1}/{self.total_chunks} =====\n"
                    f"训练集：{chunk_name}\n"
                    f"训练平均Loss: {chunk_avg_loss:.6f}\n"
                    f"验证准确率: {100*valid_acc:.2f}%\n"
                    f"当前学习率: {current_lr:.8f}\n"
                )

                with open(LOG_PATH, 'a') as f:
                    f.write(log_info)
                
                # 保存阶段性模型
                if chunk_avg_loss < self.best_loss:
                    self.best_loss = chunk_avg_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': self.best_loss,
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
        
        print(f'\n {self.total_chunks} 个数据集训练完成 | 最优Loss: {self.best_loss:.6f}')
        with open(LOG_PATH, 'a') as f:
            f.write(f'训练完成 | 总Chunks: {self.total_chunks} | 最优Loss: {self.best_loss:.6f}\n')
    
    def train_per_chunk_optimized(self):
        '''每个chunk训练一次并评估模型(优化版本)'''

        for global_epoch in range(0, SG_NUM_GLOBAL_EPOCHS):
            print(f"\n>>> 【全局 Epoch {global_epoch + 1}/{SG_NUM_GLOBAL_EPOCHS}】 <<<")
            self.global_epoch = global_epoch

            # 打乱chunk顺序 
            current_order = self.chunk_files.copy()
            random.shuffle(current_order)
            print("  已打乱Chunk顺序")

            for idx_in_order, chunk_path in enumerate(current_order):
                self.cur_chunk = idx_in_order + 1
                chunk_name = os.path.basename(chunk_path)
                print(f"\n  --- (Global {idx_in_order+1}/{self.total_chunks}) ---")
                print(f"  文件: {chunk_name}")

                try:
                    avg_loss = self._train_one_chunk(chunk_path)
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(f"  训练完成 | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

                    # 定期验证
                    should_validate = (idx_in_order + 1) % SG_VALID_INTERNAL == 0
                    is_last_chunk = (idx_in_order + 1) == self.total_chunks

                    if should_validate or is_last_chunk:
                        print("\n  正在验证...")
                        val_acc = self.validate_model()
                        print(f"  验证结果 | Top-1: {100*val_acc:.2f}%")

                        # 保存最佳模型
                        if val_acc > self.best_vali_acc:
                            self.best_vali_acc = val_acc
                            print(f"  保存最佳模型 (Best: {100*self.best_vali_acc:.2f}%)")
                            torch.save({
                                'model_state_dict': self.model.state_dict(),
                                'global_epoch': global_epoch,
                                'best_vali_acc': self.best_vali_acc,
                                'total_batches': self.total_batches
                            }, SG_BEST_MODEL_PATH)
                        
                        with open(LOG_PATH, 'a', encoding='utf-8') as f:
                            f.write(f"[GE{global_epoch+1} C{idx_in_order+1}] Loss={avg_loss:.4f} | Val1={100*val_acc:.2f}% | Best={100*self.best_vali_acc:.2f}%\n")
                        
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'global_epoch': global_epoch,
                        'best_vali_acc': self.best_vali_acc,
                        'total_batches': self.total_batches
                    }, SG_SAVE_MODEL_PATH)
                
                except Exception as e:
                    print(f'训练出错: {e}')
                    continue
                finally:
                    if DEVICE == 'cuda': torch.cuda.empty_cache()
                    gc.collect()
            
            print(f"\n训练完成 最佳验证准确率: {100*self.best_vali_acc:.2f}%")

    def _train_one_chunk(self, chunk_path):
        dataset = SgChunkDataset(chunk_path, is_train=True)
        loader = DataLoader(
            dataset=dataset,
            batch_size=SG_BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            with autocast():
                pred = self.model(x)
                loss = self.loss_fn(pred, y) / SG_GRADIENT_ACCUMULATION_STEPS
            
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % SG_GRADIENT_ACCUMULATION_STEPS == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), SG_GRADIENT_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.total_batches += 1
            
            total_loss += loss.item() * SG_GRADIENT_ACCUMULATION_STEPS
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f'epoch: {self.global_epoch} | chunk: {self.cur_chunk} | batch%: {100*(batch_idx+1)/16148:.3f}% | total loss: {total_loss:.4f} | LR: {current_lr:.8f}')
        
        avg_loss = total_loss / len(loader)
        dataset.close()
        del dataset, loader
        return avg_loss


class VaDataset(Dataset):
    '''终局价值网络数据集'''
    def __init__(self, inputs, labels, train=True):
        self.inputs = inputs.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.train = train

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]

        if self.train:   # 数据增强
            k = np.random.randint(0, 4)
            x = np.rot90(x, k=k, axes=(1, 2))
            if np.random.randint(0, 2):
                x = np.flip(x, axis=2)
        
        x = x.copy()
        y = y.copy()

        return torch.tensor(x), torch.tensor(y)
        
class VaModelTrain:
    '''终局价值网络训练类'''
    def __init__(self, model=VA_TRAIN_MODEL, savepath=VA_SAVE_MODEL_PATH, target_epoch=-1):
        self.model = model.to(DEVICE)
        self.savepath = savepath
        self.target_epoch = target_epoch

        data = np.load(VA_DATASET_PATH)
        all_inputs = data['inputs']
        all_labels = data['values']
        train_inputs = all_inputs[:-5000]
        train_labels = all_labels[:-5000]
        vali_inputs = all_inputs[-5000:]  # 后1000个样本作为验证集
        vali_labels = all_labels[-5000:]
        self.train_dataset = VaDataset(train_inputs, train_labels)
        self.vali_dataset = VaDataset(vali_inputs, vali_labels, train=False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=VA_BATCH_SIZE,
            shuffle=True,
            pin_memory=True
        )
        self.vali_loader = DataLoader(
            self.vali_dataset,
            batch_size=VA_BATCH_SIZE,
            shuffle=False,
            pin_memory=True
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=VA_LEARNING_RATE,
            weight_decay=VA_WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=16, eta_min=1e-6)

        print(f"训练集样本数：{len(self.train_dataset)}")

    def validate_model(self):
        '''在验证集上评估模型 返回准确率和loss'''
        self.model.eval()
        total_correct = 0
        vali_loss = 0.0
        with torch.no_grad():
            for x, y in self.vali_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = self.model(x)
                
                loss = self.loss_fn(pred, y)
                vali_loss += loss.item() * x.size(0)

                pred_agrmax = torch.argmax(pred, dim=1)
                y_argmax = torch.argmax(y, dim=1)
                total_correct += (pred_agrmax == y_argmax).sum().item()
        
        acc = total_correct / 5000
        vali_loss /= len(self.vali_dataset)
        return acc, vali_loss

    def train(self):
        '''训练直到验证集准确率达到要求'''
        best_val_loss = float('inf')
        epoch = 0
        patience = 0
        while True:
            epoch += 1
            self.model.train()
            train_loss = 0.0
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                self.optimizer.zero_grad()

                pred = self.model(batch_x)
                loss = self.loss_fn(pred, batch_y)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
                if batch_idx % 100 == 0:
                    print(f'batch: {batch_idx:4} | Epoch: {epoch}')

            train_loss /= len(self.train_dataset)

            vali_acc, vali_loss = self.validate_model()
            self.scheduler.step()

            if vali_loss < best_val_loss:
                best_val_loss = vali_loss
                torch.save(self.model.state_dict(), self.savepath)
                print(f'保存模型 验证集损失: {best_val_loss:.4f}')
                patience = 0
            else:
                patience += 1
            
            log_info = (
                f'Epoch {epoch:2d}==== {self.target_epoch} ====== \n'
                f'Train Loss: {train_loss:.4f} \n'
                f'best vali loss: {best_val_loss:.4f} \n'
                f'Vali Loss: {vali_loss:.4f} \n'
                f'Vali acc: {vali_acc:.4f} \n'
                f'patience: {patience}\n'
            )

            print(log_info)
            with open(LOG_PATH, 'a') as f:
                f.write(log_info)

            if self.target_epoch != -1:
                if epoch >= self.target_epoch:
                    break
            else:
                if vali_acc >= 0.92 or epoch >= 70 or patience >= VA_PATIENCE:
                  break
        
        print(f"\n训练完成 最优验证损失: {best_val_loss:.4f}")
        


if __name__ == "__main__":
    sg_train = SgModelTrain()
    sg_train.train_per_chunk_optimized()
    # # 固定20epoch
    # va_train_1 = VaModelTrain(savepath='D:/01_EGGER/program/python/simple-inteligent-go/output/go_final_val_model_1_1.pth', target_epoch=20)
    # va_train_1.train()
    # 目标训练
    # va_train = VaModelTrain()
    # va_train.train()