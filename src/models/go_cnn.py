import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_DROPOUT_RATE = 0.2

class GoCNN(nn.Module):
    '''简易三层卷积网络'''
    def __init__(self, dropout_rate=TRAIN_DROPOUT_RATE):
        super().__init__()

        # 三层特征提取卷积
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(64*19*19, 512)

        # 输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 362)  # 361个棋盘位置 + 1个pass
    
    def forward(self, x):
        # 卷积特征提取
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.flatten(1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
class GoResNet(nn.Module):
    """残差网络"""
    def __init__(self):
        super().__init__()
        # 输入层：2通道 - 64通道
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 残差块
        self.res_block1 = self._make_res_block(64, 64)
        self.res_block2 = self._make_res_block(64, 64)
        self.res_block3 = self._make_res_block(64, 64)
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),  # 19×19×32 → 11552
            nn.Linear(11552, 361),  # 361个落子位置
            nn.Softmax(dim=1)  # 直接输出概率
        )
        # 弃行分支
        self.pass_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _make_res_block(self, in_channels, out_channels):
        """残差块 卷积+BN+ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # 输入层
        x = self.input_conv(x)
        # 残差块
        x = x + self.res_block1(x)
        x = x + self.res_block2(x)
        x = x + self.res_block3(x)
        # 落子概率 + 弃行概率
        policy = self.policy_head(x)  # (batch, 361)
        pass_prob = self.pass_head(x)  # (batch, 1)
        # 拼接为362维输出
        output = torch.cat([policy, pass_prob], dim=1)
        return output

# 先定义残差块（解决梯度消失，传递全局特征）
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 跳过连接（保证梯度传递）
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # 残差连接：保留全局特征
        return F.relu(x)

# 改进后的GoCNN（保留你的核心结构，新增全局特征）
class GoCNN_p(nn.Module):
    '''增强全局特征的卷积网络（适配征子等全局判断）'''
    def __init__(self, dropout_rate=0.1):
        super().__init__()

        # 1. 输入层（不变）
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 2. 替换原卷积块为残差块（增加全局特征传递）
        self.res_block1 = ResBlock(64, 128)  # 64→128，扩大通道
        self.res_block2 = ResBlock(128, 128) # 保持通道，强化特征
        self.res_block3 = ResBlock(128, 64)  # 128→64，降回原通道

        # 3. 新增全局特征融合层（关键！捕捉19×19全局关联）
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化（提取全局特征）
        self.global_fc = nn.Linear(64, 64*19*19)   # 全局特征映射回空间维度

        # 4. 原全连接层（微调，融合全局+局部特征）
        self.fc1 = nn.Linear(64*19*19 * 2, 512)  # ×2：局部特征+全局特征
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 362)

    def forward(self, x):
        # 局部特征提取（原逻辑）
        x_local = self.conv_block1(x)
        x_local = self.res_block1(x_local)
        x_local = self.res_block2(x_local)
        x_local = self.res_block3(x_local)  # (batch, 64, 19, 19)

        # 全局特征提取+融合（新增！）
        x_global = self.global_pool(x_local)  # (batch, 64, 1, 1)
        x_global = x_global.flatten(1)        # (batch, 64)
        x_global = self.global_fc(x_global)   # (batch, 64*19*19)
        x_global = x_global.reshape(-1, 64, 19, 19)  # 映射回空间维度

        # 融合局部+全局特征（关键：让模型同时看局部和全局）
        x_fuse = torch.cat([x_local.flatten(1), x_global.flatten(1)], dim=1)

        # 全连接层（原逻辑）
        x = F.relu(self.fc1(x_fuse))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = GoCNN_p()
    print(model)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters()):,} 个')

    # 测试前向传播
    test_input = torch.randn(1, 2, 19, 19)
    test_output = model(test_input)
    print(f'输入形状: {test_input.shape}', f'输出形状: {test_output.shape}')
