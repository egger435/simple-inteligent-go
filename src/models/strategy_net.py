'''策略选择网络'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GoCNN(nn.Module):
    '''简易三层卷积网络'''
    def __init__(self, dropout_rate=0.2):
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

# 残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # 残差连接
        return F.relu(x)

class GoCNN_p(nn.Module):
    '''增强全局特征的卷积网络'''
    def __init__(self, dropout_rate=0.1):
        super().__init__()

        # 输入层
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 替换原卷积块为残差块
        self.res_block1 = ResBlock(64, 128)  
        self.res_block2 = ResBlock(128, 128) 
        self.res_block3 = ResBlock(128, 64)  

        # 全局特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        self.global_fc = nn.Linear(64, 64*19*19)   

        # 全连接层
        self.fc1 = nn.Linear(64*19*19 * 2, 512)  # 局部特征 全局特征
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 362)

    def forward(self, x):
        # 局部特征提取
        x_local = self.conv_block1(x)
        x_local = self.res_block1(x_local)
        x_local = self.res_block2(x_local)
        x_local = self.res_block3(x_local)  # (batch, 64, 19, 19)

        # 全局特征提取融合
        x_global = self.global_pool(x_local)  # (batch, 64, 1, 1)
        x_global = x_global.flatten(1)        # (batch, 64)
        x_global = self.global_fc(x_global)   # (batch, 64*19*19)
        x_global = x_global.reshape(-1, 64, 19, 19)  # 映射回空间维度

        # 融合局部全局特征
        x_fuse = torch.cat([x_local.flatten(1), x_global.flatten(1)], dim=1)

        # 全连接层
        x = F.relu(self.fc1(x_fuse))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
class AlphaCNN(nn.Module):
    '''AlphaGo同款卷积网络'''
    def __init__(self, NUM_RES_BLOCKS=12):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            *[self._res_block(64) for i in range(NUM_RES_BLOCKS)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16*19*19, 362),
            nn.Softmax(dim=1)
        )
    
    def _res_block(self, chs):
        return nn.Sequential(
            nn.Conv2d(chs, chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(chs, chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(chs),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_blocks(x)
        prob = self.policy_head(x)

        return prob


if __name__ == "__main__":
    model = AlphaCNN()
    print(model)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters()):,} 个')

    # 测试前向传播
    test_input = torch.randn(1, 2, 19, 19)
    test_output = model(test_input)
    print(f'输入形状: {test_input.shape}', f'输出形状: {test_output.shape}')
