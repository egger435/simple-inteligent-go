import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_DROPOUT_RATE = 0.2

class GoCNN(nn.Module):
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

        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 512)

        # 输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 362)  # 361个棋盘位置 + 1个pass
    
    def forward(self, x):
        # 卷积特征提取
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # 全局平均池化
        x = self.pool(x)
        x = x.flatten(1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = GoCNN()
    print(model)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters()):,} 个')

    # 测试前向传播
    test_input = torch.randn(1, 2, 19, 19)
    test_output = model(test_input)
    print(f'输入形状: {test_input.shape}', f'输出形状: {test_output.shape}')
