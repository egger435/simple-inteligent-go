'''终局价值判断网络'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out
    
class GoValueNet(nn.Module):
    '''终局价值判断模型'''
    def __init__(
            self,
            num_res_blocks = 6,
            channels = 128,
            dropout_rate = 0.3
    ):
        super().__init__()
        self.conv_initial = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_head = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.conv_initial(x)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        x = x.flatten(1)

        logits = self.fc_head(x)
        probs = F.softmax(logits, dim=1)

        return probs
    
if __name__ == '__main__':
    model = GoValueNet()
    print(model)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters()):,} 个')

    # 测试前向传播
    test_input = torch.randn(1, 2, 19, 19)
    test_output = model(test_input)
    print(f'输入形状: {test_input.shape}', f'输出形状: {test_output.shape}')