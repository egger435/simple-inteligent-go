'''策略梯度训练器。

从经验回放缓冲区采样，用 REINFORCE 算法更新策略网络。
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import BOARD_SIZE, PASS_LABEL, get_strategy_model, get_value_model


class RLTrainer:
    '''REINFORCE 策略梯度训练器。

    用法::

        trainer = RLTrainer(device='cuda', lr=1e-5)
        loss = trainer.train_step(boards, moves, rewards)
    '''

    def __init__(self, device: str = 'cuda', lr: float = 1e-5):
        self.device = device
        self.policy_net = get_strategy_model()
        self.policy_net.train()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self._total_steps = 0

    # ------------------------------------------------------------------
    def train_step(self, boards: np.ndarray, moves: np.ndarray,
                   rewards: np.ndarray) -> dict:
        '''单步策略梯度更新。

        Args:
            boards: (B, 2, 19, 19) numpy 局面张量
            moves: (B,) numpy 落子索引
            rewards: (B,) numpy 奖励值

        Returns:
            {'loss': float, 'mean_reward': float}
        '''
        boards_t = torch.tensor(boards, dtype=torch.float32).to(self.device)
        moves_t = torch.tensor(moves, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # 标准化奖励（稳定训练）
        if rewards_t.std() > 0:
            rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        # 前向
        logits = self.policy_net(boards_t)       # (B, 362)
        log_probs = nn.functional.log_softmax(logits, dim=1)

        # 选中落子的 log 概率
        selected_log_probs = log_probs.gather(1, moves_t.unsqueeze(1)).squeeze(1)

        # REINFORCE 损失
        loss = -(selected_log_probs * rewards_t).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self._total_steps += 1

        return {
            'loss': loss.item(),
            'mean_reward': rewards_t.mean().item(),
        }

    # ------------------------------------------------------------------
    def get_model_state(self) -> dict:
        '''获取当前模型参数（用于保存 + Arena 比对）。'''
        return {k: v.cpu().clone() for k, v in self.policy_net.state_dict().items()}

    # ------------------------------------------------------------------
    def load_model_state(self, state_dict: dict):
        '''加载模型参数。'''
        self.policy_net.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str):
        '''保存检查点。'''
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self._total_steps,
        }, path)

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str):
        '''加载检查点。'''
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self._total_steps = ckpt.get('total_steps', 0)

    # ------------------------------------------------------------------
    def eval(self):
        '''切换到评估模式。'''
        self.policy_net.eval()

    def train(self):
        '''切换到训练模式。'''
        self.policy_net.train()
