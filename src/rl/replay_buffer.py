'''经验回放缓冲区。

滑动窗口缓冲池，自动丢弃旧数据，保证训练数据的新鲜度和多样性。
'''

import random
import numpy as np


class ReplayBuffer:
    '''固定容量的经验回放缓冲区。

    存储自对弈产生的 (局面, 落子, 奖励) 三元组，
    超出容量时自动丢弃最旧数据。

    用法::

        buf = ReplayBuffer(capacity=500000)
        buf.push(board, move_idx, reward)
        batch = buf.sample(256)
    '''

    def __init__(self, capacity: int = 500000):
        self.capacity = capacity
        self.boards: list = []       # numpy arrays (2, 19, 19)
        self.moves: list = []        # int 0~361
        self.rewards: list = []      # float
        self._pos = 0

    # ------------------------------------------------------------------
    def push(self, board: np.ndarray, move_idx: int, reward: float):
        '''存入一步数据。board 形状 (2, 19, 19)。'''
        if len(self.boards) < self.capacity:
            self.boards.append(board)
            self.moves.append(move_idx)
            self.rewards.append(reward)
        else:
            self.boards[self._pos] = board
            self.moves[self._pos] = move_idx
            self.rewards[self._pos] = reward
            self._pos = (self._pos + 1) % self.capacity

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> tuple:
        '''随机抽取 batch。

        Returns:
            (boards, moves, rewards)
            boards: (B, 2, 19, 19) numpy array
            moves: (B,) numpy array of int
            rewards: (B,) numpy array of float
        '''
        indices = random.choices(range(len(self.boards)), k=batch_size)
        batch_boards = np.stack([self.boards[i] for i in indices])
        batch_moves = np.array([self.moves[i] for i in indices], dtype=np.int64)
        batch_rewards = np.array([self.rewards[i] for i in indices], dtype=np.float32)
        return batch_boards, batch_moves, batch_rewards

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.boards)

    def is_ready(self, min_size: int = 1000) -> bool:
        return len(self) >= min_size
