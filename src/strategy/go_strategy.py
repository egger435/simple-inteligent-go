'''落子策略选择'''
import numpy as np
import torch
from sgfmill import boards
from common import *

class GoStrategySelector:
    '''落子策略选择类'''
    def __init__(self):
        self.model = STRATEGY_MODEL.to(DEVICE)
        self._load_model()
        self.model.eval()
    
    def _load_model(self):
        checkpoint = torch.load(STRATEGY_MODEL_PATH, map_location=DEVICE)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
    
    def _preprocess_input(self, cur_board: boards.Board, cur_color: chr):
        '''将当前棋盘状态和行棋方信息转化为模型输入格式'''
        # 生成棋盘矩阵
        board_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = cur_board.get(r, c)
                if stone == 'b':
                    board_matrix[r, c] = 1.0
                elif stone == 'w':
                    board_matrix[r, c] = 2.0
        board_matrix = board_matrix / 2.0  # 归一化

        # 生成行棋方通道
        color_code = 1 if cur_color == 'b' else 2
        player_channel = np.full((BOARD_SIZE, BOARD_SIZE), COLOR_MAP[color_code], dtype=np.float32)

        # 合并得到模型输入
        model_input = np.stack([board_matrix, player_channel], axis=0)
        model_input = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return model_input

    def predict(self, cur_board: boards.Board, cur_color: chr):
        '''根据当前局面选择下一手位置'''
        input_tensor = self._preprocess_input(cur_board, cur_color)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.softmax(output, dim=1)
        
        prob_np = prob.cpu().numpy()[0]
        sorted_indices = np.argsort(prob_np)[::-1]  # 根据概率排序
        
        def _is_legal(pos):
            '''检查合法落子'''
            occpuied_points, _ = cur_board.list_occupied_points()
            for point in occpuied_points:
                if pos == point[1]:
                    return False
            return True

        candidates = []
        for idx in sorted_indices:
            if idx == PASS_LABEL:
                candidates.append(('pass', prob_np[idx]))
                continue
            r = idx // BOARD_SIZE
            c = idx % BOARD_SIZE
            if _is_legal((r, c)):
                candidates.append(((r, c), prob_np[idx]))
            if len(candidates) >= TOP_K:
                break
        
        return candidates

    