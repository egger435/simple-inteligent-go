import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from models.go_cnn import GoCNN

NPZ_DATA_PATH = 'E:/go_dataset/go_dataset.npz'

class GoNPZDataset(Dataset):
    def __init__(self, npz_data_path=NPZ_DATA_PATH):
        self.data = np.load(npz_data_path)