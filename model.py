import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from torchvision.models import resnet18

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingModel, self).__init__()
        
        self.base_model = resnet18(pretrained=False)
        self.base_model.fc = nn.Identity()
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding_layer(x)
        return x

from torch.utils.data import Dataset

class NormalizeWindow:
    def __call__(self, eeg_window):
        mean = np.mean(eeg_window, axis=1, keepdims=True)
        std = np.std(eeg_window, axis=1, keepdims=True)
        return (eeg_window - mean) / (std + 1e-6)

class EEGWindowDataset(Dataset):
    def __init__(self, data, segment_length=128, stride=4, transform=None):
        
        self.segment_length = segment_length
        self.transform = transform if transform else NormalizeWindow()
        self.stride = stride
        self.segment_length = segment_length
        self.data = data

    def __getitem__(self, index):

        anchor = self.data[:, index*self.stride:index*self.stride + self.segment_length]
        anchor = self.transform(anchor)
        
        return torch.tensor(anchor, dtype=torch.float32).unsqueeze(0)
        

    def __len__(self):
        return (self.data.shape[1] - self.segment_length) // self.stride + 1