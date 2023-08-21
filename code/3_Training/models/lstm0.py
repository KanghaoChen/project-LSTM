import torch
from torch import nn

import pdb


class lstm0Classifier(nn.Module):

    def __init__(self, dropout: float):
        # batch_size by timepoints by bands by bins
        super().__init__()
        self.flatten = nn.Flatten(0, 1)
        self.conv1d = nn.Conv1d(1, 64, 3)
        self.max1d = nn.MaxPool1d(kernel_size=15) # pooling layer
        
        self.lstm = nn.LSTM(64, 64, batch_first = True)
        self.dropout = nn.Dropout(p = dropout)
        self.linear1 = nn.Linear(64, 256)
        self.linear2 = nn.Linear(256, 32)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(32, 2)

    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        size = data.size()
        x = self.flatten(data).unsqueeze(dim=1)
        x = self.conv1d(x)
        x = self.max1d(x)
        x = x.view((size[0], size[1], -1))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x