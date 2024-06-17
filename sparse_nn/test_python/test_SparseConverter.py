import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from sparse_converter import SparseConverter

class NetSparse(nn.Module):
    def __init__(self):
        super(NetSparse, self).__init__()
        self.l1_weight = 0.001
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

module = NetSparse()

converter = SparseConverter(module, verbose=True)