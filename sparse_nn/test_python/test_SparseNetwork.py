import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from sparse_network import SparseNetwork
sys.path.append('../python_build')
import SparseNN

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

sparse_network = SparseNetwork(module, verbose=True)


array = SparseNN.ArrayFloat(28, 28)
array.fill(0.1)

out = sparse_network(array)
print(out)

#out = sparse_network.Forward(array)
#print(out)