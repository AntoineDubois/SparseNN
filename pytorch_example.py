import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from pathlib import Path

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def loss(self, output, target):
        return F.nll_loss(output, target)

class NetSparse(nn.Module):
    def __init__(self):
        super(NetSparse, self).__init__()
        self.l1_weight = 0.001
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def loss(self, output, target):
        l1_penalty = torch.autograd.Variable( torch.FloatTensor(1), requires_grad=True)
        for W in self.parameters():
            l1_penalty = l1_penalty + W.norm(1)
        return F.nll_loss(output, target) + self.l1_weight * l1_penalty


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, target) 
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # Training settings
    batch_size = 1#64
    test_batch_size = 1000
    epochs = 14
    lr = 1.0    
    gamma = 0.7    
    seed = 1
    log_interval = 10
    save_model = True
    save_model_scripted = True

    torch.manual_seed(seed)    
    device = "cpu" # get_device()

    # prepare data set
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size)

    # load model and train
    model_names_list = ["mnist_cnn_sparse", "mnist_cnn"]
    for model_name in model_names_list:
        if Path("./weights/"+ model_name +".pth").is_file():
            continue
        
        if model_name == "mnist_cnn":
            model = Net()
        elif model_name == "mnist_cnn_sparse":
            model = NetSparse()
        
        model.to(device)
            
        optimizer = optim.Adadelta(model.parameters(), lr=lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, log_interval)
            test(model, device, test_loader)
            scheduler.step()

        if save_model:
            torch.save(model.state_dict(), "./weights/"+ model_name +".pth")
            print("model", model_name, "saved")
    
        if save_model_scripted:
            model_scripted = torch.jit.script(model)
            model_scripted.save("./weights/" + model_name + ".pt")


    