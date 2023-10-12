import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        print(x.size())
        
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('batch:{} Loss:{:.6f}'.format(batch_idx, loss.item()))

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train = True, download=True, transform = transform)
    print(dataset1)
    
    train_loader = torch.utils.data.DataLoader(dataset1)
    
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr = 1.0)
    
    train(model, train_loader, optimizer)
    

if __name__ == '__main__':
    main()
        