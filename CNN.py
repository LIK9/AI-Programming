import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

training_epochs = 3
batch_size = 100
mnist_train = dsets.MNIST(root = 'MNIST_data/',
                          train = True,
                          transform = transforms.ToTensor(),
                          download = True)
mnist_test = dsets.MNIST(root = 'MNIST_data/',
                         train = False,
                         transform = transforms.ToTensor(),
                         download = True)

data_loader = DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle = True)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# torch.manual_seed(777)
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
      )
    self.layer2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )
    self.fc = nn.Sequential(nn.Linear(7*7*64, 100), # W = 10 X 7*7*64
                            nn.ReLU(),
                            nn.Linear(100, 10)) # b 10
  def forward(self, x):
    out=  self.layer1(x)
    out = self.layer2(out)
    out = out.view(-1, 7*7*64)
    out = self.fc(out)
    return out
model = CNN().to(device)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(training_epochs):
  avg_cost = 0
  total_batch = len(data_loader)
  for idx, (x_train, y_train) in enumerate(data_loader):
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    prediction = model(x_train)
    cost = criterion(prediction, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch
  print(avg_cost.item())

