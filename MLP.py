import torch
import torchvision
from torch.utils.data import DataLoader

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

training_epochs = 3
batch_size =  100
mnist_train = torchvision.datasets.MNIST(root = 'MNIST_data/',
                                         train = True,
                                         transform = torchvision.transforms.ToTensor(),
                                         download = True)
mnist_test = torchvision.datasets.MNIST(root = 'MNIST_data/',
                                        train = False,
                                        transform = torchvision.transforms,
                                        download = True)
data_loader = DataLoader(mnist_train, batch_size = 100, shuffle = True)

class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.model = torch.nn.Sequential(torch.nn.Linear(784, 100),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(100, 100),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(100, 10))
  def forward(self, x):
    return self.model(x)

model = MLP().to(device)
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
for epoch in range(training_epochs):
  avg_cost=  0
  correct_prediction = 0
  for idx, (x_train, y_train) in enumerate(data_loader):
    x_train = x_train.view(100, 784).to(device)
    y_train = y_train.to(device)
    prediction = model(x_train)
    cost = criterion(prediction, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_cost += (cost.item() / 600)
    correct_prediction += (torch.argmax(prediction, 1) == y_train).float().mean() / len(data_loader)

  print(avg_cost, correct_prediction * 100)


# x 100, 784
# W 10, 784
# b 10
# y 100
# prediction 100, 10
