import torch
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

mnist_train = torchvision.datasets.MNIST(root = 'MNIST_data/',
                                         train = True,
                                         transform = torchvision.transforms.ToTensor(),
                                         download = True) # 60000, 28*28

mnist_test = torchvision.datasets.MNIST(root = 'MNIST_data/',
                                        train = False,
                                        transform = torchvision.transforms.ToTensor(),
                                        download = True) # 10000, 28*28
training_epochs = 15
batch_size = 100 # iteration 600

dataloader = DataLoader(mnist_train, batch_size = batch_size, shuffle = True)

class MNIST_SoftRegression(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(28*28, 10)
  def forward(self, x):
    return self.linear(x)

model = MNIST_SoftRegression()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
print(len(dataloader))
for epoch in range(training_epochs):
  avg_cost = 0
  correct_prediction = 0
  for idx, samples in enumerate(dataloader): # 600번 반복
    x_train, y_train = samples
    x_train = x_train.view(100, 28*28)
    # y_train 100

    prediction = model(x_train) # 100, 10
    # W 10, 28*28
    # b 10

    cost = torch.nn.functional.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_cost += (cost / 600 )

    correct_prediction += (torch.argmax(prediction, 1) == y_train).float().mean().item() / 600
    # print(len((torch.argmax(prediction, 1) == y_train).float()))

  # print(f'{epoch+1}: {avg_cost}, {correct_prediction * 100}%')
  print(f'{avg_cost:.6f}')

# x 100, 784
# W 10, 784
# b 10
# y 100
# prediction 100, 10
