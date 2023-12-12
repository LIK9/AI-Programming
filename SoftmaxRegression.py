import torch

x_train = torch.FloatTensor([[6.3, 3.3, 6.0, 2.5],
                             [5.8, 2.7, 5.1, 1.9],
                             [7.1, 3.0, 5.9, 2.1],
                             [5.1, 3.5, 1.4, 0.2],
                             [4.9, 3.0, 1.4, 0.2],
                             [4.7, 3.2, 1.3, 0.2],
                             [7.0, 3.2, 4.7, 1.4],
                             [6.4, 3.2, 4.5, 1.5],
                             [6.9, 3.1, 4.9, 1.5]]) # 9, 4

y_train = torch.LongTensor([0, 0, 0, 1, 1, 1 , 2, 2, 2]) # 9
nb_epochs = 1000
# lr = 1e-1

class SoftmaxRegression(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(4, 3)
  def forward(self, x):
    return self.linear(x)


model = SoftmaxRegression()

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(nb_epochs+1):
  # prediction = torch.nn.functional.softmax(model(x_train), dim = 1)
  prediction = model(x_train) # 9, 3

  # cost = torch.nn.functional.cross_entropy(prediction, y_train)
  cost = criterion(prediction, y_train)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 100 == 0:
    correct_prediction = torch.argmax(prediction, 1) == y_train
    accuracy = correct_prediction.float().sum() / len(correct_prediction)
    print(correct_prediction.shape)

    print(f'{epoch} : {accuracy} {len(correct_prediction)}')

# x 9, 4
# W 3, 4
# b 3
# y 9
# prediction 9, 3
