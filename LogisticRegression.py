import torch

class LogisticRegression(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(2, 1)
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    return self.sigmoid(self.linear(x))

x_train = torch.FloatTensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]) # 6, 2
y_train = torch.FloatTensor([0, 0, 0, 1, 1, 1]).view(-1, 1) # 6, 1
nb_epochs = 1000
# lr = 1e-1

model = LogisticRegression()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)

print(list(model.parameters())[0].shape)

for epoch in range(nb_epochs+1):
  prediction = model(x_train)
  cost = torch.nn.functional.binary_cross_entropy(prediction, y_train)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 100 == 0:
    prediction_class = prediction >= torch.FloatTensor([0.5])
    print(prediction_class)
    correct_prediction = prediction_class.float() == y_train
    accuracy = correct_prediction.float().sum() / len(correct_prediction)
    print(f'{epoch} : {cost.item():.6f}, {accuracy}')

print((model(torch.FloatTensor([[5, 5], [1, 4]])) >= torch.FloatTensor([0.5])).float())

# x 6, 2
# W 1, 2
