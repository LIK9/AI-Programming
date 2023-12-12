import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([1, 2, 3]).view(-1, 1)
y_train = torch.FloatTensor([2, 4, 6]).view(-1, 1)
nb_epochs = 1000
# lr = 0.01

W = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)
optimizer = torch.optim.SGD([W, b], lr = 0.01)

for epoch in range(nb_epochs+1):
  hypo = x_train * W + b
  # cost = torch.mean((y_train - hypo)**2)
  cost = torch.nn.functional.mse_loss(hypo, y_train)

  optimizer.zero_grad() # 미분값 초기화
  cost.backward() # 현재 손실 함숫값을 바탕으로 미분값 계산
  optimizer.step() # 현재 미분값 및 학습률을 바탕으로 변수 업데이트 1회 수행

  # print(f'epoch: {epoch} W: {W.item():.3f} b: {b.item():.3f} cost: {cost.item():.6f}')

  if epoch % 100 == 0:
    print(f'epoch: {epoch} W: {W.item():.3f} b: {b.item():.3f} cost: {cost.item():.6f}' )
