from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/ants_bees.zip ./
!unzip ants_bees.zip

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torch.nn as nn
import os

data_transforms = {
    'train': transforms.Compose([
      transforms.Resize(256),
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
data_dir = 'ants_bees'
image_datasets = {x: dsets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size = 4, shuffle = True, num_workers = 4)
for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(image_datasets)

import time
import copy

def train_model(model, criterion, optimizer, num_epochs = 10):
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0

  for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs-1}')
    print('-'*10)

    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()
      running_loss = 0.0
      running_corrects = 0
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          if phase == 'train':
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    print()
  time_elapsed = time.time() - since
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val Acc: {best_acc:.4f}')
  model.load_state_dict(best_model_wts)
  return model

import torchvision.models as models
import torch.optim as optim

model_ft = models.resnet18(pretrained = False, num_classes = 2)
print(model_ft.fc)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)
model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs = 10)

model_ft = models.resnet18(pretrained = True)
print(model_ft.fc)
num_ftrs = model_ft.fc.in_features # 512
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)
model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs = 10)

model_ft = models.efficientnet_v2_s(pretrained = False, num_classes = 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)
model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs = 10)

model_ft = models.efficientnet_v2_s(pretrained = True)
num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)
model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs = 10)
