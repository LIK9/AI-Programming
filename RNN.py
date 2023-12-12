import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
import string
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')
!cp '/content/drive/MyDrive/IMDB Dataset.csv.zip' ./
!unzip 'IMDB Dataset.csv.zip'

is_cuda = torch.cuda.is_available()
device = torch.device("cuda")
print(device)

base_csv = 'IMDB Dataset.csv'
df = pd.read_csv(base_csv)
df.head()

X, y = df['review'].values, df['sentiment'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(x_train.shape)
print(y_test.shape)

def processing_string(s):
  s = re.sub(r"[^\w\s]", '', s)
  s = re.sub(r"\s+", '', s)
  s = re.sub(r"\d", '', s)
  return s

def tockenize(x_train, y_train, x_val, y_val):
  word_list = []
  stop_words = set(stopwords.words('english'))
  for sent in x_train:
    for word in sent.lower().split():
      word = processing_string(word)
      if word not in stop_words and word != "":
        word_list.append(word)
  corpus = Counter(word_list)
  corpus_ = sorted(corpus, key = corpus.get, reverse = True)[:1000]
  onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
  final_list_train, final_list_test = [], []
  for sent in x_train:
    final_list_train.append([onehot_dict[processing_string(word)] for word in sent.lower().split()
    if processing_string(word) in onehot_dict.keys()])
  for sent in x_val:
    final_list_test.append([onehot_dict[processing_string(word)] for word in sent.lower().split()
    if processing_string(word) in onehot_dict.keys()])
  encoded_train = [1 if label == 'positive' else 0 for label in y_train]
  encoded_test = [1 if label == 'positive' else 0 for label in y_val]
  return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(encoded_test), onehot_dict

x_train, y_train, x_test, y_test, vocab = tockenize(x_train, y_train, x_test, y_test)
print(len(vocab))

def padding_(sentences, seq_len):
  features = np.zeros((len(sentences), seq_len), dtype = int)
  for ii, review in enumerate(sentences):
    if len(review) != 0:
      features[ii, -len(review):] = np.array(review)[:seq_len]
  return features
print(x_train)
x_train_pad = padding_(x_train, 200)
x_test_pad = padding_(x_test, 200)
print(x_train_pad.shape)

train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))
batch_size = 50
train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size)
test_loader = DataLoader(test_data, shuffle = True, batch_size = batch_size)

dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)
print(sample_x.size())
print(sample_y.shape)

class GRU_model(nn.Module):
  def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, device):
    super(GRU_model, self).__init__()
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.device = device

    self.embed = nn.Embedding(n_vocab, embed_dim)
    self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
    self.out = nn.Linear(self.hidden_dim, n_classes)
  def forward(self, x):
    x = self.embed(x)
    h_0 = self._init_state(batch_size = x.size(0))
    x, _ = self.gru(x, h_0)
    h_t = x[:, -1, :]
    logit = self.out(h_t)
    return logit
  def _init_state(self, batch_size):
    new_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
    return new_state

n_layers = 1
vocab_size = len(vocab) + 1
hidden_dim = 128
embed_dim = 100
n_classes = 2
model = GRU_model(n_layers, hidden_dim, vocab_size, embed_dim, n_classes, device).to(device)

def train(model, criterion, optimizer, data_loader):
  model.train()
  train_loss = 0
  for i, (x, y) in enumerate(data_loader):
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logit = model(x)
    loss = criterion(logit, y)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * x.size(0)
  return train_loss / len(data_loader.dataset)
def evaluate(model, data_loader):
  model.eval()
  corrects, total_loss = 0, 0
  for i, (x, y) in enumerate(data_loader):
    x, y = x.to(device), y.to(device)
    logit = model(x)
    corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
  size = len(data_loader.dataset)
  avg_accuracy = 100.0*corrects / size
  return avg_accuracy

num_epoch = 10
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
for e in range(1, num_epoch+1):
  train_loss = train(model, criterion, optimizer, train_loader)
  test_accuracy = evaluate(model, test_loader)
  print(train_loss, test_accuracy)

class LSTM_model(nn.Module):
  def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, device):
    super(LSTM_model, self).__init__()
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.device = device
    self.embed = nn.Embedding(n_vocab, embed_dim)
    self.lstm = nn.LSTM(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
    self.out = nn.Linear(self.hidden_dim, n_classes)

  def forward(self, x):
    x = self.embed(x)
    h_0 = self._init_state(batch_size = x.size(0))
    x, _ = self.lstm(x, h_0)
    print(x.shape)
    h_t = x[:, -1, :]
    logit = self.out(h_t)
    return logit
  def _init_state(self, batch_size):
    new_cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
    new_hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
    return (new_hidden_state, new_cell_state)

model = LSTM_model(n_layers, hidden_dim, vocab_size, embed_dim, n_classes, device).to(device)
critetion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=  lr)
for e in range(1, num_epoch+1):
  train_loss = train(model, critetion, optimizer, train_loader)
  test_accuracy = evaluate(model, test_loader)
  print(train_loss, test_accuracy)

class RNN_model(nn.Module):
  def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, device):
    super(RNN_model, self).__init__()
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.device = device
    self.embed = nn.Embedding(n_vocab, embed_dim)
    self.rnn = nn.RNN(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
    self.out = nn.Linear(self.hidden_dim, n_classes)
  def forward(self,x):
    x = self.embed(x)
    h_0 = self._init_state(batch_size = x.size(0))
    x, _ = self.rnn(x, h_0)
    h_t = x[:, -1, :]
    logit = self.out(h_t)
    print(logit.shape)
    return logit
  def _init_state(self, batch_size):
    new_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
    return new_state

model = RNN_model(n_layers, hidden_dim, vocab_size, embed_dim, n_classes, device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
for e in range(num_epoch):
  train_loss = train(model, criterion, optimizer, train_loader)
  test_accuracy = evaluate(model, test_loader)
  print(train_loss, test_accuracy)
