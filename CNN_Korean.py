import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import urllib.request
from konlpy.tag import Okt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

print(train_data)
print(test_data)

train_data.drop_duplicates(subset = ['document'], inplace = True)
train_data.groupby('label').size()
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 ]", "")
train_data
train_data['document'] = train_data['document'].str.replace('^ +', "")
train_data['document'].replace('', np.nan, inplace = True)
print(train_data.isnull().sum())
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

okt = Okt()
X_train = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)
print(X_train)

X_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(str(sentence), stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test.append(stopwords_removed_sentence)
print(X_test)

def tokenize(x_train, y_train, x_val, y_val):
  word_list = []
  for sent in x_train:
    for word in sent:
      word_list.append(word)
  corpus = Counter(word_list)
  corpus_ = sorted(corpus, key = corpus.get, reverse = True)[:10000]
  onehot_dict = {w:i+1 for i, w in enumerate(corpus_)}
  print(onehot_dict)
  final_list_train, final_list_test = [], []
  for sent in x_train:
    final_list_train.append([onehot_dict[word] for word in sent if word in onehot_dict.keys()])
  for sent in x_val:
    final_list_test.append([onehot_dict[word] for word in sent if word in onehot_dict.keys()])
  return np.array(final_list_train), np.array(y_train), np.array(final_list_test), np.array(y_val), onehot_dict

x_train, y_train, x_test, y_test, vocab = tokenize(X_train, train_data['label'], X_test, test_data['label'])
print(len(vocab))

def padding(sentences, seq_len):
  print(sentences[0])

  features = np.zeros((len(sentences), seq_len), dtype = int)
  for ii, review in enumerate(sentences):
    if len(review) != 0:
      # print(ii, review)
      features[ii, -len(review):] = np.array(review)[:seq_len]
  return features
x_train_pad = padding(x_train, 50)
x_test_pad = padding(x_test, 50)
# print(x_train[0])
# print(x_test[0])
print(x_train_pad[0])
print(x_test_pad[0])

train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

batch_size = 50
train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size)
test_loader = DataLoader(test_data, shuffle = True, batch_size = batch_size)

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

device = torch.device("cuda")
print(device)

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
    corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum().item()
  size = len(data_loader.dataset)
  avg_accuracy = 100.0*corrects / size
  return avg_accuracy


num_epochs = 10
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

for e in range(num_epochs):
  train_loss = train(model, criterion, optimizer, train_loader)
  test_accuracy = evaluate(model, test_loader)
  print(train_loss, test_accuracy)
