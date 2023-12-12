train_data = 'you need to know how to code'
word_set = set(train_data.split()) # 중복을 제거한 단어들의 집합인 단어 집합 생성.
vocab = {word: i+2 for i, word in enumerate(word_set)}  # 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab['<unk>'] = 0
vocab['<pad>'] = 1
print(vocab)

sample = 'you need to run'.split()
idxes = []
for word in sample:
  try:
    idxes.append(vocab[word])
  except KeyError:
    idxes.append(vocab['<unk>'])
idxes = torch.LongTensor(idxes)
print(idxes)


import torch.nn as nn
embedding_layer = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 3, padding_idx = 1)
lookup_result = embedding_layer(idxes)
print(lookup_result)

