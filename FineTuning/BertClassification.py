import torch
from nlp import load_dataset

train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])


print(train_dataset)
print(test_dataset)

params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

training_generator = torch.utils.data.DataLoader(train_dataset, **params)

for batch, label in training_generator:
    print(batch['text'])
    print(label)