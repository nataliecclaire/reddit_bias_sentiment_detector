from torch.nn import Module
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn.init import xavier_uniform_
import torch
import pandas as pd
from torch.autograd import Variable
import torch.utils.data as Data1
from math import sqrt


class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 362)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(362, 64)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        self.hidden3 = Linear(64, 1)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        return X


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
exp_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Experiments/execution_logs/'


demo = 'race' # 'religion1' # 'religion2' # 'gender' # 'race' #'orientation'  #
df_features = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_features.csv')
df_labels = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_diff.csv')
# print(df_labels.head())
df_labels = df_labels['diff_perplex']

data = df_features.join(df_labels)
print(data.shape)

data = data[(data['diff_perplex'] < 300) & (data['diff_perplex'] > -300)]
print('After filtering diff perplexity {}'.format(data.shape))

X = data.drop('diff_perplex', axis=1)

# X = X.iloc[:, [768, 1537]]
print(X.head())
y = data['diff_perplex']

# x = torch.unsqueeze(X, dim=1)

X_tensor = torch.tensor(X.values)
y_tensor = torch.tensor(y.values)

print(X_tensor)
print(type(X_tensor))
print(X_tensor.shape)
print(y_tensor)

x, y = Variable(X_tensor), Variable(y_tensor)

net = torch.nn.Sequential(
        torch.nn.Linear(1538, 800),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(800, 400),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(400, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 1),
)

# model = MLP(1538)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

BATCH_SIZE = 64
EPOCH = 5

torch_dataset = Data1.TensorDataset(x, y)

loader = Data1.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2,)

net = net.float()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step

        b_x = Variable(batch_x).float()
        b_y = Variable(batch_y).float()

        prediction = net(b_x)  # input x and predict based on x

        loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)
        print(sqrt(loss))
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()


