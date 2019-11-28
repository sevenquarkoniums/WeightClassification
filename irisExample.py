import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np

trainingSeed = 0
shuffleLabel = 1 # shuffle the training set label to make random weights.
save = 1
showFig = 0
print('training seed: %d' % trainingSeed)
torch.manual_seed(trainingSeed)
np.random.seed(0) # fix the train_test_split output.

iteration = 200*1000

class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        num1, num2 = 8, 8
        self.fc1 = nn.Linear(4, num1)
        self.fc2 = nn.Linear(num1, num2)
        self.fc3 = nn.Linear(num2, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X
    
# load IRIS dataset
dataset = pd.read_csv('data/iris.csv')

# transform species to numerics
dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2


train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                    dataset.species.values, test_size=0.3333)

train_X = torch.Tensor(train_X)
test_X = torch.Tensor(test_X)
train_y = torch.Tensor(train_y).long()
test_y = torch.Tensor(test_y).long()
testsize = train_y.shape[0]

criterion = nn.CrossEntropyLoss()# cross entropy loss

now = datetime.datetime.now()
accuracy, weight = [], []
for iteration in range(iteration):
    if shuffleLabel:
        train_y = train_y[torch.randperm(testsize)]
    if iteration % 1000 == 0:
        print('iter %d' % iteration)
    net = Net()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.97)
    if showFig:
        loss_train = []
    for epoch in range(100):
        optimizer.zero_grad()
        out = net(train_X)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if showFig:
            loss_train.append(loss)
    net.eval()
    predict_out = net(test_X)
    _, predict_y = torch.max(predict_out, 1)
    accuracy.append(accuracy_score(test_y.data, predict_y.data))
    weightlist = [net.state_dict()['fc1.weight'].view(1, -1).squeeze(),
                  net.state_dict()['fc1.bias'],
                  net.state_dict()['fc2.weight'].view(1, -1).squeeze(),
                  net.state_dict()['fc2.bias'],
                  net.state_dict()['fc3.weight'].view(1, -1).squeeze(),
                  net.state_dict()['fc3.bias']
                  ]
    catweight = torch.cat(weightlist, dim=0)
    weight.append(catweight)

if showFig:
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(loss_train, '-', color='forestgreen')
    plt.tight_layout()

print('avg accuracy: %.3f +- %.3f' % (sum(accuracy)/len(accuracy), np.std(accuracy)))
allAcc = torch.FloatTensor(accuracy)
allWeight = torch.stack(weight, dim=0)
if shuffleLabel:
    name1 = 'data/irisShuffleAcc_%d.pt' % trainingSeed
    name2 = 'data/irisShuffleWeight_%d.pt' % trainingSeed
else:
    name1 = 'data/irisAcc_%d.pt' % trainingSeed
    name2 = 'data/irisWeight_%d.pt' % trainingSeed
if save:
    torch.save(allAcc, name1)
    torch.save(allWeight, name2)

print('finished in %d seconds.' % (datetime.datetime.now()-now).seconds)
