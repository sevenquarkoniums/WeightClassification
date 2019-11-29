from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn import datasets as sklearndatasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import scale,LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.patches as mpatches
from torch.utils.data.sampler import SubsetRandomSampler

'''
Warning:
    1 num_workers>0 is super slow on windows.
'''

mode = 'weight'
netType = 'gcn' # linear, 1hidden, 2hidden, 3hidden, gnn, gcn.
numEpoch = 10
middleOutput = 1
save_model = 1
torch.manual_seed(0)
fontsize = 15
np.random.seed(0) # fix the train_test_split output.

class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, 3, 1)
#        self.conv2 = nn.Conv2d(32, 64, 3, 1)
#        self.dropout1 = nn.Dropout2d(0.25)
#        self.dropout2 = nn.Dropout2d(0.5)
#        self.fc1 = nn.Linear(9216, 128)
#        self.fc2 = nn.Linear(128, 10)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 2, 3, 1)
        self.fc1 = nn.Linear(72, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
#        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
#        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class irisNet(nn.Module):
    def __init__(self):
        super(irisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.softmax(x, dim=1)
            # test_loss could keep increase if using log_softmax.
        return output

class weightNet(nn.Module):
    def __init__(self):
        super(weightNet, self).__init__()
        if netType == 'linear':
            self.fc1 = nn.Linear(139, 1)
        elif netType == '1hidden':
            num1 = 1000
            self.fc1 = nn.Linear(139, num1)
            self.fc2 = nn.Linear(num1, 1)
        elif netType == '2hidden':
            num1, num2 = 1000, 100
            self.fc1 = nn.Linear(139, num1)
            self.fc2 = nn.Linear(num1, num2)
            self.fc3 = nn.Linear(num2, 1)
        elif netType == '3hidden':
            num1, num2, num3 = 1000, 300, 100
            self.fc1 = nn.Linear(139, num1)
            self.fc2 = nn.Linear(num1, num2)
            self.fc3 = nn.Linear(num2, num3)
            self.fc4 = nn.Linear(num3, 1)
        elif netType == 'gnn':
            # No weights are shared. very slow. need update.
            self.channel = 10
            self.link = nn.ModuleDict()
            for c in range(self.channel):
                for i in range(8):
                    self.link['(1,%d,0,%d)' % (c,i)] = nn.Linear(13, 1)
                for i in range(8):
                    self.link['(1,%d,1,%d)' % (c,i)] = nn.Linear(12, 1)
                for i in range(3):
                    self.link['(1,%d,2,%d)' % (c,i)] = nn.Linear(4, 1)
            self.fc1 = nn.Linear(190, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 1)
        elif netType == 'gcn':
            # Sharing weights.
            self.channel = 16
            self.link = nn.ModuleDict()
            self.link['(1,0)'] = nn.Linear(21, self.channel)
            self.link['(1,1)'] = nn.Linear(23, self.channel)
            self.link['(1,2)'] = nn.Linear(12, self.channel)
            self.fc1 = nn.Linear(19*self.channel, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 1)
#            self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        if netType == 'linear':
            x = torch.sigmoid(self.fc1(x))
        elif netType == '1hidden':
            x = F.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
        elif netType == '2hidden':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
        elif netType == '3hidden':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
        elif netType == 'gnn':
            # mapping is wrong.
            layer1 = []
            for c in range(self.channel):
                for i in range(8):
                    idx = [4*i+0,4*i+1,4*i+2,4*i+3,i+32,8*i+40,8*i+41,8*i+42,8*i+43,8*i+44,8*i+45,8*i+46,8*i+47]
                    layer1.append(self.link['(1,%d,0,%d)' % (c,i)](x[:,idx]))
                for i in range(8):
                    idx = [8*i+40,8*i+41,8*i+42,8*i+43,8*i+44,8*i+45,8*i+46,8*i+47,i+104,3*i+112,3*i+113,3*i+114]
                    layer1.append(self.link['(1,%d,1,%d)' % (c,i)](x[:,idx]))
                for i in range(3):
                    idx = [3*i+112,3*i+113,3*i+114,i+136]
                    layer1.append(self.link['(1,%d,2,%d)' % (c,i)](x[:,idx]))
            x = F.relu(torch.cat(layer1, dim=1))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
        elif netType == 'gcn':
            layer1 = []
            for i in range(8):
                idx = [4*i+0,4*i+1,4*i+2,4*i+3,i+32,i+40,i+48,i+56,i+64,i+72,i+80,i+88,i+96,104,105,106,107,108,109,110,111]
                layer1.append(self.link['(1,0)'](x[:,idx]))
            for i in range(8):
                idx = [32,33,34,35,36,37,38,39,8*i+40,8*i+41,8*i+42,8*i+43,8*i+44,8*i+45,8*i+46,8*i+47,i+104,i+112,i+120,i+128,136,137,138]
                layer1.append(self.link['(1,1)'](x[:,idx]))
            for i in range(3):
                idx = [104,105,106,107,108,109,110,111,3*i+112,3*i+113,3*i+114,i+136]
                layer1.append(self.link['(1,2)'](x[:,idx]))
            x = F.relu(torch.cat(layer1, dim=1))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
#            x = self.leakyrelu(torch.cat(layer1, dim=1))
#            x = self.leakyrelu(self.fc1(x))
#            x = self.leakyrelu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
        return x

class irisCreator(Dataset):
    def __init__(self, x, y):
        self.X = x.astype(np.float32)
#        self.Y = y
        self.mlb = LabelBinarizer()
        self.Y = self.mlb.fit_transform(y)
    def __getitem__(self, index):
        feature = torch.from_numpy(self.X[index])
        label = torch.LongTensor(self.Y[index])
        return feature, label
    def __len__(self):
        return len(self.X)

class weightCreator(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y
    def __getitem__(self, index):
        feature = self.X[index, :]
        label = self.Y[index, :]
        return feature, label
    def __len__(self):
        return self.X.shape[0]

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    if mode in ['iris','mnist']:
        length = len(train_loader.dataset)
    elif mode == 'weight':
        length = 0
    lossSum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if mode == 'iris':
            label = torch.max(target, 1)[1]
            loss = criterion(output, label)
        elif mode == 'mnist':
            loss = F.nll_loss(output, target, reduction='sum')
        elif mode == 'weight':
            loss = criterion(output, target)
            length += data.shape[0]
        lossSum += loss.item()
        loss.backward()
        optimizer.step()
        if middleOutput and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}]'.format(
                epoch, batch_idx * len(data)))
    lossSum /= length
    return lossSum


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    aggregate = 0
    if mode in ['iris','mnist']:
        length = len(test_loader.dataset)
    elif mode == 'weight':
        length = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if mode == 'iris':
                label = torch.max(target, 1)[1]
                test_loss += criterion(output, label).item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                aggregate += pred.eq(label.view_as(pred)).sum().item()
            elif mode == 'mnist':
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                aggregate += pred.eq(target.view_as(pred)).sum().item()
            elif mode == 'weight':
                test_loss += criterion(output, target).item()
                aggregate += torch.abs(output - target).sum()
                viewtest = torch.cat([output,target], dim=1)[:10, :]
                length += data.shape[0]
    test_loss /= length
    accuracy = aggregate / length
    if middleOutput:
        if mode in ['iris','mnist']:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, aggregate, length, 100. * accuracy))
        elif mode == 'weight':
            print('Test set: Average loss: {:.4f}, Error percentage: {:.1f}%\n'.format(
                test_loss, 100. * accuracy))
            print(viewtest)
    return test_loss, accuracy

def draw(loss_train, loss_test, acc):
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    colors = ['b','forestgreen','r']
    ax.plot(loss_train, '-', color=colors[0])
    ax.plot(loss_test, '-', color=colors[1])
    ax.plot(acc, '-', color=colors[2])
    patches, labels = [], ['train loss', 'test loss']
    if mode in ['iris','mnist']:
        ax.hlines(1, ax.get_xlim()[0], ax.get_xlim()[1], colors='k', linestyles='dashed')
        labels.append('accuracy')
    elif mode == 'weight':
        labels.append('avg error')
    for i in range(len(colors)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    ax.legend(handles=patches, labels=labels, loc='upper right', ncol=1, fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('figure/%s.png' % mode)

def build():
    # Training settings
    parser = argparse.ArgumentParser(description='Example')
#    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                        help='input batch size for training (default: 64)')
#    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                        help='input batch size for testing (default: 1000)')
#    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
#                        help='number of epochs to train (default: 14)')
#    parser.add_argument('--lr', type=float, default=1, metavar='LR',
#                        help='learning rate (default: 1.0)')
#    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
#                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
#    parser.add_argument('--seed', type=int, default=0, metavar='S',
#                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

#    parser.add_argument('--save-model', action='store_true', default=True,
#                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    # loading data.
    print('loading data..')
    if mode == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=64, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=1000, shuffle=True, **kwargs)
    elif mode == 'iris':
        iris = sklearndatasets.load_iris()
        scaled = scale(iris.data)
        X_train, X_test, Y_train, Y_test = train_test_split(scaled, iris.target, test_size=0.3333, random_state=0)
        trainset = irisCreator(X_train, Y_train)
        testset = irisCreator(X_test, Y_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=True, **kwargs)
    elif mode == 'weight':
        test_size = 0.2
        acc, weight = torch.load('data/irisAcc_0.pt'), torch.load('data/irisWeight_0.pt')
        accShuffle, weightShuffle = torch.load('data/irisShuffleAcc_0.pt'), torch.load('data/irisShuffleWeight_0.pt')
        accAll = torch.cat([acc, accShuffle], dim=0).unsqueeze(1)
        weightAll = torch.cat([weight, weightShuffle], dim=0)
        dataset = weightCreator(weightAll, accAll)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_size * dataset_size))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, sampler=test_sampler, **kwargs)

    print('finish loading.')

    if mode == 'mnist':
        model = mnistNet().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        criterion = None
    elif mode == 'iris':
        model = irisNet().to(device)
#        optimizer = optim.Adadelta(model.parameters(), lr=1)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.97)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    elif mode == 'weight':
        model = weightNet().to(device)
#        optimizer = optim.Adadelta(model.parameters(), lr=1)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        criterion = torch.nn.MSELoss(reduction='sum')
    loss_train, loss_test, acc = [], [], []
    for epoch in range(numEpoch):
        if epoch % 100 == 0:
            print('epoch %d' % epoch)
        thisTrain = train(args, model, device, train_loader, optimizer, epoch, criterion)
        loss_train.append(thisTrain)
        thisTest = test(args, model, device, test_loader, criterion)
        loss_test.append(thisTest[0])
        acc.append(thisTest[1])
#        scheduler.step() # no need scheduler if using adam.
#    print(model.state_dict())
    if save_model:
        torch.save(model.state_dict(), "%s.pt" % mode)

    draw(loss_train, loss_test, acc)
    
def randomInput():
    device = torch.device("cuda")
    model = weightNet().to(device)
    model.load_state_dict(torch.load('weight.pt'))
    model.eval()
    data = torch.Tensor(np.random.normal(loc=0.0, scale=0.5, size=(100000, 139))).to(device)
    output = model(data)
    outputnp = output.squeeze().cpu().detach().numpy()
    n, bins, patches = plt.hist(outputnp, 300, facecolor='deepskyblue')
    plt.savefig('figure/randomInput.png')
    plt.tight_layout()
    
def main():
    now = datetime.datetime.now()
    
    build()
#    randomInput()
    
    print('finished in %d seconds.' % (datetime.datetime.now()-now).seconds)

if __name__ == '__main__':
    main()
