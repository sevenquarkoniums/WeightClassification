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
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.preprocessing import scale,LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
import datetime

mode = 'mnist'

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
        output = F.log_softmax(x, dim=1)
        return output

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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    length = len(train_loader.dataset)
#    length = len(train_loader[0][0])
    if mode == 'iris':
        criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if mode == 'iris':
            label = torch.max(target, 1)[1]
            loss = criterion(output, label)
        elif mode == 'mnist':
            loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), length,
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    length = len(test_loader.dataset)
#    length = len(test_loader[0][0])
    if mode == 'iris':
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if mode == 'iris':
                label = torch.max(target, 1)[1]
                test_loss += criterion(output, label)
            elif mode == 'mnist':
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            if mode == 'iris':
                correct += pred.eq(label.view_as(pred)).sum().item()
            elif mode == 'mnist':
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= length
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, length,
        100. * correct / length))
    return test_loss

def draw(loss_train, loss_test):
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(loss_train, '-b')
    ax.plot(loss_test, '-', color='forestgreen')
    plt.tight_layout()

def main():
    now = datetime.datetime.now()
    # Training settings
    parser = argparse.ArgumentParser(description='Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # loading data.
    if mode == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif mode == 'iris':
        print('loading iris..')
        iris = sklearndatasets.load_iris()
        scaled = scale(iris.data)
        X_train, X_test, Y_train, Y_test = train_test_split(scaled, iris.target, test_size=0.20, random_state=0)
        trainset = irisCreator(X_train, Y_train)
        testset = irisCreator(X_test, Y_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=30, shuffle=False, **kwargs)

#        X_train = Variable(torch.Tensor(X_train).float())
#        X_test = Variable(torch.Tensor(X_test).float())
#        Y_train = Variable(torch.Tensor(Y_train).long())
#        Y_test = Variable(torch.Tensor(Y_test).long())
#        train_loader = [(X_train, Y_train)]
#        test_loader = [(X_test, Y_test)]
        print('finish loading.')

    torch.manual_seed(0)
    if mode == 'mnist':
        model = mnistNet().to(device)
    elif mode == 'iris':
        model = irisNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_train, loss_test = [], []
    for epoch in range(1, args.epochs + 1):
        loss_train.append(train(args, model, device, train_loader, optimizer, epoch))
        loss_test.append(test(args, model, device, test_loader))
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "%s.pt" % mode)

    draw(loss_train, loss_test)
    print('finished in %d seconds.' % (datetime.datetime.now()-now).seconds)

if __name__ == '__main__':
    main()
