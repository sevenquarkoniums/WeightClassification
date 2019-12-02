import torch
import matplotlib.pyplot as plt

mode = 'mnist'

def combine():
#    accShuffle, weightShuffle = [], []
#    for i in range(200):
#        accShuffle.append(torch.load('data/mnistShuffleAcc_%d.pt' % i))
#        weightShuffle.append(torch.load('data/mnistShuffleWeight_%d.pt' % i))
#    accShuffle = torch.cat(accShuffle, dim=0)
#    weightShuffle = torch.cat(weightShuffle, dim=0)
#    torch.save(accShuffle, 'data/mnistShuffleAcc.pt')
#    torch.save(weightShuffle, 'data/mnistShuffleWeight.pt')
    
    acc, weight = [], []
    for i in range(200):
        acc.append(torch.load('data/mnistAcc_%d.pt' % i))
        weight.append(torch.load('data/mnistWeight_%d.pt' % i))
    acc = torch.cat(acc, dim=0)
    weight = torch.cat(weight, dim=0)
    torch.save(acc, 'data/mnistAcc.pt')
    torch.save(weight, 'data/mnistWeight.pt')

def drawWeight():
    shuffle = 1
    if mode == 'iris':
        acc, weight = torch.load('data/irisAcc_0.pt'), torch.load('data/irisWeight_0.pt')
        accShuffle, weightShuffle = torch.load('data/irisShuffleAcc_0.pt'), torch.load('data/irisShuffleWeight_0.pt')
        idxStart = [0,32,40,104,112,136]
        idxEnd = [32,40,104,112,136,139]
        position =['weight1','bias1','weight2','bias2','weight3','bias3']
    elif mode == 'mnist':
        acc, weight = torch.load('data/mnistAcc.pt'), torch.load('data/mnistWeight.pt')
        accShuffle, weightShuffle = torch.load('data/mnistShuffleAcc.pt'), torch.load('data/mnistShuffleWeight.pt')
        idxStart = [0,144,160,736,740,3044,3060,3220]
        idxEnd = [144,160,736,740,3044,3060,3220,3230]
        position =['weight1','bias1','weight2','bias2','weight3','bias3','weight4','bias4']
    for i in range(len(position)):
        if shuffle:
            n, bins, patches = plt.hist(weightShuffle[:,idxStart[i]:idxEnd[i]].reshape(-1), 300, facecolor='forestgreen')
            plt.savefig('figure/%sShuffle_%s.png' % (mode, position[i]))
        else:
            n, bins, patches = plt.hist(weight[:,idxStart[i]:idxEnd[i]].reshape(-1), 300, facecolor='forestgreen')
            plt.savefig('figure/%s_%s.png' % (mode, position[i]))
        plt.tight_layout()
        plt.close()

def drawAcc():
    if mode == 'iris':
        acc, weight = torch.load('data/irisAcc_0.pt'), torch.load('data/irisWeight_0.pt')
        accShuffle, weightShuffle = torch.load('data/irisShuffleAcc_0.pt'), torch.load('data/irisShuffleWeight_0.pt')
    elif mode == 'mnist':
        acc = torch.load('data/mnistAcc.pt')
        accShuffle = torch.load('data/mnistShuffleAcc.pt')
    n, bins, patches = plt.hist(acc, 300, facecolor='orange')
    plt.savefig('figure/%sAcc.png' % mode)
    plt.close()
    n, bins, patches = plt.hist(accShuffle, 300, facecolor='orange')
    plt.savefig('figure/%sAccShuffle.png' % mode)
    plt.close()
    
#combine()
drawWeight()
#drawAcc()
