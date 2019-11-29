import torch
import matplotlib.pyplot as plt
shuffle = 0
acc, weight = torch.load('data/irisAcc_0.pt'), torch.load('data/irisWeight_0.pt')
accShuffle, weightShuffle = torch.load('data/irisShuffleAcc_0.pt'), torch.load('data/irisShuffleWeight_0.pt')
idxStart = [0,32,40,104,112,136]
idxEnd = [32,40,104,112,136,139]
position =['weight1','bias1','weight2','bias2','weight3','bias3']
for i in range(6):
    if shuffle:
        n, bins, patches = plt.hist(weightShuffle[:,idxStart[i]:idxEnd[i]].reshape(-1), 300, facecolor='forestgreen')
        plt.savefig('figure/irisShuffle_%s.png' % (position[i]))
    else:
        n, bins, patches = plt.hist(weight[:,idxStart[i]:idxEnd[i]].reshape(-1), 300, facecolor='forestgreen')
        plt.savefig('figure/iris_%s.png' % (position[i]))
    plt.tight_layout()
    plt.close()


