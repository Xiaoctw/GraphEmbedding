from __future__ import division
from __future__ import print_function

import time
import argparse  # argparse 是python自带的命令行参数解析包，可以用来方便地读取命令行参数
import numpy as np
import torch
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset',type=str,default='cora')

# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# adj, features, labels,idx_train, idx_val, idx_test=load_data()
# print(adj.shape)
# print(features.shape)
# print(labels.shape)
# Load data
from utils import *
dataset=args.dataset


adj_mat, features, labels =load_prepared_data(dataset)
N=adj_mat.shape[0]
# print(adj.shape)
# print(features.shape)
# print(labels.shape)
idx_train = range(N//5)
idx_val = range(N//3,(N//3)*2)
idx_test=idx_val
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# 数据写入cuda，便于后续加速
if args.cuda:
    model.cuda()  # . cuda()会分配到显存里（如果gpu可用）
    features = features.cuda()
    adj_mat = adj_mat.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()  # 返回当前时间
    model.train()
    optimizer.zero_grad()
    # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
    # pytorch中每一轮batch需要设置optimizer.zero_gra
    output = model(features, adj_mat)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # 由于在算output时已经使用了log_softmax，这里使用的损失函数就是NLLloss，如果前面没有加log运算，
    # 这里就要使用CrossEntropyLoss了
    # 损失函数NLLLoss() 的输入是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率，
    # 适合最后一层是log_softmax()的网络. 损失函数 CrossEntropyLoss() 与 NLLLoss() 类似,
    # 唯一的不同是它为我们去做 softmax.可以理解为：CrossEntropyLoss()=log_softmax() + NLLLoss()
    # https://blog.csdn.net/hao5335156/article/details/80607732
    acc_train = accuracy(output[idx_train], labels[idx_train])  # 计算准确率
    loss_train.backward()  # 反向求导  Back Propagation
    optimizer.step()  # 更新所有的参数  Gradient Descent
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()  # eval() 函数用来执行一个字符串表达式，并返回表达式的值
        output = model(features, adj_mat)
    # print(output[idx_val].shape)
    # print(labels[idx_val].shape)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])  # 验证集的损失函数
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


# 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
def test():
    model.eval()
    output = model(features, adj_mat)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def save_embeddings(dataset='cora'):
    model.eval()
    output = model.savector(features, adj_mat)
    outVec = output.cpu().detach().numpy()
    path = Path(__file__).parent / 'GCN_result'/('{}_outVec.txt'.format(dataset))
    #这里第一个位置为地址，第二个位置为内容
    np.savetxt(path, outVec)
    path = Path(__file__).parent / 'GCN_result'/('{}_labels.txt'.format(dataset))
    # outLabel = labels.cpu().detach().numpy()
    # np.savetxt(path, outLabel)


if __name__ == '__main__':
    # Train model  逐个epoch进行train，最后test
    # 观察是否修改
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('Start testing...')
    test()
    print('End of the test')
    print('Retain embedding results.')
    save_embeddings(args.dataset)
