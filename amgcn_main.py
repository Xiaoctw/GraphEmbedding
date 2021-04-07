import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pathlib import Path
from models import GAT
import scipy.sparse as sp
from utils import *
import warnings
from models import *

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=3e-4,
                    help='The learning rate')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--epochs', type=list, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=400)
parser.add_argument('--embedding_dim', type=int, default=32)
# num_feat, num_hidden, n_class
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=5e-2)
parser.add_argument('--sigma', type=float, default=1)

args = parser.parse_args()
adj_mat, feat_mat, labels = load_prepared_data(args.dataset)

N = feat_mat.shape[0]
square_x = torch.pow(feat_mat, 2)
sum_square_x = torch.sum(square_x, 1)
sum_mat = sum_square_x.view(N, 1) + sum_square_x.view(1, N)
feat_adj_mat = sum_mat - 2 * torch.mm(feat_mat, feat_mat.t())
feat_adj_mat = normalize_np(feat_adj_mat.numpy())
feat_adj_mat=torch.FloatTensor(feat_adj_mat)
N = adj_mat.shape[0]
num_feat = feat_mat.shape[1]
# ppmi_mat = PPMI_matrix(surfing_mat)
# ppmi_mat = torch.from_numpy(ppmi_mat).float()
GPU = args.cuda and torch.cuda.is_available()
# if args.dataset in {'pubmed'}:
#     GPU=False
if GPU:
    torch.cuda.set_device(1)
lr = args.lr
epochs = args.epochs


def train_AMGCN(model: nn.Module, adj1, adj2, Y, idx_train, idx_val, epochs, lr, print_every,
                GPU=False,
                feat_X=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, )
    optimizer.zero_grad()
    if feat_X is None:
        X = torch.FloatTensor(np.diag(np.ones(Y.shape[0])))
    else:
        X = feat_X
    if GPU:
        model = model.cuda()
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()
        X = X.cuda()
        Y = Y.cuda()
    pre_time = time.time()
    for epoch in range(epochs):
        # output = model(X, adj1, adj2)
        lt, lc, ld, output = model.compute_loss(X, adj1, adj2, Y, idx_train)
        loss = lt + lc + ld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % print_every == 0 or epoch == 0:
            model.eval()
            tem_time = time.time()
            print(' epoch:{}, lt:{:.4f}, lc:{:.4f}, ld:{:.4f}, time:{:.4f}, accuracy:{}'.format(epoch + 1, lt, lc, ld,
                                                                                                (
                                                                                                            tem_time - pre_time) / print_every,
                                                                                                accuracy(
                                                                                                    output[idx_val],
                                                                                                    Y[idx_val])))
            pre_time = tem_time
            model.train()
    return model


model = AMGCN(num_feat=num_feat, num_hidden=args.hidden_dim, alpha=args.alpha, beta=args.beta, sigma=args.sigma,
              embedding_dim=args.embedding_dim, n_class=labels.max().item() + 1)

if __name__ == '__main__':
    print('节点数量:{}'.format(N))
    print('特征个数:{}'.format(num_feat))
    print(adj_mat.shape)
    idx_train = range(N // 4)
    idx_val = range(N // 2, )
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    train_AMGCN(model, adj1=adj_mat, adj2=feat_adj_mat, Y=labels, idx_train=idx_train, idx_val=idx_val,
                epochs=epochs,
                lr=lr, print_every=args.print_every, GPU=GPU, feat_X=feat_mat)
    model = model.cpu()
    _, _, _, _, w, embs = model.emb(feat_mat, adj_mat, feat_adj_mat, require_weight=True)
    embs = embs.cpu().detach().numpy()
    save_path = Path(__file__).parent / 'AMGCN_result' / (args.dataset + '_outVec.txt')
    np.savetxt(save_path, embs)
    print(w[:20, :])
