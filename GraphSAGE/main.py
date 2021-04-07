import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from net import GraphSage
from utils import load_prepared_data
from sampling import multihop_sampling
from collections import namedtuple
from pathlib import Path
import os
import time

#torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='europe-airports')

args = parser.parse_args()
# print(adj.shape)
# print(features.shape)
# print(labels.shape)


BTACH_SIZE = 128  # 批处理大小
epochs = args.epochs
NUM_BATCH_PER_EPOCH = 20  # 每个epoch循环的批次数
learning_rate = args.lr  # 学习率
device = "cuda" if torch.cuda.is_available() else "cpu"
if device=='cuda':
    torch.cuda.set_device(1)
dataset = args.dataset
if dataset in {'pubmed','Flickr','BlogCatalog'}:
    device='cpu'
# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)
adj, features, labels, neighbor_table = load_prepared_data(dataset)
input_dim = features.shape[1]  # 输入维度
N = features.shape[0]
hidden_dim = [128, 32]  # 隐藏单元节点数
idx_train = range(N // 4)
idx_val = range(N // 3, (N // 3) * 2)
idx_test = idx_val
# 避免过多占用内存
num_neighbors_list = [5, 5, 5]  # 每阶采样邻居的节点数
# Note: 采样的邻居阶数需要与GCN的层数保持一致
hidden_dim.append(max(labels) + 1)
assert len(hidden_dim) == len(num_neighbors_list)
model = GraphSage(input_dim=input_dim, hidden_dim=hidden_dim,
                  num_neighbors_list=num_neighbors_list).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
features = features.to(device)
labels = labels.to(device)


def train_batch():
    model.train()
    for e in range(epochs):
        loss = 0
        for batch in range(NUM_BATCH_PER_EPOCH):
            # batch_src_index = np.random.choice(idx_train, size=(BTACH_SIZE,))
            # batch_src_label = torch.from_numpy(labels[batch_src_index]).long().to(DEVICE)
            # batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, neighbor_table)
            # batch_sampling_x = [torch.from_numpy(features[idx]).float().to(DEVICE) for idx in batch_sampling_result]
            # 在这里全部转化为tensor进行处理
            batch_src_index = torch.from_numpy(np.random.choice(idx_train, size=(BTACH_SIZE,))).long()
            batch_src_label = labels[batch_src_index].to(device)
            batch_sampling_result = multihop_sampling(batch_src_index, num_neighbors_list, neighbor_table)
            batch_sampling_x = [features[idx].to(device) for idx in batch_sampling_result]
            batch_train_logits = model(batch_sampling_x)
            batch_loss = criterion(batch_train_logits, batch_src_label)
            optimizer.zero_grad()
            batch_loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            loss += batch_loss.item() * BTACH_SIZE
        print("Epoch {:03d} Loss: {:.4f}".format(e, loss / (NUM_BATCH_PER_EPOCH * BTACH_SIZE)))
        test()


def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multihop_sampling(torch.LongTensor(idx_test), num_neighbors_list, neighbor_table)
        test_x = []
        for idx in test_sampling_result:
            test_x.append(features[idx])
        #  test_x = [features[idx] for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = labels[idx_test].long()
        predict_y = test_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
    return accuarcy


def save_embeddings():
    model.eval()
    with torch.no_grad():
        src_index = torch.LongTensor(range(features.shape[0]))
        sampling_result = multihop_sampling(src_index, num_neighbors_list, neighbor_table)
        # print('{}:sampling_x前:{}'.format(epoch, torch.cuda.memory_allocated()))
        sampling_x = []
        for idx in sampling_result:
            # 这种方法会共享内存
            sampling_x.append(features[idx])
        train_logits = model.saveVec(sampling_x)
    outVec = train_logits.cpu().detach().numpy()
    path = Path(__file__).parent / ('{}_outVec.txt'.format(dataset))
    np.savetxt(path, outVec)


if __name__ == '__main__':
    model.train()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    for epoch in range(epochs):
        model.train()
        time1=time.time()
        src_index = torch.LongTensor(range(features.shape[0]))
        sampling_result = multihop_sampling(src_index, num_neighbors_list, neighbor_table)
        # print('{}:sampling_x前:{}'.format(epoch, torch.cuda.memory_allocated()))
        sampling_x = []
        for idx in sampling_result:
            # 这种方法会共享内存
            sampling_x.append(features[idx])
        train_logits = model(sampling_x)
        sampling_x = []  # 释放内存
        loss = criterion(train_logits[idx_train], labels[idx_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = test()
        time2=time.time()
        print("Epoch {:03d} Loss: {:.4f}, Accuracy: {:.4f}, time:{:.1f}".format(epoch, loss.item(),
                                                                   accuracy,time2-time1))
    save_embeddings()
