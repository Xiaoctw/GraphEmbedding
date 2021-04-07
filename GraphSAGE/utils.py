from pathlib import Path
import scipy.sparse as sp
import numpy as np
import torch
import itertools


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def load_prepared_data(dataset='cora'):
    if dataset in {'BlogCatalog', 'citeseer', 'cora', 'Flickr', 'pubmed'}:
        path = Path(__file__).parent.parent / 'data'
        print('Loading {} dataset...'.format(dataset))
        labels = np.load(path / '{}_labels.npy'.format(dataset))
        features = sp.load_npz(path / '{}_features.npz'.format(dataset))
        adj = sp.load_npz(path / '{}_adj.npz'.format(dataset))
        from collections import defaultdict
        neighbor_table = defaultdict(lambda: [])
        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
        adj = normalize(adj)
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj= torch.FloatTensor(adj.todense())
        i_s, j_s = torch.where(adj > 0)
        for i in range(len(i_s)):
            neighbor_table[i_s[i].item()].append(j_s[i].item())
        #  print(neighbor_table)
        # print(adj.shape)
        # print(i_s.shape)
        # print(j_s.shape)
        # for i in range(len(i_s)):
        #     if len(neighbor_table[i_s[i].item()]) == 0:
        #         print(i)
        # print(neighbor_table[500])
        print('特征维数:{}'.format(features.shape[1]))
        return adj, features, labels, neighbor_table
    else:
        path = Path(__file__).parent.parent / 'data'
        print('Loading {} dataset...'.format(dataset))
        labels = np.load(path / '{}_labels.npy'.format(dataset))
        adj = sp.load_npz(path / '{}_adj.npz'.format(dataset))
        from collections import defaultdict
        neighbor_table = defaultdict(lambda: [])
        adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
        adj = normalize(adj)
        labels = torch.LongTensor(labels)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj = torch.FloatTensor(adj.todense())
        i_s, j_s = torch.where(adj > 0)
        for i in range(len(i_s)):
            neighbor_table[i_s[i].item()].append(j_s[i].item())
        #  print(neighbor_table)
        # print(adj.shape)
        # print(i_s.shape)
        # print(j_s.shape)
        # for i in range(len(i_s)):
        #     if len(neighbor_table[i_s[i].item()]) == 0:
        #         print(i)
        # print(neighbor_table[500])
        features= torch.FloatTensor(np.diag(np.ones(labels.shape[0])))
        print('特征维数:{}'.format(features.shape[1]))
        return adj, features, labels, neighbor_table



if __name__ == '__main__':
    load_prepared_data()
