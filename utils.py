from pathlib import Path
import scipy.sparse as sp
import numpy as np
import torch
from typing import *


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def normalize_np(mx: np.ndarray) -> np.ndarray:
    # 对每一行进行归一化
    rows_sum = np.array(mx.sum(1)).astype('float')  # 对每一行求和
    rows_inv = np.power(rows_sum, -1).flatten()  # 求倒数
    rows_inv[np.isinf(rows_inv)] = 0  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    # rows_inv = np.sqrt(rows_inv)
    rows_mat_inv = np.diag(rows_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = rows_mat_inv.dot(mx)  # .dot(cols_mat_inv)
    return mx


def load_prepared_data(dataset='cora'):
    # sp.save_npz('{}_adj.npz'.format(dataset), adj)
    # sp.save_npz('{}_features.npz'.format(dataset), features)
    # # sp.save_npz('{}_labels.npz'.format(dataset),save_labels)
    # np.save('{}_labels.npy'.format(dataset), save_labels)
    print('Loading {} dataset'.format(dataset))
    path = Path(__file__).parent / 'data'
    labels = np.load(path / '{}_labels.npy'.format(dataset))
    adj = sp.load_npz(path / '{}_adj.npz'.format(dataset))
    adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
    # adj = normalize(adj)
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(adj.todense())
    print('finish loading')
    if dataset in {'BlogCatalog', 'citeseer', 'cora', 'Flickr', 'pubmed'}:
        features = sp.load_npz(path / '{}_features.npz'.format(dataset))
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = torch.FloatTensor(np.diag(np.ones(labels.shape[0])))
    return adj, features, labels


def load_ripple_data(dataset='cora'):
    print('Loading {} dataset'.format(dataset))
    path = Path(__file__).parent / 'data'
    labels = np.load(path / '{}_labels.npy'.format(dataset))
    ripple = sp.load_npz(path / '{}_ripple.npz'.format(dataset))
    ripple = normalize(ripple + sp.eye(ripple.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
    # adj = normalize(adj)
    labels = torch.LongTensor(labels)
    ripple = torch.FloatTensor(ripple.todense())
    print('finish loading')
    if dataset in {'BlogCatalog', 'citeseer', 'cora', 'Flickr', 'pubmed'}:
        features = sp.load_npz(path / '{}_features.npz'.format(dataset))
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = torch.FloatTensor(np.diag(np.ones(labels.shape[0])))
    return ripple, features, labels


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)


def sampling(src_nodes: torch.TensorType, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    在src_nodes中每个节点都采样，采样sample_num个
    Arguments:
        src_nodes {Tensor} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表
    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    for i in range(src_nodes.shape[0]):
        sid = src_nodes[i].item()
        # 从节点的邻居中进行有放回地进行采样
        # print(sid)
        # print(len(neighbor_table[500]))
        # if len(neighbor_table[sid])==0:
        #     print('sid:{}'.format(sid))
        #     print('neighbor:{}'.format(neighbor_table[sid]))
        res = np.random.choice(neighbor_table[sid], size=(sample_num,))
        results.append(res)
    return torch.from_numpy(np.asarray(results).flatten())


def multihop_sampling(src_nodes: torch.TensorType, sample_nums, neighbor_table) -> List[torch.TensorType]:
    """根据源节点进行多阶采样
    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射

    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result
