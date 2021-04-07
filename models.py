import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from pathlib import Path
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    # 初始化层：输入feature，输出feature，权重，偏移
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # FloatTensor建立tensor
        # 常见用法self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))：
        # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter
        # 绑定到这个module里面，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            # Parameters与register_parameter都会向parameters写入参数，但是后者可以支持字符串命名
        self.reset_parameters()

    # 初始化权重
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
       # self.weight.data.uniform_(-stdv, stdv)  # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        nn.init.xavier_uniform_(self.weight,gain=1)
        if self.bias is not None:
            #均匀分布初始化
            nn.init.uniform_(self.bias,-stdv,stdv)
           # self.bias.data.uniform_(-stdv, stdv)
          #这样就不行了，因为没有fall_in和fall_out
            #nn.init.xavier_uniform_(self.bias,gain=1)
    '''
    前馈运算 即计算A~ X W(0)
    input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
    直接输入与权重之间进行torch.mm操作，得到support，即XW
    support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias
    '''
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        output=torch.mm(adj, support)
        #output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
#通过设置断点，可以看出output的形式是0.01，0.01，0.01，0.01，0.01，#0.01，0.94]，里面的值代表该x对应标签不同的概率，故此值可转换为#[0,0,0,0,0,0,1]，对应我们之前把标签onthot后的第七种标签

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):  # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
        super(GCN, self).__init__()  # super()._init_()在利用父类里的对象构造函数
        self.gc1 = GraphConvolution(nfeat, nhid)  # gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2=GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)  # gc2输入尺寸nhid，输出尺寸ncalss
        self.dropout = dropout

    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        x = F.relu(self.gc2(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

    def savector(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        x = F.relu(self.gc2(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        # x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        # x = self.gc2(x, adj)
        return x

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha  # 学习因子
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # 建立都是0的矩阵，大小为（输入维度，输出维度）
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # 见下图
        # print(self.a.shape)  torch.Size([16, 1])
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # print(input.shape)
        # print(self.W.shape)
        h = torch.mm(input, self.W)
        # print(h.shape)  torch.Size([2708, 8]) 8是label的个数
        N = h.size()[0]
        # print(N)  2708 nodes的个数
        # 计算attention的方便简单的方法
        # numpy的repeat的方法是在指定的轴repeat相应地次数，比如说：
        # a.repeat(4,axis=0),由(2,3)变为(8,3)
        #attention的核心，对输入进行交叉处理
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)  # 见下图
        # print(a_input.shape)  torch.Size([2708, 2708, 16])
        # idxs1, idxs2 = [], []
        # 这样添加太慢了
        # for i in range(N):
        #     for j in range(N):
        #         idxs1.append(i)
        #         idxs2.append(j)
        # a_input = torch.cat([h[idxs1], h[idxs2]], dim=1).view(N, N, -1)
        #torch.mul矩阵点乘
        #torch.mm和torch.matmul,前者针对二维矩阵，后者针对高维，在这里是高维
        #torch.matmul支持高位运算，如：
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # 即论文里的eij
        # squeeze除去维数为1的维度
        # [2708, 2708, 16]与[16, 1]相乘再除去维数为1的维度，故其维度为[2708,2708],与领接矩阵adj的维度一样

        zero_vec = -9e15 * torch.ones_like(e)
        # 维度大小与e相同，所有元素都是-9*10的15次方
        # zero_vec=zero_vec.mul((adj <= 0).int())
        attention = torch.where(adj > 0, e, zero_vec)
        # attention = e.add(zero_vec)
        '''这里我们回想一下在utils.py里adj怎么建成的：两个节点有边，则为1，否则为0。
        故adj的领接矩阵的大小为[2708,2708]。(不熟的自己去复习一下图结构中的领接矩阵)。
        print(adj）这里我们看其中一个adj
        tensor([[0.1667, 0.0000, 0.0000,  ..., 0.0000, 0.0000,   0.0000],
        [0.0000, 0.5000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.2000,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 0.2000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.2000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.2500]])
        不是1而是小数是因为进行了归一化处理
        故当adj>0，即两结点有边，则用gat构建的矩阵e，若adj=0,则另其为一个很大的负数，这么做的原因是进行softmax时，这些数就会接近于0了。

        '''
        attention = F.softmax(attention, dim=1)
        # 对应论文公式3，attention就是公式里的αij
        '''print(attention)
        tensor([[0.1661, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.5060, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.2014,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 0.1969, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.1998, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.2548]]'''
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        # 如果concat, 说明后面还有层，加上个激活函数，如果没有层那么concat为负值，直接返回
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid,nclass, dropout, alpha, nheads,emb_dim=32):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        # 多个attention并排构成multi-attention
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        # 输入到隐藏层,将这些attention层添加到模块中，便于求导
        # add_module(name,layer)在正向传播的过程中可以使用添加时的name来访问改layer
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        #self.linear=nn.Linear(emb_dim,nclass)
        # attention层的输出会用elu激活函数
        # multi-head 隐藏层到输出

    def forward(self, x, adj):
        # 在使用F.dropout时，必须传入当前网络的training状态，用nn.Dropout则不必
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x=self.out_att(x,adj)
        x = F.elu(x)
        return F.log_softmax(x, dim=1)

    def savector(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

def random_surfing(adj: np.ndarray, epochs: int, alpha: float) -> np.ndarray:
    """
    :param adj: 邻接矩阵，numpy数组
    :param epochs: 最大迭代次数
    :param alpha: random surf 过程继续的概率
    :return: numpy数组
    """
    N = adj.shape[0]
    # 在此进行归一化
    P0, P = np.eye(N), np.eye(N)
    mat = np.zeros((N, N))
    for _ in range(epochs):
        P = alpha * adj.dot(P) + (1 - alpha) * P0
        mat = mat + P
    return mat

def PPMI_matrix(mat: np.ndarray) -> np.ndarray:
    """
    :param mat: 上一步构建完成的corjuzhen
    """
    m, n = mat.shape
    assert m == n
    D = np.sum(mat)
    col_sums = np.sum(mat, axis=0)
    row_sums = np.sum(mat, axis=1).reshape(-1, 1)
    dot_mat = row_sums * col_sums
    PPMI = np.log(D * mat / dot_mat)
    PPMI = np.maximum(PPMI, 0)
    PPMI[np.isinf(PPMI)] = 0
    #  PPMI = PPMI / PPMI.sum(1).reshape(-1, 1)
    return PPMI

class AutoEncoderLayer(nn.Module):
    """
    堆叠的自编码器中的单一一层
    """
    def __init__(self, input_dim, output_dim, zero_ratio, GPU, activation):
        super(AutoEncoderLayer, self).__init__()
        self.zero_ratio = zero_ratio
        self.GPU = GPU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            # nn.LeakyReLU(negative_slope=0.2)
        )
        if activation == 'relu':
            # 前一个是名字，后一个是参数
            self.decoder.add_module(activation, nn.ReLU())
        elif activation == 'leaky_relu':
            self.decoder.add_module(activation, nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, zero=True):
        if not self.GPU:
            if zero:
                x = x.cpu().clone()
                rand_mat = torch.rand(x.shape)
                zero_mat = torch.zeros(x.shape)
                # 随机初始化为0
                x = torch.where(rand_mat > self.zero_ratio, x, zero_mat)
        else:
            if zero:
                x = x.clone()
                # 直接在cuda中创建数据，速度会快很多
                rand_mat = torch.rand(x.shape, device='cuda')
                zero_mat = torch.zeros(x.shape, device='cuda')
                # 随机初始化为0
                x = torch.where(rand_mat > self.zero_ratio, x, zero_mat)
        x = self.encoder(x)
        #   x = F.leaky_relu(x, negative_slope=0.2)
        x = self.decoder(x)
        return x

    def emb(self, x):
        """
        获得该层的嵌入向量
        """
        x = self.encoder(x)
        return x

class StackAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, zero_ratio, GPU=False):
        super(StackAutoEncoder, self).__init__()
        assert len(hidden_dims) >= 1
        self.num_layers = len(hidden_dims) + 1
        self.zero_ratio = zero_ratio
        self.GPU = GPU
        setattr(self, 'autoEncoder0', AutoEncoderLayer(input_dim, hidden_dims[0], zero_ratio=zero_ratio, GPU=GPU,
                                                       activation='relu'))
        for i in range(1, len(hidden_dims)):
            setattr(self, 'autoEncoder{}'.format(i),
                    AutoEncoderLayer(hidden_dims[i - 1], hidden_dims[i], zero_ratio=zero_ratio, GPU=GPU,
                                     activation='relu'))
        setattr(self, 'autoEncoder{}'.format(self.num_layers - 1),
                AutoEncoderLayer(hidden_dims[-1], output_dim, zero_ratio=zero_ratio, GPU=GPU,
                                 activation='relu'))
        self.init_weights()

    def emb(self, x):
        for i in range(self.num_layers):
            x = getattr(self, 'autoEncoder{}'.format(i)).emb(x)
        return x

    def forward(self, x):
        # for i in range(self.num_layers):
        #     x = getattr(self, 'autoEncoder{}'.format(i))(x, False)
        for i in range(self.num_layers):
            x = getattr(self, 'autoEncoder{}'.format(i)).encoder(x)
        #      x=F.leaky_relu(x,negative_slope=0.2)
        for i in range(self.num_layers - 1, -1, -1):
            x = getattr(self, 'autoEncoder{}'.format(i)).decoder(x)
        #      x = F.leaky_relu(x, negative_slope=0.2)
        return x

    def init_weights(self):
        # 初始化参数十分重要，可以显著降低loss值
        # 初始化模型参数
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                # mean=0,std = gain * sqrt(2/fan_in + fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1)
            #   nn.init.xavier_uniform_(m.bias,gain=1)
            #    nn.init.uniform_()
            if isinstance(m, nn.BatchNorm1d):
                # nn.init.constant(m.weight, 1)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # nn.init.constant(m.bias, 0)

class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """聚合节点邻居
        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    #对模型的参数进行初始化
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)
#这个才是相当于神经网络中的一层
class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 #默认激活函数为relu
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """SageGCN层定义
        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
            当aggr_hidden_method=sum, 输出维度为hidden_dim
            当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.b = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.b)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.b)

        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            #这里输出的维度为2*hidden_dim
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)

class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def saveVec(self,X):
        self.eval()
        for l in range(self.num_layers):
            X=self.gcn[l](X,)
        return X

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )


class AMGCN(nn.Module):
    def __init__(self, num_feat, num_hidden, n_class, alpha, beta, sigma, embedding_dim=32, dropout=0.5):
        super(AMGCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        for i in range(1, 4):
            setattr(self, 'gc{}'.format(2 * i - 1), GraphConvolution(num_feat, num_hidden))
            setattr(self, 'gc{}'.format(2 * i), GraphConvolution(num_hidden, embedding_dim))
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim)))
            setattr(self, 'b{}'.format(i), nn.Parameter(torch.FloatTensor(embedding_dim)))
        self.Q = nn.Parameter(torch.FloatTensor(embedding_dim, 1))
        self.linear = nn.Linear(embedding_dim, n_class)
        self.norm1 = nn.BatchNorm1d(embedding_dim)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.embedding_dim)
        nn.init.xavier_uniform_(self.W1, gain=1)
        nn.init.xavier_uniform_(self.W2, gain=1)
        nn.init.xavier_uniform_(self.W3, gain=1)
        nn.init.xavier_uniform_(self.linear.weight,gain=1)
        nn.init.uniform_(self.b1, -stdv, stdv)
        nn.init.uniform_(self.b2, -stdv, stdv)
        nn.init.uniform_(self.b3, -stdv, stdv)
        nn.init.uniform_(self.Q, -stdv, stdv)
        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.zero_()


    def emb(self, x, adj1, adj2, require_weight=False):
        x1 = F.leaky_relu(self.gc1(x, adj1), negative_slope=0.2)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x1 = F.dropout(x1, self.dropout, training=self.training)  # x要dropout
        x1 = self.gc2(x1, adj1)
        x2 = F.leaky_relu(self.gc3(x, adj2), negative_slope=0.2)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x2 = F.dropout(x2, self.dropout, training=self.training)  # x要dropout
        x2 = self.gc4(x2, adj2)
        x1_c = F.leaky_relu(self.gc5(x, adj1), negative_slope=0.2)
        x1_c = F.dropout(x1_c, self.dropout, training=self.training)
        x1_c = self.gc6(x1_c, adj1)
        x2_c = F.leaky_relu(self.gc5(x, adj2), negative_slope=0.2)
        x2_c = F.dropout(x2_c, self.dropout, training=self.training)
        x2_c = self.gc6(x2_c, adj1)
        x_c = (x1_c + x2_c) / 2
        w1 = F.leaky_relu((torch.mm(x1, self.W1) + self.b1).mm(self.Q), negative_slope=0.2)
        w2 = F.leaky_relu((torch.mm(x2, self.W2) + self.b2).mm(self.Q), negative_slope=0.2)
        w3 = F.leaky_relu((torch.mm(x_c, self.W3) + self.b3).mm(self.Q), negative_slope=0.2)
        w = torch.softmax(torch.cat((w1, w2, w3), dim=1), dim=1)
        # print(x1.shape)
        # print(w[:,0].shape)
        emb = w[:, 0].reshape(-1, 1) * x1 + w[:, 1].reshape(-1, 1) * x2 + w[:, 2].reshape(-1, 1) * x_c
        emb = self.norm1(emb)
        if require_weight:
            return x1, x2, x1_c, x2_c, w, emb
        return x1, x2, x1_c, x2_c, emb

    def forward(self, x, adj1, adj2):
        x1, x2, x1_c, x2_c, emb = self.emb(x, adj1, adj2)
        x = self.linear(emb)
        return x1, x2, x1_c, x2_c, emb, torch.log_softmax(x, dim=1)

    def compute_loss(self, x, adj1, adj2, Y, idx_train):
        x1, x2, x1_c, x2_c, emb, output = self.forward(x, adj1, adj2)
        l_t = F.nll_loss(output[idx_train], Y[idx_train])
        # combined的向量尽可能接近
        l_c=0
        l_d=0
        # l_c = torch.mean(torch.pow(torch.mm(x1_c, x1_c.t()) - torch.mm(x2_c, x2_c.t()), 2))
        # k_x1 = self.kernel(x1)
        # k_x1_c = self.kernel(x1_c)
        # k_x2 = self.kernel(x2)
        # k_x2_c = self.kernel(x2_c)
        #     # 这里的向量尽可能远离
        # l_d = self.hsic(k_x1, k_x1_c) + self.hsic(k_x2, k_x2_c)
        return l_t , self.alpha * l_c , self.beta * l_d, output

    def hsic(self, kX, kY):
        kXY = torch.mm(kX, kY)
        n = kXY.shape[0]
        # print(kXY.shape)
        h = torch.trace(kXY) / (n * n) + torch.mean(kX) * torch.mean(kY) - 2 * torch.mean(kXY) / n
        return h * n ** 2 / (n - 1) ** 2

    def kernel(self, X):
        n = X.shape[0]
        square_x = torch.pow(X, 2)
        sum_square_x = torch.sum(square_x, 1)
        sum_mat = sum_square_x.view(n, 1) + sum_square_x.view(1, n)
        sum_mat = sum_mat - 2 * torch.mm(X, X.t())
        return sum_mat
