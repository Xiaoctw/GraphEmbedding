
import argparse
import warnings
import torch
from models import *
from train import *
import time
from utils import *
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
import torch.utils.data as Data
import torch.optim as optim
parser.add_argument('--lr', type=float, default=1e-4,
                    help='The learning rate')
parser.add_argument('--surfing_epoch', type=int, default=4)
parser.add_argument('--alpha', type=float, default=0.9, help='The probability of going on to the next jump')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--zero_ratio', type=float, default=0.4, help='The probability of random 0.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dataset', type=str, default='europe-airports')
parser.add_argument('--print_every', type=int, default=20)
parser.add_argument('--w', type=float, default=0.1)
parser.add_argument('--dropout', type=int, default=0.5)
parser.add_argument('--ratio', type=int, default=0.5)
parser.add_argument('--hidden_dims', type=list, default=[512,256,40])
parser.add_argument('--epoch1s', type=list, default=[100, 100,100,100],
                    help='Number of epochs to train.')
parser.add_argument('--output_dim', type=int, default=32)
args = parser.parse_args()
save_path=Path(__file__).parent/(args.dataset+'_outVec.txt')

def load_dngr(dataset):
    if dataset in {'BlogCatalog', 'citeseer', 'cora', 'Flickr', 'pubmed'}:
        print('Loading {} dataset'.format(dataset))
        path = Path(__file__).parent / 'data'
        labels = np.load(path / '{}_labels.npy'.format(dataset))
        features = sp.load_npz(path / '{}_features.npz'.format(dataset))
        adj = sp.load_npz(path / '{}_adj.npz'.format(dataset))
        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
        adj = normalize(adj)
        features =np.array(features.todense())
        adj = adj.todense().astype('float')
        print('finish loading')
        return adj, features, labels
    else:
        print('Loading {} dataset'.format(dataset))
        path = Path(__file__).parent / 'data'
        labels = np.load(path / '{}_labels.npy'.format(dataset))
        adj = sp.load_npz(path / '{}_adj.npz'.format(dataset))
        adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
        adj = normalize(adj)
        adj = adj.todense()
        features = torch.FloatTensor(np.diag(np.ones(labels.shape[0])))
        print('finish loading')
        return adj, features, labels

adj_mat, features, labels =load_dngr(args.dataset)
N = adj_mat.shape[0]
# adj_mat=torch.FloatTensor(adj_mat)
# Y=torch.LongTensor(Y)

pco_mat = random_surfing(adj_mat, epochs=args.surfing_epoch, alpha=args.alpha)
ppmi_mat = PPMI_matrix(pco_mat)

ppmi_mat = torch.from_numpy(ppmi_mat).float()
GPU = args.cuda and torch.cuda.is_available()
# torch.set_default_tensor_type(torch.DoubleTensor)
sdae = StackAutoEncoder(input_dim=N, hidden_dims=args.hidden_dims, output_dim=args.output_dim,
                        zero_ratio=args.zero_ratio, GPU=GPU)

class DataSet(Data.Dataset):

    def __init__(self, mat):
        self.Adj = mat
        self.num_node = self.Adj.shape[0]

    def __getitem__(self, index):
        return self.Adj[index]

    def __len__(self):
        return self.num_node

def train_DNGR(model: nn.Module, X, epoch1s, lr1=3e-5, lr2=3e-6, batch_size=128, epochs2=50,
          print_every=20):
    for param in model.parameters():
        param.requires_grad = False  # 首先冻结所有的层
    num_layers = model.num_layers
    for i in range(num_layers):
        train_layer(model, i, X, lr1, epoch1s[i], print_every=print_every)
    for param in model.parameters():
        param.requires_grad = True  # 恢复计算各个层的参数梯度
    # 对模型整体进行微调
    dataSet = DataSet(X)
    dataLoader = Data.DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=True, )
    optimizer = optim.Adam(model.parameters(), lr=lr2, )  # weight_decay=5e-4)
    criterion = nn.MSELoss()
    for epoch in range(epochs2):
        loss = 0
        time1=time.time()
        for batch_x in dataLoader:
            output = model(batch_x)
            batch_loss = criterion(batch_x, output)
            optimizer.zero_grad()
            batch_loss.backward()
            loss += batch_loss.item() * batch_x.shape[0] * batch_x.shape[1]
            optimizer.step()
        if (epoch + 1) % print_every == 0 or epoch==0:
            print('Adjust model parameters, epoch:{}, loss: {:.4f}, time:{:.4f}'.format(epoch + 1, loss / X.shape[0],time.time()-time1))
    return model


def train_layer(model, i, X, lr=3e-5, epochs=200, print_every=20, batch_size=128):
    for param in getattr(model, 'autoEncoder{}'.format(i)).parameters():
        param.requires_grad = True
    for j in range(i):
        # 通过前面各层
        X = getattr(model, 'autoEncoder{}'.format(j)).emb(X)
    dataSet = DataSet(X)
    dataLoader = Data.DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=True, )
    optimizer = optim.Adam(getattr(model, 'autoEncoder{}'.format(i)).parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()
    pre_time = time.time()
    for epoch in range(epochs):
        loss = 0
        for batch_x in dataLoader:
            output_x = getattr(model, 'autoEncoder{}'.format(i))(batch_x)
            batch_loss = criterion(batch_x, output_x)
            optimizer.zero_grad()
            batch_loss.backward()  # 反向传播计算参数的梯度
            loss += batch_loss.item() * batch_x.shape[0] * batch_x.shape[1]
            optimizer.step()
        if (epoch + 1) % print_every == 0 or epoch==0:
            tem_time = time.time()
            print('Train layer {}, epoch:{}, loss:{:.4f}, time:{:.4f}'.format(i, epoch + 1, loss / X.shape[0],
                                                                              (tem_time - pre_time) / print_every))
            pre_time = tem_time
    for param in getattr(model, 'autoEncoder{}'.format(i)).parameters():
        param.requires_grad = False




if __name__ == '__main__':
    if GPU:
        sdae=sdae.cuda()
        ppmi_mat=ppmi_mat.cuda()


    train_DNGR(sdae, ppmi_mat, lr1=args.lr, lr2=args.lr, epoch1s=args.epoch1s, epochs2=args.epochs,
              batch_size=args.batch_size)
    embs = sdae.emb(ppmi_mat)
    embs=embs.cpu().detach().numpy()
    path = Path(__file__).parent / 'DNGR_result'/('{}_outVec.txt'.format(args.dataset))
    np.savetxt(path, embs)
    outputs = sdae(ppmi_mat)
    print(outputs[0][:20])
    print(ppmi_mat[0][:20])