import numpy as np
import pylab
from sklearn.manifold import TSNE
from pathlib import Path
import networkx as nx

if __name__ == "__main__":
    dataset='citeseer'
    emb_path=Path(__file__).parent/('{}_outVec.txt'.format(dataset))
    label_path=Path(__file__).parent/('{}_labels.txt'.format(dataset))
    X = np.loadtxt(emb_path)
    labels_data = np.loadtxt(label_path).astype(int)
    tsne = TSNE(n_components=2, init='random')
    Y = tsne.fit_transform(X)    #后面两个参数分别是邻居数量以及投影的维度
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels_data)
    pylab.savefig('{}_embedding.png'.format(dataset))
    pylab.show()
