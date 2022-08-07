import os.path as osp
import torch
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric import utils
from torch_geometric.nn import Sequential

from sklearn.metrics import normalized_mutual_info_score as NMI

from just_balance_pyg import just_balance_pool

torch.manual_seed(1) # for (inconsistent) reproducibility
torch.cuda.manual_seed(1)

# Load dataset
dataset = 'cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

# Compute connectivity matrix
delta = 0.85
edge_index, edge_weight = utils.get_laplacian(data.edge_index, data.edge_weight, normalization='sym')
L = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
A = torch.eye(data.num_nodes) - delta*L
data.edge_index, data.edge_weight = utils.dense_to_sparse(A)


class Net(torch.nn.Module):
    def __init__(self, 
                 mp_units,
                 mp_act,
                 in_channels, 
                 n_clusters, 
                 mlp_units=[],
                 mlp_act="Identity"):
        super().__init__()
        
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)
        
        # Message passing layers
        mp = [
            (GCNConv(in_channels, mp_units[0], normalize=False, cached=False), 'x, edge_index, edge_weight -> x'),
            mp_act
        ]
        for i in range(len(mp_units)-1):
            mp.append((GCNConv(mp_units[i], mp_units[i+1], normalize=False, cached=False), 'x, edge_index, edge_weight -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        out_chan = mp_units[-1]
        
        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))
        

    def forward(self, x, edge_index, edge_weight):
        
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight)
        
        # Cluster assignments (logits)
        s = self.mlp(x)
        
        # Compute loss
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, b_loss = just_balance_pool(x, adj, s)
        
        return torch.softmax(s, dim=-1), b_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = Net([64]*10, "ReLU", dataset.num_features, dataset.num_classes, [16], "ReLU").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train():
    model.train()
    optimizer.zero_grad()
    _, loss = model(data.x, data.edge_index, data.edge_weight)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    clust, _ = model(data.x, data.edge_index, data.edge_weight)
    return NMI(clust.max(1)[1].cpu(), data.y.cpu())
    

for epoch in range(1, 1001):
    train_loss = train()
    nmi = test()
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, ' f'NMI: {nmi:.3f}')
