import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GINConv, DenseGINConv
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (to_dense_batch, 
                                   get_laplacian,
                                   to_dense_adj,
                                   dense_to_sparse)

# Local imports
from just_balance import just_balance_pool


class NormalizeAdj(BaseTransform):
    """
    Applies the following transformation:

        A --> I - delta * L
    """
    def __init__(self, delta: float = 0.85) -> None:
        self.delta = delta
        super().__init__()

    def forward(self, data: torch.Any) -> torch.Any:
        edge_index, edge_weight = get_laplacian(data.edge_index, data.edge_weight, normalization='sym')
        L = to_dense_adj(edge_index, edge_attr=edge_weight)
        A_norm = torch.eye(data.num_nodes) - self.delta * L
        data.edge_index, data.edge_weight = dense_to_sparse(A_norm)
        return data


### Get the data
dataset = TUDataset(root="../data/TUDataset", name='NCI1', pre_transform=NormalizeAdj())
train_loader = DataLoader(dataset[:0.9], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[0.9:], batch_size=32)


### Model definition
class Net(torch.nn.Module):
    def __init__(self,
                 hidden_channels = 64,
                 mlp_units=[16],
                 mlp_act="ReLU"
                 ):
        super().__init__()

        num_features = dataset.num_features
        num_classes = dataset.num_classes
        n_clusters = int(dataset._data.x.size(0) / len(dataset)) # average number of nodes per graph
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)
        
        # First MP layer
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )
        
        self.mlp = MLP([hidden_channels] + mlp_units + [n_clusters], act=mlp_act, norm=None)
        
        # Second MP layer
        self.conv2 = DenseGINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )

        # Readout layer
        self.lin = torch.nn.Linear(hidden_channels, num_classes)


    def forward(self, x, edge_index, batch=None):

        # First MP layer
        x = self.conv1(x, edge_index)

        # Transform to dense batch
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        
        # Cluster assignments (logits)
        s = self.mlp(x)

        # Pooling
        x_pool, adj_pool, aux_loss = just_balance_pool(x, adj, s, mask, normalize=True)

        # Second MP layer
        x = self.conv2(x_pool, adj_pool)

        # Global pooling
        x = x.mean(dim=1)

        # Readout layer
        x = self.lin(x)

        return F.log_softmax(x, dim=-1), aux_loss
        

### Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train():
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, aux_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y.view(-1)) + aux_loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch)[0].max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)


### Training loop
best_val_acc = test_acc = 0
for epoch in range(1, 501):
    train_loss = train()
    val_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, '
          f'Val Acc: {val_acc:.3f}')