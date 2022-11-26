Software implementation and code to reproduce the results of the *Just Balance GNN* (JBGNN) model for graph clustering as presented in the paper [Simplifying Clustering with Graph Neural Networks](https://arxiv.org/abs/2207.08779).

The JBGNN architecture consists of:

- a GCN layer operating on the connectivity matrix: $\mathbf{I} - \delta ( \mathbf{I} - \mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}} )$;
- a pooling layer that computes a cluster assignment matrix $\mathbf{S} \in \mathbb{R}^{N \times K}$ as

$$ \mathbf{S} = \texttt{softmax} \left( \texttt{MLP} \left( \mathbf{\bar X}, \boldsymbol{\Theta}_\text{MLP} \right) \right) $$

>> where $\mathbf{\bar X}$ are the node features returned by a stack of GCN layers.
Each pooling layer is associated with an unsupervised loss that balances the size of the clusters and prevents degenerate partitions

$$\mathcal{L} = \text{Tr}\left( \sqrt{\mathbf{S}^T\mathbf{S} } \right).$$

## Clustering of graph vertices
<img align="left" width="30" height="30" src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" alt="Tensorflow icon">

### Tensorflow
The official implementation of the [JustBalancePool](https://graphneural.network/layers/pooling/#justbalancepool) layer is on [Spektral](https://graphneural.network/getting-started/), the Tensorflow/Keras library for Graph Neural Networks.

Run [``main.py``](https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/blob/main/main.py) to perform node clustering.

<img align="left" width="30" height="30" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" alt="Pytorch icon">

### Pytorch
[``just_balance_pyg.py``](https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/blob/main/just_balance_pyg.py) provides a Pytorch implementation based on [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html#). Run [``main_pyg.py``](https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/blob/main/main_pyg.py) to perform node clustering in Pytorch.

**Note**: the results on the paper are based on the Tensorflow implementation.

## Poster 
The poster presentation of the paper *Simplifying Clustering with Graph Neural Networks* can be downloaded [here](https://drive.google.com/file/d/1cXA0LTHcdTV8Q0-1cjabr7eayM7gKBbh/view?usp=share_link).

## Citation

    @misc{bianchi2022simplifying,
      doi = {10.48550/ARXIV.2207.08779},
      author = {Bianchi, Filippo Maria},
      title = {Simplifying Clustering with Graph Neural Networks},
      publisher = {arXiv},
      year = {2022},
    }
