Code to reproduce the results of the *Just Balance GNN* (JBGNN) model for graph clustering (and for graph pooling), as presented in the paper [Simplifying Clustering with Graph Neural Networks](https://arxiv.org/abs/2207.08779).

### Tensorflow
The official implementation of the [JustBalancePool](https://graphneural.network/layers/pooling/#justbalancepool) layer is on [Spektral](https://graphneural.network/getting-started/), the Tensorflow/Keras library for Graph Neural Networks.

Run [``main.py``](https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/blob/main/main.py) to perform node clustering.

### Pytorch
[``just_balance_pyg.py``](https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/blob/main/just_balance_pyg.py) provides a Pytorch implementation based [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html#).

## Citation

    @misc{bianchi2022simplifying,
      doi = {10.48550/ARXIV.2207.08779},
      author = {Bianchi, Filippo Maria},
      title = {Simplifying Clustering with Graph Neural Networks},
      publisher = {arXiv},
      year = {2022},
    }
