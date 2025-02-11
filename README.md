[![arXiv](https://img.shields.io/badge/arXiv-2207.08779-b31b1b.svg?)](https://arxiv.org/abs/2207.08779)
[![poster](https://custom-icon-badges.demolab.com/badge/poster-pdf-orange.svg?logo=note&logoSource=feather&logoColor=white)](https://drive.google.com/file/d/1cXA0LTHcdTV8Q0-1cjabr7eayM7gKBbh/view?usp=share_link)

Software implementation and code to reproduce the results of the *Just Balance GNN* (JBGNN) model for graph clustering as presented in the paper [Simplifying Clustering with Graph Neural Networks](https://arxiv.org/abs/2207.08779).

The JBGNN architecture consists of:

- a GCN layer operating on the connectivity matrix: $\mathbf{I} - \delta ( \mathbf{I} - \mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}} )$;
- a pooling layer that computes a cluster assignment matrix $\mathbf{S} \in \mathbb{R}^{N \times K}$ as

$$ \mathbf{S} = \texttt{softmax} \left( \texttt{MLP} \left( \mathbf{\bar X}, \boldsymbol{\Theta}_\text{MLP} \right) \right) $$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where $\mathbf{\bar X}$ are the node features returned by a stack of GCN layers.

Each pooling layer is associated with an unsupervised loss that balances the size of the clusters and prevents degenerate partitions

$$\mathcal{L} = - \text{Tr}\left( \sqrt{\mathbf{S}^T\mathbf{S} } \right).$$

## Node clustering and graph classification
<img align="left" width="30" height="30" src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" alt="Tensorflow icon">

### Tensorflow
A TF/Keras implementation of the [JustBalancePool](https://graphneural.network/layers/pooling/#justbalancepool) layer is on [Spektral](https://graphneural.network/getting-started/).

Run [``example_clustering_tf.py``](https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/blob/main/example_clustering_tf.py) to perform node clustering.

<img align="left" width="30" height="30" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" alt="Pytorch icon">

### Pytorch
[``just_balance.py``](https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/blob/main/just_balance.py) provides a Pytorch implementation based on [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html#). 

Run [``example_clustering.py``](https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/blob/main/example_clustering.py) to perform node clustering in Pytorch.

Run [``example_classification.py``](https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/blob/main/example_classification.py) to perform graph classification in Pytorch.


> [!IMPORTANT]  
> The results on the paper were obtained using the Tensorflow/Spektral implementation.



## Citation

```bibtex
@misc{bianchi2022simplifying,
  doi = {10.48550/ARXIV.2207.08779},
  author = {Bianchi, Filippo Maria},
  title = {Simplifying Clustering with Graph Neural Networks},
  publisher = {arXiv},
  year = {2022},
}
```