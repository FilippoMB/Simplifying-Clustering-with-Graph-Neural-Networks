import os
import os.path as osp
import numpy as np
from spektral.data import Dataset, Graph
from spektral.utils import label_to_one_hot
import scipy.sparse as sp
import sys, ssl, errno, urllib


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

def download_url(url: str, folder: str, log: bool = True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    filename = filename if filename[0] == '?' else filename.split('?')[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def _preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


class DBLP(Dataset):
    """
    A subset of the DBLP computer science bibliography website, 
    as collected in the "MAGNN: Metapath Aggregated Graph Neural Network 
    for Heterogeneous Graph Embedding" paper.
    """
    def __init__(self, normalize_x = False, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.normalize_x = normalize_x
        super().__init__(**kwargs)

    url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz'

    def download(self):
        print("Downloading DBLP dataset.")
        download_url(self.url, self.path)

    def read(self):
        f = np.load(osp.join(self.path, 'dblp.npz'))


        x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                          f['attr_shape']).toarray()
        x[x > 0] = 1

        if self.normalize_x:
            print("Pre-processing node features")
            x = _preprocess_features(x)

        a = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                            f['adj_shape'])#.tocoo()

        y = f['labels']
        y = label_to_one_hot(y, np.unique(y))

        return [
            Graph(
                x=x.astype(self.dtype),
                a=a.astype(self.dtype),
                y=y.astype(self.dtype),
            )
        ]
