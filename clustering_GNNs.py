from time import time
import itertools
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.stats import entropy
from sklearn.metrics.cluster import silhouette_score
import tensorflow as tf
import tensorflow.keras.backend as K

from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.datasets.citation import Citation
from spektral.utils import init_logging
from spektral.utils import log
from spektral.utils.convolution import normalized_adjacency

from karateclub.dataset import GraphReader

from utils.GNN_clustering_models import GNNClustering
from utils.extra_datasets import Flickr, DBLP
from utils.eval_metrics import eval_metrics
from sklearn.metrics.cluster import normalized_mutual_info_score


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def get_one_hot(targets, nb_classes):
    targets = targets.astype(np.int32)
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    if len(keys) > 0:
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))
    else:
        for _ in [dict(), ]:
            yield _


spektral_sets = ['cora', 'citeseer', 'pubmed', 'dblp', 'flickr']
kc_sets = ['lastfm', 'deezer', 'facebook', 'twitch', 'wikipedia', 'github']
unsupervised_losses = ['asymmcheeger',
                       'totvar_with_ortho', 'mincut', 'diffpool', 'dmon']

# Parameters
P = OrderedDict(
    n_runs=1,
    learning_rate=1e-3,
    epochs=100,
    delta_coeff=0.001,
    epsilon=1e-3,
    normalize_a=False,
    normalize_x=True)
log_dir = init_logging("test/")
# log_dir = init_logging("clustering_gnn_asymmcheeger/")   # Create log directory and files
log(P)

# Tunables
tunables = OrderedDict([
    ('dataset', ['dblp']),  # spektral_sets + kc_sets),
    ('aux_loss', ['dmon']),  # unsupervised_losses
    # ['lqv', 'weighted_lqv', 'gtv_single', 'weighted_gtv_single']
    ('mp_method', ['gcs']),
    ('mp_layers', [[16]]),
    ('mp_activation', ["elu"]),
    ('mlp_hidden', [None]),
    ('mlp_activation', ["relu"]),
    ('delta_coeff', [1.0]),  # [0.1, 0.01, 0.001]
    ('learning_rate', [1e-2])])  # [1e-2, 1e-3, 1e-4]

log(tunables)

df_out = None
for T in product_dict(**tunables):
    # Update params with current config
    P.update(T)
    log(T)

    results = {'acc_scores': [],
               'f1_macro_scores': [],
               'f1_micro_scores': [],
               'ari_scores': [],
               'hs_scores': [],
               'cs_scores': [],
               'nmi_scores': [],
               'silhouette_scores': [],
               'entropy_scores': []}

    if P['dataset'] in ['cora', 'citeseer']:
        device_name = "CPU"
    else:
        device_name = "GPU"

    with tf.device(device_name):

        for run in range(P['n_runs']):
            ############################################################################
            # LOAD DATASET
            ############################################################################

            if P['dataset'] in spektral_sets:
                if P['dataset'] in ['cora', 'citeseer', 'pubmed']:
                    dataset = Citation(
                        P['dataset'], normalize_x=P['normalize_x'])

                elif P['dataset'] == 'flickr':
                    dataset = Flickr(normalize_x=P['normalize_x'])

                elif P['dataset'] == 'dblp':
                    dataset = DBLP(normalize_x=P['normalize_x'])

                X = dataset.graphs[0].x
                A = dataset.graphs[0].a
                Y = dataset.graphs[0].y
                y = np.argmax(Y, axis=-1)
                n_clust = Y.shape[-1]

            elif P['dataset'] in kc_sets:
                reader = GraphReader(P['dataset'])
                X = reader.get_features().toarray().astype(np.float32)

                if P['normalize_x']:
                    X = preprocess_features(X)

                A = nx.adjacency_matrix(reader.get_graph()).astype(np.float32)
                y = reader.get_target().astype(np.float32)
                n_clust = len(np.unique(y))
                Y = get_one_hot(y, n_clust).astype(np.float32)

            if P['normalize_a']:
                A = normalized_adjacency(A, symmetric=True)

            ############################################################################
            # MODEL
            ############################################################################

            opt = tf.keras.optimizers.Adam(learning_rate=P['learning_rate'])
            loss_func = tf.keras.losses.CategoricalCrossentropy()

            model = GNNClustering(n_clust,
                                  mp_method=P['mp_method'],
                                  mp_layers=P['mp_layers'],
                                  mp_activation=P['mp_activation'],
                                  aux_loss=P['aux_loss'],
                                  mlp_hidden=P['mlp_hidden'],
                                  mlp_activation=P['mlp_activation'],
                                  delta_coeff=P['delta_coeff'],
                                  epsilon=P['epsilon'],
                                  totvar_coeff=1.0,
                                  balance_coeff=1.0,
                                  collapse_regularization=1.0)

            ############################################################################
            # TRAINING
            ############################################################################

            @tf.function(input_signature=None)
            def train_step(model, inputs, labels):
                with tf.GradientTape() as tape:
                    preds = model(inputs, training=True)
                    loss = sum(model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(gradients, model.trainable_variables))
                return model.losses

            A = sp_matrix_to_sp_tensor(A)
            inputs = [X, A]
            labels = Y
            epochs = P['epochs']
            loss_history = []
            ep_time = []
            # Fit model
            # for _ in tqdm(range(epochs)):
            for ep in range(epochs):
                time_s = time()
                outs = train_step(model, inputs, labels)
                time_e = time()
                # print([outs[i].numpy() for i in range(len(outs))])
                loss_history.append([outs[i].numpy()
                                     for i in range(len(outs))])
                ep_time.append(time_e - time_s)

                S_ = model(inputs, training=False)
                s = np.argmax(S_, axis=-1)
                nmi = normalized_mutual_info_score(y, s)
                print("ep:", ep, ", elapsed:", (time_e - time_s),
                      ", nmi:", nmi, ", loss:", outs[0].numpy())

            ep_time.pop(0)
            print("avg time: ", np.average(ep_time))

            ############################################################################
            # RESULTS
            ############################################################################
            S_ = model(inputs, training=False)
            s = np.argmax(S_, axis=-1)
            acc, f1_macro, f1_micro, ari, hs, cs, nmi = eval_metrics(y, s)

            if len(np.unique(s)) > 1:
                silhouette = silhouette_score(S_, s, metric="euclidean")
            else:
                silhouette = 0.
            entropy_score = np.mean(entropy(S_, axis=1))

            results['acc_scores'].append(acc)
            results['f1_macro_scores'].append(f1_macro)
            results['f1_micro_scores'].append(f1_micro)
            results['ari_scores'].append(ari)
            results['hs_scores'].append(hs)
            results['cs_scores'].append(cs)
            results['nmi_scores'].append(nmi)
            results['silhouette_scores'].append(silhouette)
            results['entropy_scores'].append(entropy_score)

            K.clear_session()

        P['acc_mean'] = np.mean(results['acc_scores'])
        P['acc_std'] = np.std(results['acc_scores'])
        P['f1_macro_mean'] = np.mean(results['f1_macro_scores'])
        P['f1_macro_std'] = np.std(results['f1_macro_scores'])
        P['f1_micro_mean'] = np.mean(results['f1_micro_scores'])
        P['f1_micro_std'] = np.std(results['f1_micro_scores'])
        P['ari_mean'] = np.mean(results['ari_scores'])
        P['ari_std'] = np.std(results['ari_scores'])
        P['hs_mean'] = np.mean(results['hs_scores'])
        P['hs_std'] = np.std(results['hs_scores'])
        P['cs_mean'] = np.mean(results['cs_scores'])
        P['cs_std'] = np.std(results['cs_scores'])
        P['nmi_mean'] = np.mean(results['nmi_scores'])
        P['nmi_std'] = np.std(results['nmi_scores'])
        P['silhouette_score_mean'] = np.mean(results['silhouette_scores'])
        P['silhouette_score_std'] = np.std(results['silhouette_scores'])
        P['entropy_mean'] = np.mean(results['entropy_scores'])
        P['entropy_std'] = np.std(results['entropy_scores'])

        if df_out is None:
            df_out = pd.DataFrame([P])
        else:
            df_out = pd.concat([df_out, pd.DataFrame([P])])
        df_out.to_csv(log_dir + 'results.csv')

        print("NMI: ", P['nmi_mean'])
