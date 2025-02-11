from time import time
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.datasets import Citation, DBLP
from spektral.utils.convolution import normalized_laplacian
from spektral.layers import GCNConv
from spektral.layers.pooling import JustBalancePool

from eval_metrics import eval_metrics

# Hyperparameters
dataset_name = 'cora'
delta = 0.85
mp_layers = 10
mp_channels = 64
mlp_hidden = [16]
learning_rate = 1e-4
epochs = 2000

# Load dataset
if dataset_name in ['cora', 'citeseer', 'pubmed']:
    dataset = Citation(dataset_name, normalize_x=True)

elif dataset_name == 'dblp':
    dataset = DBLP(normalize_x=True)

X = dataset.graphs[0].x
A = dataset.graphs[0].a
Y = dataset.graphs[0].y
y = np.argmax(Y, axis=-1)
n_clusters = Y.shape[-1]
N, F = X.shape

# Build connectivity matrix
A_tilde = sp.eye(N) - delta*normalized_laplacian(A)
A_tilde = sp_matrix_to_sp_tensor(A_tilde)

# Build model
x_in = Input(shape=(F,), name="X_in")
a_in = Input(shape=(None,), name="A_in", sparse=True)

x_bar = x_in
for _ in range(mp_layers):
    x_bar = GCNConv(mp_channels, activation='relu')([x_bar, a_in])

_, _, s = JustBalancePool(n_clusters, 
                          mlp_hidden=mlp_hidden,
                          mlp_activation='relu',
                          return_selection=True)([x_bar, a_in])
model = Model([x_in, a_in], [s])

# Training
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

@tf.function(input_signature=None)
def train_step(model, inputs, labels):
    with tf.GradientTape() as tape:
        _ = model(inputs, training=True)
        loss = sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return model.losses

loss_history = []
nmi_history = []
ep_time = []
for _ in tqdm(range(epochs)):
    time_s = time()
    outs = train_step(model, [X, A_tilde], Y)
    time_e = time()
    
    loss_history.append([outs[i].numpy()
                         for i in range(len(outs))])
    ep_time.append(time_e - time_s)

    S_ = model([X, A_tilde], training=False)
    s = np.argmax(S_, axis=-1)
    nmi = normalized_mutual_info_score(y, s)
    nmi_history.append(nmi)

# Print results
S_ = model([X, A_tilde], training=False)
s = np.argmax(S_, axis=-1)
acc, nmi = eval_metrics(y, s)
ep_time.pop(0)
print(f"ACC: {acc:.3f}, NMI: {nmi:.3f}, avg seconds/step: {np.average(ep_time):.3f}s")

# Plots
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(loss_history, label="Balance loss")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Iteration")

plt.subplot(122)
plt.plot(nmi_history, label="NMI")
plt.legend()
plt.ylabel("NMI")
plt.xlabel("Iteration")

plt.tight_layout()
plt.show()