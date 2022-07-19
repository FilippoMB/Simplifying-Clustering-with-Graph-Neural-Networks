import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool


class JustBalancePool(SRCPool):
    """
    JustBalancePool layer
    """

    def __init__(self,
                 k,
                 mlp_hidden=None,
                 mlp_activation="relu",
                 normalized_loss=False,
                 return_selection=False,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                 ):
        super().__init__(
            return_selection=return_selection,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.k = k
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = mlp_activation
        self.normalized_loss = normalized_loss

    def build(self, input_shape):
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.mlp = Sequential(
            [
                Dense(channels, self.mlp_activation, **layer_kwargs)
                for channels in self.mlp_hidden
            ]
            + [Dense(self.k, "softmax", **layer_kwargs)]
        )

        super().build(input_shape)

    def call(self, inputs, mask=None):
        x, a, i = self.get_inputs(inputs)
        return self.pool(x, a, i, mask=mask)

    def select(self, x, a, i, mask=None):
        s = self.mlp(x)
        if mask is not None:
            s *= mask[0]

        self.add_loss(self.balance_loss(s))

        return s

    def reduce(self, x, s, **kwargs):
        return ops.modal_dot(s, x, transpose_a=True)

    def connect(self, a, s, **kwargs):
        a_pool = ops.matmul_at_b_a(s, a)

        # Post-processing of A
        a_pool = tf.linalg.set_diag(
            a_pool, tf.zeros(K.shape(a_pool)[:-1], dtype=a_pool.dtype)
        )
        a_pool = ops.normalize_A(a_pool)

        return a_pool

    def reduce_index(self, i, s, **kwargs):
        i_mean = tf.math.segment_mean(i, i)
        i_pool = ops.repeat(i_mean, tf.ones_like(i_mean) * self.k)

        return i_pool

    def get_config(self):
        config = {
            "k": self.k,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def balance_loss(self, s):
        ss = ops.modal_dot(s, s, transpose_a=True)
        loss = -tf.linalg.trace(tf.math.sqrt(ss))

        if self.normalized_loss:
            n = float(tf.shape(s, out_type=tf.int32)[-2])
            c = float(tf.shape(s, out_type=tf.int32)[-1])
            loss = loss / tf.math.sqrt(n * c)
        return loss
