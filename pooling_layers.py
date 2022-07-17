import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool
from spektral.layers.pooling import MinCutPool


class BasePool(SRCPool):
    """
    Placeholder
    """

    def __init__(self,
                 k,
                 mlp_hidden=None,
                 mlp_activation="relu",
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





class JustBalance(BasePool):

    def __init__(self,
                 k,
                 mlp_hidden=None,
                 mlp_activation="relu",
                 return_selection=False,
                 use_bias=True,
                 softmax_temperature=1.0,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                 ):
        super().__init__(
            k=k,
            mlp_hidden=mlp_hidden,
            mlp_activation=mlp_activation,
            return_selection=return_selection,
            use_bias=use_bias,
            softmax_temperature=softmax_temperature,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def select(self, x, a, i, mask=None):
        s = self.mlp(x)
        if mask is not None:
            s *= mask[0]

        # Balance loss
        self.add_loss(self.balance_loss(s))

        return s

    # def balance_loss(self, s):
    #     # Orthogonality regularization
    #     ss = ops.modal_dot(s, s, transpose_a=True)
    #     I_s = tf.eye(self.k, dtype=ss.dtype)
    #     ortho_loss = tf.norm(
    #         ss / tf.norm(ss, axis=(-1, -2), keepdims=True) - I_s / tf.norm(I_s),
    #         axis=(-1, -2),
    #     )
    #     ortho_loss /= tf.math.sqrt(2*(1 - 1/tf.math.sqrt(float(self.k)))) # Standardize to range [0, 1]

    #     return ortho_loss

    def balance_loss(self, s, normalized=False):
        ss = ops.modal_dot(s, s, transpose_a=True)
        loss = -tf.linalg.trace(tf.math.sqrt(ss))

        if normalized:
            n = float(tf.shape(s, out_type=tf.int32)[-2])
            c = float(tf.shape(s, out_type=tf.int32)[-1])
            loss = loss / tf.math.sqrt(n * c)
        return loss


class MinCutPool_custom(MinCutPool):

    def orthogonality_loss(self, s):
        ss = ops.modal_dot(s, s, transpose_a=True)
        loss = -tf.linalg.trace(tf.math.sqrt(ss))
        n = float(tf.shape(s, out_type=tf.int32)[-2])
        c = float(tf.shape(s, out_type=tf.int32)[-1])
        loss = loss / tf.math.sqrt(n * c)
        return loss
