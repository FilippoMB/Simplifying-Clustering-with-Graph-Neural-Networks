import tensorflow as tf
from tensorflow.keras import backend as K
from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv


# NOTE: All layers currently only work in single mode with sparse adjacency

class LQVConv(Conv):
    """
    Update docstring
    """

    def __init__(
        self,
        channels,
        delta_coeff=1.,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.delta_coeff = delta_coeff

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs, mask=None, n_nodes=None):
        x, a = inputs

        mode = ops.autodetect_mode(x, a)

        # Apply the weight kernel to the node features before aggregation
        x = K.dot(x, self.kernel)

        if mode == ops.modes.SINGLE and K.is_sparse(a):
            output = self._call_single(x, a)

        elif mode == ops.modes.BATCH:
            output = self._call_batch(x, a)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        return output

    def _call_single(self, x, a):

        n_nodes = tf.shape(a, out_type=(a.indices).dtype)[0]

        # Compute degree matrix
        degrees = tf.sparse.reduce_sum(a, axis=-1)
        d_mat = tf.sparse.SparseTensor(tf.stack([tf.range(n_nodes)] * 2, axis=1),
                                       degrees,
                                       [n_nodes, n_nodes])

        # Compute Laplacian
        l_mat = tf.sparse.add(d_mat, tf.sparse.map_values(tf.multiply, a, -1))

        # Compute adjusted laplacian
        l_mat = tf.sparse.add(tf.sparse.eye(n_nodes),
                              tf.sparse.map_values(tf.multiply, l_mat, -2 * self.delta_coeff))

        # Aggregate features with adjusted laplacian
        output = ops.modal_dot(l_mat, x)

        return output

    def _call_batch(self, x, a):
        degrees = tf.math.reduce_sum(a, axis=-1)
        l = -a
        l = tf.linalg.set_diag(l, degrees - tf.linalg.diag_part(a))
        l = tf.eye(a.shape[-1]) - 2 * self.delta_coeff * l

        output = tf.matmul(l, x)

        return output

    @property
    def config(self):
        return {"channels": self.channels,
                "delta_coeff": self.delta_coeff}


# TODO: ADD WEIGHTED_LQV, GTV and WEIGTHED_GTV (BOTH SINGLE AND MULTI)

class WLQVConv(LQVConv):
    def _call_single(self, x, a):
        n_nodes = tf.shape(a, out_type=(a.indices).dtype)[0]

        # Compute D^(-1/2)
        degrees_sqrt = 1 / tf.math.sqrt(tf.sparse.reduce_sum(a, axis=-1))
        d_sqrt_mat = tf.sparse.SparseTensor(tf.stack([tf.range(n_nodes)] * 2, axis=1),
                                            degrees_sqrt,
                                            [n_nodes, n_nodes])

        # Compute symmetrically normalized adjacency matrix: A_symm =  D^(-1/2)*A*D^(-1/2)
        a_symm = ops.matmul_at_b_a(d_sqrt_mat, a)

        # Compute normalized Laplacian: L_symm = I - A_symm
        l_symm = tf.sparse.add(tf.sparse.eye(n_nodes),
                               tf.sparse.map_values(tf.multiply, a_symm, -1))

        # Compute the connectivity matrix: I - \delta*L_symm
        conn = tf.sparse.add(tf.sparse.eye(n_nodes),
                             tf.sparse.map_values(tf.multiply, l_symm, -self.delta_coeff))

        # Aggregate the node features
        output = ops.modal_dot(conn, x)

        return output

    def _call_batch(self, x, a):
        degrees_sqrt = 1 / \
            tf.math.sqrt(tf.math.reduce_sum(a, axis=-1, keepdims=True))
        a_symm = (degrees_sqrt * a) * tf.transpose(degrees_sqrt, perm=[0, 2, 1])
        # (This line uses that (I - delta*(I-A_symm)) = (I + delta*(A_symm-I)))
        l = tf.linalg.set_diag(a_symm, 1 + self.delta_coeff *
                               (tf.linalg.diag_part(a_symm) - 1))

        output = tf.matmul(l, x)

        return output
