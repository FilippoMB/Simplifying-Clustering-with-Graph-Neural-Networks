import tensorflow as tf
from tensorflow.keras import regularizers
from spektral.layers import GCSConv
from spektral.layers import MinCutPool, DiffPool

from .aggregation_layers import LQVConv, WLQVConv, GTVSingleConv, WGTVSingleConv
from .aggregation_layers import GTVMultiConv, WGTVMultiConv
from .pooling_layers import *

class GNNClustering(tf.keras.Model):
    def __init__(self,
                 num_clusters,
                 mp_method="lqv",
                 mp_layers=[16],
                 mp_activation="elu",
                 aux_loss="mincut",
                 mlp_hidden=None,
                 mlp_activation=None,
                 delta_coeff = 1.,
                 epsilon=1e-10,
                 **kwargs):
        super().__init__()

        mp_dict = {"lqv": LQVConv,
                   "weighted_lqv": WLQVConv,
                   "gtv_single": GTVSingleConv,
                   "gtv_multi": GTVMultiConv,
                   "weighted_gtv_single": WGTVSingleConv,
                   "weighted_gtv_multi": WGTVMultiConv,
                   "gcs": GCSConv}

        if mp_method not in mp_dict.keys():
            raise ValueError("\"{}\" is not a valid option for mp_method".format(mp_method))


        self.mp = [mp_dict[mp_method](_channels,
                                      delta_coeff=delta_coeff,
                                      epsilon=epsilon,
                                      kernel_initializer="he_normal",
                                      activation=mp_activation)
                   for _channels in mp_layers]

        if aux_loss == "mincut":
            self.pool = MinCutPool(
                num_clusters,
               mlp_hidden=mlp_hidden,
               mlp_activation=mlp_activation,
               return_selection=True)

        elif aux_loss == "diffpool":
            self.pool = DiffPool(
                num_clusters,
                 return_selection=True)

        elif aux_loss == "dmon":
            self.pool = DMoNPool(
                num_clusters,
                 mlp_hidden=mlp_hidden,
                 mlp_activation=mlp_activation,
                 return_selection=True,
                 **kwargs)

        elif aux_loss == "asymmcheeger":
            self.pool = AsymmetricCheegerCutPool(
                num_clusters,
                 mlp_hidden=mlp_hidden,
                 mlp_activation=mlp_activation,
                 return_selection=True,
                 **kwargs)

        elif aux_loss == "totvar_with_ortho":
            self.pool = TotvarOrthoPool(
                num_clusters,
                mlp_hidden=mlp_hidden,
                mlp_activation=mlp_activation,
                return_selection=True,
                **kwargs)

        elif aux_loss == "just_balance":
            self.pool = JustBalance(
                num_clusters,
                mlp_hidden=mlp_hidden,
                mlp_activation=mlp_activation,
                return_selection=True,
                **kwargs)

        elif aux_loss == "mincut_custom":
            self.pool = MinCutPool_custom(
                num_clusters,
                mlp_hidden=mlp_hidden,
                mlp_activation=mlp_activation,
                return_selection=True,
                **kwargs)


        else:
            raise ValueError("\"{}\" is not a valid option for aux_loss".format(aux_loss))


    def call(self, inputs):
        x, a = inputs

        for mp in self.mp:
            x = mp([x, a])
        out = x

        _, _, s_pool = self.pool([out, a])
        return s_pool