from egnn_pytorch import EGNN as EGNNLayer
from lie_conv.utils import Pass
from torch import nn

from tepid_invariance.models.point_neural_network import PointNeuralNetwork


class EGNNPass(nn.Module):
    def __init__(self, egnn):
        super().__init__()
        self.egnn = egnn

    def forward(self, x):
        if len(x) == 2:
            coors, feats = x
            mask = None
        else:
            coors, feats, mask = x
        feats, coors = self.egnn(feats=feats, coors=coors, mask=mask)
        return coors, feats, mask


class EGNN(PointNeuralNetwork):

    # We have our own initialisation methods for EGNN
    @staticmethod
    def xavier_init(m):
        pass

    def _process_inputs(self, x):
        return tuple([i.cuda() for i in x])

    def build_net(self, dim_input, dim_output=1, dim_hidden=12, nbhd=0,
                  dropout=0.0, num_layers=6, fourier_features=16,
                  norm_coords=True, norm_feats=True, thin_mlps=False,
                  **kwargs):
        m_dim = 16
        layer_class = EGNNLayer
        egnn = lambda: layer_class(
            dim=dim_hidden, m_dim=m_dim, norm_coors=norm_coords,
            norm_feats=norm_feats, dropout=dropout,
            fourier_features=fourier_features, num_nearest_neighbors=nbhd,
            init_eps=1e-2)

        return nn.Sequential(
            Pass(nn.Linear(dim_input, dim_hidden), dim=1),
            *[EGNNPass(egnn()) for _ in range(num_layers)],
            Pass(nn.Linear(dim_hidden, 1), dim=1)
        )

    def forward(self, x):
        return self.layers(x)[1].squeeze()
