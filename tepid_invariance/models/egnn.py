from egnn_pytorch import EGNN as EGNNLayer
from eqv_transformer.utils import Swish
from lie_conv.utils import Pass
from tepid_invariance.models.point_neural_network import PointNeuralNetwork
from torch import nn


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


class EnEquivariant(PointNeuralNetwork):
    """Adapted from https://github.com/anonymous-code-0/lie-transformer"""

    def _process_inputs(self, x):
        """This network takes a tuple."""
        return x[1].cuda(), x[0].cuda()

    def build_net(self, dim_input, dim_hidden, kernel_dim, num_layers,
                  **kwargs):
        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * (num_layers + 1)

        self.net = nn.Sequential(
            EGNNLayer(dim_input, dim_hidden[0], kernel_dim),
            *[EGNNLayer(dim_hidden[i - 1], dim_hidden[i], kernel_dim)
              for i in range(1, num_layers)],
            Pass(nn.Linear(dim_hidden[-1], dim_hidden[-1] // 2), dim=0),
            Pass(Swish(), dim=0),
            Pass(nn.Linear(dim_hidden[-1] // 2, 1), dim=0)
        )

    def forward(self, x):
        for layer in self.net:
            if isinstance(layer, EGNN):
                x = layer(*x)
            else:
                x = layer(x)
        return x[0].squeeze()


class EGNN(PointNeuralNetwork):

    # We have our own initialisation methods for EGNN
    @staticmethod
    def xavier_init(m):
        pass

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return [i.cuda() for i in x]

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
            Pass(nn.Linear(dim_input, k), dim=1),
            *[EGNNPass(egnn()) for _ in range(num_layers)],
            Pass(nn.Linear(k, k), dim=1),
            Pass(nn.SiLU(), dim=1),
            Pass(nn.Linear(k, 1), dim=1)
        )

    def forward(self, x):
        return self.layers(x)[1].flatten()
