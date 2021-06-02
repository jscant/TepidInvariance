# This is higher than usual, because for infinite Lie groups, there are several
# approximations (MC sampling, for example) which reduce the true invariance
# to approximate invariance.
from pathlib import Path

import numpy as np
import torch
from lie_conv.lieGroups import SE3

from tepid_invariance.preprocessing.data_loaders import random_rotation
from tepid_invariance.utils import to_numpy, _set_precision


def setup():
    _set_precision('float')

    # Tests should be repeatable
    torch.random.manual_seed(2)
    np.random.seed(2)

    # Check if we can write to /tmp/; if not, write to test directory
    dump_path = Path('/tmp/pointvs_test')
    try:
        open(dump_path / 'probe')
    except IOError:
        dump_path = Path('test/dump_path')
    return dump_path


EPS = 1e-4
MODEL_KWARGS = {
    'dim_input': 12,
    'dim_hidden': 16,
    'num_layers': 6,
    'num_heads': 1,
    'group': SE3(0.2),
    'liftsamples': 1,
    'block_norm': "layer_pre",
    'kernel_norm': "none",
    'kernel_type': 'mlp',
    'kernel_dim': 16,
    'kernel_act': 'relu',
    'mc_samples': 20,
    'fill': 1.0,
    'attention_fn': 'dot_product',
    'feature_embed_dim': None,
    'max_sample_norm': None,
    'lie_algebra_nonlinearity': None,
    'dropout': 0.0,
    'k': 16,
}

"""
im_input, dim_output=1, k=12, nbhd=0,
                  dropout=0.0, num_layers=6, fourier_features=16,
                  norm_coords=True, norm_feats=False, thin_mlps=False,
"""

N_SAMPLES = 10
ORIGINAL_COORDS = torch.rand(1, 100, 3).cuda()
ROTATED_COORDS = [
    torch.from_numpy(random_rotation(to_numpy(ORIGINAL_COORDS))).float().cuda()
    for _ in range(N_SAMPLES)]
FEATS = torch.rand(1, 100, 12).cuda()
MASK = torch.ones(1, 100).bool().cuda()
