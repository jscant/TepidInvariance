import random

import numpy as np
import torch
from lie_conv.lieGroups import SE3

from lie_transformer.lie_transformer import LieTepid
from parse_args import parse_args
from preprocessing.data_loaders import LieTransformerLabelledAtomsDataset

if __name__ == '__main__':
    args = parse_args()

    model_kwargs = {
        'dim_input': 12,
        'dim_hidden': args.channels,
        'num_layers': args.layers,
        'num_heads': 8,
        'global_pool': True,
        'global_pool_mean': True,
        'group': SE3(0.2),
        'liftsamples': args.liftsamples,
        'block_norm': "layer_pre",
        'output_norm': "none",
        'kernel_norm': "none",
        'kernel_type': args.kernel_type,
        'kernel_dim': args.kernel_dim,
        'kernel_act': args.activation,
        'mc_samples': args.nbhd,
        'fill': 1.0,
        'attention_fn': args.attention_fn,
        'feature_embed_dim': None,
        'max_sample_norm': None,
        'lie_algebra_nonlinearity': None,
        'dropout': args.dropout
    }
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    ds = LieTransformerLabelledAtomsDataset(
        args.train_data_root, binary_threshold=args.binary_threshold)
    dl = torch.utils.data.DataLoader(
        ds, shuffle=True, batch_size=args.batch_size, collate_fn=ds.collate)
    mode = 'regression' if args.binary_threshold is None else 'classification'
    model = LieTepid(args.save_path, 0.001, 1e-6, mode=mode, **model_kwargs)
    model.optimise(dl, 1)
