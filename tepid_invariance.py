from lie_conv.lieGroups import SE3

from lie_transformer.lie_transformer import LieTepid
from parse_args import parse_args
import torch
import numpy as np

if __name__ == '__main__':
    args = parse_args()

    model_kwargs = {
        'dim_input': 12,
        'dim_output': 2,
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

    model = LieTepid(args.save_path, args.learning_rate, args.weight_decay,
                     args.wandb_project, args.wandb_run, **model_kwargs)
    coords = torch.randn(32, 20, 3).cuda()
    feats = torch.randn(32, 20, 12).cuda()
    mask = torch.ones(32, 20).byte().cuda()
    print(model((coords, feats, mask)))
    import pypdb