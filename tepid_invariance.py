import random

import numpy as np
import torch
import wandb
import yaml
from lie_conv.lieGroups import SE3

from tepid_invariance.models.egnn import EnEquivariant
from tepid_invariance.models.lie_conv import LieResNet
from tepid_invariance.models.lie_transformer import LieTransformer
from tepid_invariance.parse_args import parse_args
from tepid_invariance.preprocessing.data_loaders import \
    LieTransformerLabelledAtomsDataset, get_lt_collate_fn

if __name__ == '__main__':
    args = parse_args()
    args.save_path.expanduser().mkdir(parents=True, exist_ok=True)

    with open(args.save_path.expanduser() / 'cmd_line_args.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    dim_input = 5 if args.use_atomic_numbers else 11
    if args.use_rasa:
        dim_input += 1

    model_kwargs = {
        'dim_input': dim_input,
        'dim_hidden': args.channels,
        'num_layers': args.layers,
        'num_heads': 8,
        'group': SE3(0.2),
        'liftsamples': args.liftsamples,
        'block_norm': "layer_pre",
        'kernel_norm': "none",
        'kernel_type': args.kernel_type,
        'kernel_dim': args.kernel_dim,
        'kernel_act': args.activation,
        'mc_samples': args.nbhd,
        'fill': 1.0,
        'attention_fn': args.attention_fn,
        'feature_embed_dim': None,
        'max_sample_norm': None,
        'lie_algebra_nonlinearity': args.lie_algebra_nonlinearity,
        'dropout': args.dropout,
    }

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    ds = LieTransformerLabelledAtomsDataset(
        args.train_data_root, radius=args.radius, feature_dim=dim_input,
        use_rasa=args.use_rasa, use_atomic_numbers=args.use_atomic_numbers,
        atom_filter=args.filter)
    dl = torch.utils.data.DataLoader(
        ds, shuffle=True, batch_size=args.batch_size,
        collate_fn=get_lt_collate_fn(dim_input), drop_last=True)

    mode = 'classification'
    if args.model == 'lieconv':
        model = LieResNet(
            args.save_path, args.learning_rate, args.weight_decay, mode=mode,
            weighted_loss=args.weighted_loss,
            **model_kwargs)
    elif args.model == 'lietransformer':
        model = LieTransformer(
            args.save_path, args.learning_rate, args.weight_decay, mode=mode,
            weighted_loss=args.weighted_loss,
            **model_kwargs)
    elif args.model == 'egnn':
        model = EnEquivariant(
            args.save_path, args.learning_rate, args.weight_decay, mode=mode,
            weighted_loss=args.weighted_loss,
            **model_kwargs)
    else:
        raise NotImplementedError(
            'supplied model arg must be either lieconv or lietransformer')
    if args.wandb_project is not None:
        args_to_record = vars(args)
        args_to_record.update(model_kwargs)
        wandb_init_kwargs = {
            'project': args.wandb_project, 'allow_val_change': True,
            'config': args_to_record
        }
        wandb.init(**wandb_init_kwargs)
        if args.wandb_run is not None:
            wandb.run.name = args.wandb_run
        wandb.watch(model)
    model.optimise(dl, args.epochs)
