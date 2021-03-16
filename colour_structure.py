import argparse
from pathlib import PosixPath, Path

import torch
import yaml
from lie_conv.lieGroups import SE3

from models.lie_conv import LieResNet
from models.lie_transformer import LieTransformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=PosixPath)
    parser.add_argument('receptor', type=PosixPath)
    parser.add_argument('ligand', type=PosixPath)
    parser.add_argument('output_fname', type=PosixPath)

    args = parser.parse_args()
    s = SE3

    with open(args.model.parents[1] / 'model_kwargs.yaml', 'r') as f:
        model_kwargs = yaml.load(f, Loader=yaml.Loader)
    with open(args.model.parents[1] / 'cmd_line_args.yaml', 'r') as f:
        cmd_line_args = yaml.load(f, Loader=yaml.Loader)
    print(model_kwargs)
    print(cmd_line_args)

    if cmd_line_args['binary_threshold'] is None:
        mode = 'regression'
    else:
        mode = 'classification'
    model = cmd_line_args['model']
    model_class = {
        'lietransformer': LieTransformer, 'lieconv': LieResNet}[model]
    model_obj = model_class(Path(), 0, 0, **model_kwargs)

    checkpoint = torch.load(args.model.expanduser())
    model_obj.load_state_dict(checkpoint['model_state_dict'])
    model_obj.eval()

    model_obj.colour_pdb(
        args.receptor.expanduser(), args.ligand.expanduser(),
        args.output_fname.expanduser(), radius=cmd_line_args['radius'])
