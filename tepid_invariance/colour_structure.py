import argparse
from pathlib import PosixPath, Path

import torch
import yaml
from lie_conv.lieGroups import SE3

from tepid_invariance import label_ligands
from tepid_invariance.models.lie_conv import LieResNet
from tepid_invariance.models.lie_transformer import LieTransformer


def colour_structure(model, rec, lig, out):
    model = model.expanduser()
    rec = rec.expanduser()
    lig = lig.expanduser()
    out = out.expanduser()
    with open(model.parents[1] / 'model_kwargs.yaml', 'r') as f:
        model_kwargs = yaml.load(f, Loader=yaml.Loader)
    with open(model.parents[1] / 'cmd_line_args.yaml', 'r') as f:
        cmd_line_args = yaml.load(f, Loader=yaml.Loader)

    model_type = cmd_line_args['model']
    model_class = {
        'lietransformer': LieTransformer, 'lieconv': LieResNet}[model_type]
    model_obj = model_class(Path(), 0, 0, silent=True, **model_kwargs)

    checkpoint = torch.load(model)
    model_obj.load_state_dict(checkpoint['model_state_dict'])
    model_obj.eval()

    model_obj.colour_pdb(rec, lig, out, radius=cmd_line_args['radius'])


def colour_structures(model, recs, output_path):
    model = model.expanduser()
    with open(model.parents[1] / 'model_kwargs.yaml', 'r') as f:
        model_kwargs = yaml.load(f, Loader=yaml.Loader)
    with open(model.parents[1] / 'cmd_line_args.yaml', 'r') as f:
        kwargs = yaml.load(f, Loader=yaml.Loader)
    kwargs.update(model_kwargs)
    model_type = kwargs['model']
    model_class = {
        'lietransformer': LieTransformer, 'lieconv': LieResNet}[model_type]
    atom_filter = kwargs['filter']
    model_obj = model_class(Path(), 0, 0, silent=True, **model_kwargs)

    checkpoint = torch.load(model)
    model_obj.load_state_dict(checkpoint['model_state_dict'])
    model_obj.eval()
    for rec in recs:
        rec = rec.expanduser()
        rec_name = rec.parent.name
        lig = next(Path(
            rec.parents[2], 'ligands', rec_name).expanduser().glob('*.sdf'))

        label_ligands.sdf_to_labelled_pdb(
            lig, Path(output_path, '{0}_{1}_ligands'.format(
                rec_name, atom_filter)).expanduser(), atom_filter)
        rec_output_path = Path(
            output_path, '{}_receptor.pdb'.format(rec_name)).expanduser()

        model_obj.colour_pdb(
            rec, lig, rec_output_path, radius=kwargs['radius'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=PosixPath)
    parser.add_argument('receptor', type=PosixPath)
    parser.add_argument('ligand', type=PosixPath)
    parser.add_argument('output_fname', type=PosixPath)

    args = parser.parse_args()
    s = SE3

    colour_structure(args.model, args.receptor, args.ligand, args.output_fname)
