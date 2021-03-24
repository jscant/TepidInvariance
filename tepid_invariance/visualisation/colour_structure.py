import argparse
from pathlib import Path

import torch
import yaml
from lie_conv.lieConv import LieResNet
from lie_conv.lieGroups import SE3
from plip.basic import config
from plip.basic.supplemental import create_folder_if_not_exists, start_pymol
from plip.structure.preparation import PDBComplex
from plipcmd import logger
from pymol import cmd

from tepid_invariance.models.lie_transformer import LieTransformer
from tepid_invariance.preprocessing.pdb_to_parquet import DistanceCalculator
from tepid_invariance.visualisation.plip_subclasses import \
    PyMOLVisualizerWithBFactorColouring, VisualizerDataWithMolecularInfo


def visualize_in_pymol(model, plcomplex, radius=12):
    """Visualizes the given Protein-Ligand complex at one site in PyMOL.

    This function is based on the origina plip.visualization.vizualise_in_pymol
    function, with the added functinoality which uses a machine learning model
    to colour the receptor atoms by
    """

    vis = PyMOLVisualizerWithBFactorColouring(plcomplex)

    #####################
    # Set everything up #
    #####################

    pdbid = plcomplex.pdbid
    lig_members = plcomplex.lig_members
    chain = plcomplex.chain
    if config.PEPTIDES:
        vis.ligname = 'PeptideChain%s' % plcomplex.chain
    if config.INTRA is not None:
        vis.ligname = 'Intra%s' % plcomplex.chain

    ligname = vis.ligname
    hetid = plcomplex.hetid

    metal_ids = plcomplex.metal_ids
    metal_ids_str = '+'.join([str(i) for i in metal_ids])

    ########################
    # Basic visualizations #
    ########################

    start_pymol(run=True, options='-pcq',
                quiet=not config.VERBOSE and not config.SILENT)
    vis.set_initial_representations()

    cmd.load(plcomplex.sourcefile)
    current_name = cmd.get_object_list(selection='(all)')[0]

    logger.debug(
        f'setting current_name to {current_name} and PDB-ID to {pdbid}')
    cmd.set_name(current_name, pdbid)
    cmd.hide('everything', 'all')
    if config.PEPTIDES:
        cmd.select(ligname, 'chain %s and not resn HOH' % plcomplex.chain)
    else:
        cmd.select(ligname, 'resn %s and chain %s and resi %s*' % (
            hetid, chain, plcomplex.position))
    logger.debug(
        f'selecting ligand for PDBID {pdbid} and ligand name {ligname}')
    logger.debug(
        f'resn {hetid} and chain {chain} and resi {plcomplex.position}')

    # Visualize and color metal ions if there are any
    if not len(metal_ids) == 0:
        vis.select_by_ids(ligname, metal_ids, selection_exists=True)
        cmd.show('spheres', 'id %s and %s' % (metal_ids_str, pdbid))

    # Additionally, select all members of composite ligands
    if len(lig_members) > 1:
        for member in lig_members:
            resid, chain, resnr = member[0], member[1], str(member[2])
            cmd.select(ligname, '%s or (resn %s and chain %s and resi %s)' % (
                ligname, resid, chain, resnr))

    cmd.show('sticks', ligname)
    cmd.color('myblue')
    cmd.color('myorange', ligname)
    cmd.util.cnc('all')
    if not len(metal_ids) == 0:
        cmd.color('hotpink', 'id %s' % metal_ids_str)
        cmd.hide('sticks', 'id %s' % metal_ids_str)
        cmd.set('sphere_scale', 0.3, ligname)
    cmd.deselect()

    vis.make_initial_selections()

    vis.show_hydrophobic()  # Hydrophobic Contacts
    vis.show_hbonds()  # Hydrogen Bonds
    vis.show_halogen()  # Halogen Bonds
    vis.show_stacking()  # pi-Stacking Interactions
    vis.show_cationpi()  # pi-Cation Interactions
    vis.show_sbridges()  # Salt Bridges
    vis.show_wbridges()  # Water Bridges
    vis.show_metal()  # Metal Coordination

    dt = DistanceCalculator()
    vis.colour_b_factors(model, dt, '', True, radius=radius)

    vis.refinements()

    vis.zoom_to_ligand()

    vis.selections_cleanup()

    vis.selections_group()
    vis.additional_cleanup()
    if config.DNARECEPTOR:
        # Rename Cartoon selection to Line selection and change repr.
        cmd.set_name('%sCartoon' % plcomplex.pdbid, '%sLines' % plcomplex.pdbid)
        cmd.hide('cartoon', '%sLines' % plcomplex.pdbid)
        cmd.show('lines', '%sLines' % plcomplex.pdbid)

    if config.PEPTIDES:
        filename = "%s_PeptideChain%s" % (pdbid.upper(), plcomplex.chain)
        if config.PYMOL:
            vis.save_session(config.OUTPATH, override=filename)
    elif config.INTRA is not None:
        filename = "%s_IntraChain%s" % (pdbid.upper(), plcomplex.chain)
        if config.PYMOL:
            vis.save_session(config.OUTPATH, override=filename)
    else:
        filename = '%s_%s' % (
            pdbid.upper(),
            "_".join([hetid, plcomplex.chain, plcomplex.position]))
        print(filename)
        vis.save_session(config.OUTPATH)
    if config.PICS:
        vis.save_picture(config.OUTPATH, filename)


def process_pdb(model, pdbfile, outpath, radius=12):
    mol = PDBComplex()
    mol.output_path = outpath
    mol.load_pdb(pdbfile, as_string=False)

    for ligand in mol.ligands:
        mol.characterize_complex(ligand)

    create_folder_if_not_exists(outpath)

    complexes = [VisualizerDataWithMolecularInfo(mol, site) for site in sorted(
        mol.interaction_sets)
                 if not len(mol.interaction_sets[site].interacting_res) == 0]

    [visualize_in_pymol(model, plcomplex, radius=radius) for plcomplex in
     complexes]


def colour_structure(model, rec, out):
    model = model.expanduser()
    rec = rec.expanduser()
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

    process_pdb(model_obj, str(rec), str(out), radius=cmd_line_args['radius'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('receptor', type=str)
    parser.add_argument('output_fname', type=str)

    args = parser.parse_args()
    s = SE3

    colour_structure(
        Path(args.model), Path(args.receptor), Path(args.output_fname))
