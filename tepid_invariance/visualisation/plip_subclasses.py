import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser, DSSP
from plip.basic.remote import VisualizerData
from plip.visualization.pymol import PyMOLVisualizer
from pymol import cmd, stored


class VisualizerDataWithMolecularInfo(VisualizerData):
    """VisualizerData but with the mol, ligand and pli objects stored."""

    def __init__(self, mol, site):
        pli = mol.interaction_sets[site]
        self.ligand = pli.ligand
        self.pli = pli
        self.mol = mol
        super().__init__(mol, site)


class PyMOLVisualizerWithBFactorColouring(PyMOLVisualizer):

    def colour_b_factors(self, model, pdb_file, dt, input_dim, chain='',
                         quiet=False, radius=12, rasa=False,
                         use_atomic_numbers=False):

        def atom_data_extract(chain, atom_to_bfactor_map):
            bdat = {}

            for idx, b_factor in atom_to_bfactor_map.items():
                bdat.setdefault(chain, {})[idx] = (b_factor, '')

            return bdat

        df = dt.mol_calculate_interactions(
            self.plcomplex.mol, self.plcomplex.pli)

        all_indices = df['atom_id'].to_numpy()

        centre_coords = find_ligand_centre(self.plcomplex.ligand)
        print(self.plcomplex.uid)
        mean_x, mean_y, mean_z = centre_coords

        df['x'] -= mean_x
        df['y'] -= mean_y
        df['z'] -= mean_z
        df['sq_dist'] = (df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
        radius = 12
        df = df[df.sq_dist < radius ** 2].copy()

        pdbid = self.plcomplex.pdbid
        p = PDBParser(QUIET=True)
        structure = p.get_structure(pdbid, pdb_file)
        dssp = DSSP(structure[0], pdb_file, dssp='mkdssp')
        keys = list(dssp.keys())
        seq_map = {idx: dssp[key][3] for idx, key in enumerate(keys)}
        df['rasa'] = df['sequential_indices'].map(seq_map)

        labelled_indices = df['atom_id'].to_numpy()
        unlabelled_indices = np.setdiff1d(all_indices, labelled_indices)

        p = torch.from_numpy(
            np.expand_dims(df[df.columns[1:4]].to_numpy(), 0).astype('float32'))
        m = torch.from_numpy(np.ones((1, len(df)))).bool()

        if use_atomic_numbers:
            map_dict = {
                6: 0,
                7: 1,
                8: 2,
                16: 3
            }
            df['atomic_categories'] = df.atomic_number.map(
                map_dict).fillna(4).astype(int)
            v = torch.unsqueeze(F.one_hot(torch.as_tensor(
                df.atomic_categories.to_numpy()), input_dim), 0).float()
        else:
            v = torch.unsqueeze(F.one_hot(torch.as_tensor(
                df.types.to_numpy()), input_dim), 0).float()

        if rasa:
            v[..., -1] = torch.as_tensor(df['rasa'].to_numpy()).squeeze()

        #model = model.train()
        model_labels = torch.sigmoid(model((
            p.cuda(), v.cuda(), m.cuda()))).cpu().detach().numpy().squeeze()
        print(model_labels)

        df['probability'] = model_labels
        with pd.option_context('display.max_colwidth', None):
            with pd.option_context('display.max_rows', None):
                with pd.option_context('display.max_columns', None):
                    print(df)

        atom_to_bfactor_map = {
            labelled_indices[i]: model_labels[i] for i in range(len(df))}
        atom_to_bfactor_map.update({
            idx: 0 for idx in unlabelled_indices})

        # change self.protname to ''
        b_factor_labels = atom_data_extract(
            chain, atom_to_bfactor_map)

        def b_lookup(chain, resi, name, ID, b):
            def _lookup(chain, resi, name, ID):
                if resi in b_factor_labels[chain] and isinstance(
                        b_factor_labels[chain][resi],
                        dict):
                    return b_factor_labels[chain][resi][name][0]
                else:
                    # find data by ID
                    return b_factor_labels[chain][int(ID)][0]

            try:
                if chain not in b_factor_labels:
                    chain = ''
                b = _lookup(chain, resi, name, ID)
                if not quiet: print(
                    '///%s/%s/%s new: %f' % (chain, resi, name, b))
            except KeyError:
                if not quiet: print(
                    '///%s/%s/%s keeping: %f' % (chain, resi, name, b))
            return b

        stored.b = b_lookup

        cmd.alter(self.protname, '%s=stored.b(chain, resi, name, ID, %s)' % (
            'b', 'b'))
        print(self.ligname)
        cmd.spectrum('b', 'white_red', 'not ({})'.format(self.ligname))
        cmd.show('sticks', 'b > 0 and not ({})'.format(self.ligname))
        cmd.rebuild()


def find_ligand_centre(ligand):
    positions = []
    for atom in ligand.molecule.atoms:
        positions.append(atom.coords)
    return np.mean(np.array(positions), axis=0)
