import numpy as np
import torch
import torch.nn.functional as F
from plip.basic.remote import VisualizerData
from plip.visualization.pymol import PyMOLVisualizer
from pymol import cmd, stored

import pandas as pd
class VisualizerDataWithMolecularInfo(VisualizerData):
    """VisualizerData but with the mol, ligand and pli objects stored."""

    def __init__(self, mol, site):
        pli = mol.interaction_sets[site]
        self.ligand = pli.ligand
        self.pli = pli
        self.mol = mol
        super().__init__(mol, site)


class PyMOLVisualizerWithBFactorColouring(PyMOLVisualizer):

    def colour_b_factors(self, model, dt, chain='', quiet=False, radius=12):

        def atom_data_extract(chain, atom_to_bfactor_map):
            bdat = {}

            for idx, b_factor in atom_to_bfactor_map.items():
                bdat.setdefault(chain, {})[idx] = (b_factor, '')

            return bdat

        df = dt.mol_calculate_interactions(
            self.plcomplex.mol, self.plcomplex.pli)

        all_indices = df['atom_id'].to_numpy()

        centre_coords = find_ligand_centre(self.plcomplex.ligand)
        print(self.plcomplex.uid, centre_coords)
        mean_x, mean_y, mean_z = centre_coords

        df['x'] -= mean_x
        df['y'] -= mean_y
        df['z'] -= mean_z
        df['sq_dist'] = (df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
        df = df[df.sq_dist < radius ** 2].copy()
        del df['sq_dist']

        labelled_indices = df['atom_id'].to_numpy()
        unlabelled_indices = np.setdiff1d(all_indices, labelled_indices)

        p = torch.from_numpy(
            np.expand_dims(df[df.columns[1:4]].to_numpy(), 0).astype('float32'))
        v = torch.unsqueeze(F.one_hot(
            torch.as_tensor(df.types.to_numpy()), 11), 0).float()
        m = torch.from_numpy(np.ones((1, len(df)))).bool()

        model_labels = torch.sigmoid(
            model((p.cuda(),
                   v.cuda(),
                   m.cuda()))).cpu().detach().numpy()[0, :].squeeze()

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
                if not chain in b_factor_labels:
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
