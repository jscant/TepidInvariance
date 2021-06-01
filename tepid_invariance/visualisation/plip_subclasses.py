import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser, DSSP
from plip.basic.remote import VisualizerData
from plip.visualization.pymol import PyMOLVisualizer
from pymol import cmd

from tepid_invariance.utils import truncate_float, coords_to_string


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

        def change_bfactors(bfactors):
            """Modify bfactors based on spatial location.

            Due to inconsistencies in the indexing of atoms, residues and chains
            between openbabel, plip and pymol, we must use coordinates to
            identify atoms.

            Arguments:
                bfactors: dict of dict of dicts with the mapping:
                    x: y: z: value
                where value is the number we wish to assign to the atom (as the
                b-factor) for labelling. The x, y and z coordinates should be
                in the format of strings, with 1 decimal place to avoid
                problems with comparing floats (use '{:.1f}.format(<coord>).
            """

            def modify_bfactor(x, y, z):
                """Return b factor given the x, y, z coordinates."""
                x, y, z = ['{:.3f}'.format(coord) for coord in (x, y, z)]
                bfactor = bfactors[x][y][z]
                return bfactor

            space = {'modify_bfactor': modify_bfactor}
            cmd.alter_state(
                0, '(all)', 'b=modify_bfactor(x, y, z)', space=space,
                quiet=True)

        df = dt.mol_calculate_interactions(
            self.plcomplex.mol, self.plcomplex.pli)

        all_indices = df['atom_id'].to_numpy()

        centre_coords = find_ligand_centre(self.plcomplex.ligand)
        print(self.plcomplex.uid)
        mean_x, mean_y, mean_z = [truncate_float(i) for i in centre_coords]

        df['x'] -= mean_x
        df['y'] -= mean_y
        df['z'] -= mean_z
        df['sq_dist'] = (df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
        df = df[df.sq_dist < radius ** 2].copy()
        print(df)
        df['x'] += mean_x
        df['y'] += mean_y
        df['z'] += mean_z
        del df['sq_dist']

        x, y, z = df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()

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

        model = model.eval()
        model_labels = torch.sigmoid(model((
            p.cuda(), v.cuda(), m.cuda()))).cpu().detach().numpy().squeeze()
        print(model_labels)

        df['probability'] = model_labels
        with pd.option_context('display.max_colwidth', None):
            with pd.option_context('display.max_rows', None):
                with pd.option_context('display.max_columns', None):
                    print(df)

        atom_to_bfactor_map = {}
        for i in range(len(df)):
            bfactor = float(model_labels[i])
            atom_to_bfactor_map[coords_to_string((x[i], y[i], z[i]))] = bfactor

        cmd.alter('all', 'b=0')
        change_bfactors(atom_to_bfactor_map)
        print(self.ligname)
        cmd.spectrum('b', 'white_red', 'not ({})'.format(self.ligname),
                     minimum=0, maximum=1)
        cmd.show('sticks', 'b > 0')
        cmd.rebuild()


def find_ligand_centre(ligand):
    positions = []
    for atom in ligand.molecule.atoms:
        positions.append(atom.coords)
    return np.mean(np.array(positions), axis=0)
