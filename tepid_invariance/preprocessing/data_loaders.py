"""
The dataloader for SE(3)Transformer is heavily edited from
a script developed for similar reasons by Constantin Schneider
github.com/con-schneider
The dataloader for LieConv is my own work.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import SubsetRandomSampler


def random_rotation(x):
    M = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(M)
    return x @ Q


def one_hot(numerical_category, num_classes):
    """Make one-hot vector from category and total categories."""
    one_hot_array = np.zeros((len(numerical_category), num_classes))

    for i, cat in enumerate(numerical_category):
        one_hot_array[i, int(cat)] = 1

    return one_hot_array


class LieTransformerLabelledAtomsDataset(torch.utils.data.Dataset):
    """Class for feeding structure parquets into network."""

    def __init__(self, base_path, atom_filter, rot=True, max_suffix=np.inf,
                 radius=12, receptors=None, **kwargs):
        """Initialise dataset.
        Arguments:
            base_path: path containing the 'receptors' and 'ligands'
                directories, which in turn contain <rec_name>.parquets files
                and folders called <rec_name>_[active|decoy] which in turn
                contain <ligand_name>.parquets files. All parquets files from
                this directory are recursively loaded into the dataset.
            atom_filter: one of 'pi_stacking', 'hydrophobic', 'hba', or 'hbd';
                the type of interaction to predict for.
            rot: whether or not to randomly rotate the inputs.
            max_suffix: if the filenames end in *_<x> where <x> is some integer,
                ignore files where x > max_suffix.
            radius: size of the bounding box; all atoms further than <radius>
                Angstroms from the mean ligand atom position are discarded.
            receptors: iterable of strings denoting receptors to include in
                the dataset. if None, all receptors found in base_path are
                included.
            kwargs: keyword arguments passed to the parent class (Dataset).
        """

        super().__init__(**kwargs)
        self.filter = atom_filter
        self.rot = random_rotation if rot else lambda x: x
        self.radius = radius

        self.base_path = Path(base_path).expanduser()
        self.filenames = [f for f in self.base_path.glob('**/*.parquet')
                          if max_suffix == np.inf or
                          int(Path(f.name).stem.split('_')[-1]) <= max_suffix]

        receptors = set()
        ligand_coordinate_info = {}
        for filename in self.filenames:
            receptors.add(filename.parent)
        for receptor in receptors:
            ligand_coordinate_statistics_filename = Path(
                receptor, 'ligand_centres.yaml')
            try:
                with open(ligand_coordinate_statistics_filename, 'r') as f:
                    ligand_coordinate_info[receptor.name] = yaml.load(
                        f, yaml.FullLoader)
            except FileNotFoundError:
                print(
                    '{0} not found. Will use entire protein for receptor {1} ('
                    'no bounding box/truncation'
                    ')'.format(ligand_coordinate_statistics_filename, receptor))
        self.ligand_coordinate_info = ligand_coordinate_info

    def centre_and_truncate(self, filename):
        """Centre atomic coordinates on mean ligand atom position and truncate.

        Coordinates are centred on the mean atom position, calculated when the
        dataset was generated by distance_calculator.py. Any atoms which are
        then further than <radius> Angstroms from the origin are discarded.

        Arguments:
            filename: name of parquet file containing coordinate, atom type and
                minimum distance from ligand atom of interest information

        Returns:
            pandas.Dataframe object containing the x, y and z coordinates of
            each atom, as well as their smina-style type (1-12) and the minimum
            distance from the atoms to a ligand atom of interest (aromatic for
            now).
        """
        rec_name = filename.parent.name
        lig_name = Path(filename.name).stem
        struct = pd.read_parquet(filename)
        mean_coords = self.ligand_coordinate_info[rec_name][lig_name]
        mean_x, mean_y, mean_z = mean_coords
        struct['x'] -= mean_x
        struct['y'] -= mean_y
        struct['z'] -= mean_z
        struct['sq_dist'] = (struct['x'] ** 2 +
                             struct['y'] ** 2 +
                             struct['z'] ** 2)
        struct['atom_idx'] = np.arange(len(struct))

        struct = struct[struct.sq_dist < self.radius ** 2].copy()
        return struct

    def __len__(self):
        """Returns the total size of the dataset."""
        return len(self.filenames)

    def __getitem__(self, item):
        """Given an index, locate and preprocess relevant parquets files.
        Arguments:
            item: index in the list of filenames denoting which ligand and
                receptor to fetch
        Returns:
            Tuple containing (a) a tuple with a list of tensors: cartesian
            coordinates, feature vectors and masks for each point, as well as
            the number of points in the structure and (b) the label \in \{0, 1\}
            denoting whether the structure is an active or a decoy.
        """

        filename = self.filenames[item]

        struct = self.centre_and_truncate(filename)
        p = torch.from_numpy(np.expand_dims(self.rot(
            struct[struct.columns[:3]].to_numpy()), 0))
        v = torch.unsqueeze(F.one_hot(
            torch.as_tensor(struct.types.to_numpy()), 11), 0)
        m = torch.from_numpy(np.ones((1, len(struct))))

        dist = torch.from_numpy(struct[self.filter].to_numpy())
        atomic_numbers = torch.from_numpy(struct['types'].to_numpy())

        return p, v, m, dist, atomic_numbers, len(struct)

    @staticmethod
    def collate(batch):
        """Processing of inputs which takes place after batch is selected.

        LieConv networks take tuples of torch tensors (p, v, m), which are:
            p, (batch_size, n_atoms, 3): coordinates of each atom
            v, (batch_size, n_atoms, n_features): features for each atom
            m, (batch_size, n_atoms): mask for each coordinate slot
        Note that n_atoms is the largest number of atoms in a structure in
        each batch. LieTransformer networks take the first two of these (p, v).

        Arguments:
            batch: iterable of individual inputs.
        Returns:
            Tuple of feature vectors ready for input into a LieConv network.
        """
        max_len = max([b[-1] for b in batch])
        batch_size = len(batch)
        p_batch = torch.zeros(batch_size, max_len, 3)
        v_batch = torch.zeros(batch_size, max_len, 11)
        m_batch = torch.zeros(batch_size, max_len)
        label_batch = torch.zeros(batch_size, max_len)
        atomic_numbers = torch.zeros_like(m_batch).long()
        for batch_index, (p, v, m, dist, atomics, _) in enumerate(
                batch):
            p_batch[batch_index, :p.shape[1], :] = p
            v_batch[batch_index, :v.shape[1], :] = v
            m_batch[batch_index, :m.shape[1]] = m
            atomic_numbers[batch_index, :len(atomics)] = atomics
            try:
                label_batch[batch_index, :dist.shape[0]] = dist
            except IndexError:  # no positive labels
                pass
        return (p_batch, v_batch, m_batch.bool()), label_batch, atomic_numbers
