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
from torch.utils.data import SubsetRandomSampler

from preprocessing.preprocessing import make_bit_vector


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

    def __init__(self, base_path, rot=False, binary_threshold=None, **kwargs):
        """Initialise dataset.
        Arguments:
            base_path: path containing the 'receptors' and 'ligands'
                directories, which in turn contain <rec_name>.parquets files
                and folders called <rec_name>_[active|decoy] which in turn
                contain <ligand_name>.parquets files. All parquets files from
                this directory are recursively loaded into the dataset.
            radius: size of the bounding box; all atoms further than <radius>
                Angstroms from the mean ligand atom position are discarded.
            receptors: iterable of strings denoting receptors to include in
                the dataset. if None, all receptors found in base_path are
                included.
            kwargs: keyword arguments passed to the parent class (Dataset).
        """

        super().__init__(**kwargs)
        self.binary_threshold = binary_threshold
        self.rot = random_rotation if rot else lambda x: x

        self.base_path = Path(base_path).expanduser()
        self.filenames = list(self.base_path.glob('**/*.parquet'))

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
        struct = pd.read_parquet(filename)

        p = torch.from_numpy(np.expand_dims(self.rot(
            struct[struct.columns[:3]].to_numpy()), 0))
        v = torch.unsqueeze(make_bit_vector(struct.types.to_numpy(), 11), 0)
        m = torch.from_numpy(np.ones((1, len(struct))))
        if self.binary_threshold is None:
            dist = torch.from_numpy(struct.dist.to_numpy())
        else:
            dist = struct.dist.to_numpy()
            dist = np.ma.masked_where(
                dist < self.binary_threshold, dist, copy=False).mask
            dist = np.array(dist, dtype='bool')
            dist = torch.from_numpy(dist)

        return p, v, m, dist, filename, len(struct)

    @staticmethod
    def collate(batch):
        """Processing of inputs which takes place after batch is selected.
        LieConv networks take tuples of torch tensors (p, v, m), which are:
            p, (batch_size, n_atoms, 3): coordinates of each atom
            v, (batch_size, n_atoms, n_features): features for each atom
            m, (batch_size, n_atoms): mask for each coordinate slot
        Note that n_atoms is the largest number of atoms in a structure in
        each batch.
        Arguments:
            batch: iterable of individual inputs.
        Returns:
            Tuple of feature vectors ready for input into a LieConv network.
        """
        max_len = max([b[-1] for b in batch])
        batch_size = len(batch)
        p_batch = torch.zeros(batch_size, max_len, 3)
        v_batch = torch.zeros(batch_size, max_len, 12)
        m_batch = torch.zeros(batch_size, max_len)
        label_batch = torch.zeros(batch_size, max_len)
        filenames = []
        for batch_index, (p, v, m, dist, filename, _) in enumerate(
                batch):
            p_batch[batch_index, :p.shape[1], :] = p
            v_batch[batch_index, :v.shape[1], :] = v
            m_batch[batch_index, :m.shape[1]] = m
            label_batch[batch_index, :label_batch.shape[1]] = dist
            filenames.append(filename)
        return (p_batch, v_batch, m_batch.bool()), label_batch, filenames
