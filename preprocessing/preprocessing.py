"""Some helper functions for cutting inputs down to size."""

import torch
import torch.nn.functional as F


def make_box(struct, radius=10):
    """Truncates receptor atoms which are too far away from the ligand.
    Arguments:
        struct: DataFrame containing x, y, z, types, bp series.
        radius: maximum distance from a ligand atom a receptor atom can be
            to avoid being discarded.
        relative_to_ligand: if True, radius means minimum distance to closest
            ligand atom; if False, radius means distance to centre of ligand
    Returns:
        DataFrame of the same format as the input <struct>, with all ligand
        atoms and receptor atoms that are within <radius> angstroms of any
        ligand atom.
    """
    struct['sq_dist'] = struct['x'] ** 2 + struct['y'] ** 2 + struct['z'] ** 2

    struct = struct[struct.sq_dist < radius ** 2].copy()
    return struct


def make_bit_vector(atom_types, n_atom_types):
    """Make one-hot bit vector from indices, with switch for structure type.
    Arguments:
        atom_types: ids for each type of atom
        n_atom_types: number of different atom types in dataset
    Returns:
        One-hot bit vector (torch tensor) of atom ids, including leftmost bit
        which indicates the structure (ligand == 0, receptor == 1).
    """
    indices = torch.from_numpy(atom_types % n_atom_types).long()
    one_hot = F.one_hot(indices, num_classes=n_atom_types)
    rows, cols = one_hot.shape
    result = torch.zeros(rows, cols + 1)
    result[:, 1:] = one_hot
    type_bit = torch.from_numpy(
        (atom_types // n_atom_types).astype('bool').astype('int'))
    result[:, 0] = type_bit
    return result
