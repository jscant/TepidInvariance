"""Some helper functions for cutting inputs down to size."""

import matplotlib
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np


def make_box(struct, radius=10):
    """Truncates receptor atoms which are too far away from the ligand.
    Arguments:
        struct: DataFrame containing x, y, z, types, bp series.
        radius: maximum distance from a ligand atom a receptor atom can be
            to avoid being discarded.
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


def plot_struct(struct, threshold=10):
    """Helper function for plotting inputs."""

    def set_axes_equal(ax):
        """Make axes of 3D plot have equal scale so that spheres appear as
        spheres, cubes as cubes, etc.
        Arguments:
          ax: a matplotlib axis
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    s = struct
    xyz = s[s.columns[:3]].to_numpy()
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    dist = struct.dist.to_numpy()
    dist = np.ma.masked_where(dist < threshold, dist, copy=False).mask
    dist = np.array(dist, dtype='int32')

    colours = ['black', 'red']
    ax.scatter(x, y, z, c=dist, cmap=matplotlib.colors.ListedColormap(colours),
               marker='o', s=80)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    set_axes_equal(ax)
    plt.savefig('/home/scantleb-admin/Desktop/point_cloud.png')
    plt.show()
