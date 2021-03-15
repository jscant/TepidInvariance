"""Set up rather large command line argument list"""

import argparse
from pathlib import PosixPath

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Type of point cloud network to use '
                             '(LieTransformer only for now)')
    parser.add_argument('train_data_root', type=PosixPath,
                        help='Location of structure training *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('save_path', type=PosixPath,
                        help='Directory in which experiment outputs are '
                             'stored.')
    parser.add_argument('--load_weights', '-l', type=PosixPath, required=False,
                        help='Load a model.')
    parser.add_argument('--test_data_root', '-t', type=PosixPath,
                        required=False,
                        help='Location of structure test *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('--translated_actives', type=PosixPath,
                        help='Directory in which translated actives are stored.'
                             ' If unspecified, no translated actives will be '
                             'used. The use of translated actives are is '
                             'discussed in https://pubs.acs.org/doi/10.1021/ac'
                             's.jcim.0c00263')
    parser.add_argument('--batch_size', '-b', type=int, required=False,
                        default=32,
                        help='Number of examples to include in each batch for '
                             'training.')
    parser.add_argument('--epochs', '-e', type=int, required=False,
                        default=1,
                        help='Number of times to iterate through training set.')
    parser.add_argument('--channels', '-k', type=int, default=32,
                        help='Channels for feature vectors')
    parser.add_argument('--train_receptors', '-r', type=str, nargs='*',
                        help='Names of specific receptors for training. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--test_receptors', '-q', type=str, nargs='*',
                        help='Names of specific receptors for testing. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.002,
                        help='Learning rate for gradient descent')
    parser.add_argument('--weight_decay', '-w', type=float, default=1e-4,
                        help='Weight decay for regularisation')
    parser.add_argument('--wandb_project', type=str,
                        help='Name of wandb project. If left blank, wandb '
                             'logging will not be used.')
    parser.add_argument('--wandb_run', type=str,
                        help='Name of run for wandb logging.')
    parser.add_argument('--layers', type=int, default=6,
                        help='Number of group-invariant layers')
    parser.add_argument('--channels_in', '-chin', type=int, default=12,
                        help='Input channels')
    parser.add_argument('--liftsamples', type=int, default=1,
                        help='liftsamples parameter in LieConv')
    parser.add_argument('--radius', type=int, default=6,
                        help='Maximum distance from a ligand atom for a '
                             'receptor atom to be included in input')
    parser.add_argument('--nbhd', type=int, default=32,
                        help='Number of monte carlo samples for integral')
    parser.add_argument('--load_args', type=PosixPath,
                        help='Load yaml file with command line args. Any args '
                             'specified in the file will overwrite other args '
                             'specified on the command line.')
    parser.add_argument('--double', action='store_true',
                        help='Use 64-bit floating point precision')
    parser.add_argument('--kernel_type', type=str, default='mlp',
                        help='One of 2232, mlp, overrides attention_fn '
                             '(see original repo) (LieTransformer)')
    parser.add_argument('--attention_fn', type=str, default='dot_product',
                        help='One of norm_exp, softmax, dot_product: '
                             'activation for attention (overridden by '
                             'kernel_type) (LieTransformer)')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function')
    parser.add_argument('--kernel_dim', type=int, default=16,
                        help='Size of linear layers in attention kernel '
                             '(LieTransformer)')
    parser.add_argument('--feature_embed_dim', type=int, default=None,
                        help='Feature embedding dimension for attention; '
                             'paper had dv=848 for QM9 (LieTransformer)')
    parser.add_argument('--mc_samples', type=int, default=0,
                        help='Monte carlo samples for attention '
                             '(LieTransformer)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Chance for nodes to be inactivated on each '
                             'trainin batch (LieTransformer)')
    parser.add_argument('--binary_threshold', type=float,
                        help='Threshold for distance in binary classification. '
                             'If unspecified, the distance value will be used '
                             'for regression instead.')
    parser.add_argument('--max_suffix', type=int, default=np.inf,
                        help='Maximum integer at end of filename: for example, '
                             'CHEMBL123456_4.parquet would be included with '
                             '<--max_suffix 4> but not <--max_suffix 3>.')
    parser.add_argument('--inverse', action='store_true',
                        help='Regression is performed on the inverse distance.')
    return parser.parse_args()
