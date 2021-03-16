import time
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path

import Bio.PDB as PDB
import numpy as np
import pandas as pd
import pybel
import torch
import torch.nn.functional as F
import wandb
import yaml
from einops import repeat
from rdkit import Chem, RDLogger
from torch import nn

from preprocessing.distance_calculator import DistanceCalculator, \
    get_centre_coordinates
from utils import get_eta, format_time, print_with_overwrite


class PointNeuralNetwork(nn.Module):
    """Base class for node classification and regression tasks."""

    def __init__(self, save_path, learning_rate, weight_decay=None,
                 mode='classification', wandb_project=None, wandb_run=None,
                 **model_kwargs):
        super().__init__()
        self.batch = 0
        self.epoch = 0
        self.losses = []
        self.save_path = Path(save_path).expanduser()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.save_path / 'predictions.txt'
        self.mode = mode

        self.loss_plot_file = self.save_path / 'loss.png'

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.translated_actives = model_kwargs.get('translated_actives', None)
        self.n_translated_actives = model_kwargs.get('n_translated_actives', 0)

        self.loss_log_file = self.save_path / 'loss.log'

        if mode == 'classification':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.MSELoss()

        self.wandb_project = wandb_project
        self.wandb_path = self.save_path / 'wandb_{}'.format(wandb_project)
        self.wandb_run = wandb_run

        self.build_net(**model_kwargs)
        self.optimiser = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=weight_decay)

        with open(save_path / 'model_kwargs.yaml', 'w') as f:
            yaml.dump(model_kwargs, f)

        self.apply(self.xavier_init)
        self.cuda()

    @abstractmethod
    def build_net(self, **model_kwargs):
        raise NotImplementedError(
            'Classes inheriting from the virtual class PointNeuralNetwork must '
            'implement the build_net method.')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _get_y_true(self, y):
        """Preprocessing for getting the true node labels."""
        return y.cuda()

    def _process_inputs(self, x):
        """Preprocessing for getting the inputs."""
        return x.cuda()

    def _get_loss(self, y_true, y_pred):
        """Process the loss."""
        return self.loss(y_pred, y_true)

    def optimise(self, data_loader, epochs=1):
        """Train the network.

        Trains the neural network. Displays training information and plots the
        loss. All figures and logs are saved to save_path.

        Arguments:
            data_loader: pytorch DataLoader object for training
            epochs: number of complete training cycles
        """
        start_time = time.time()
        total_iters = epochs * len(data_loader)
        log_interval = 10
        global_iter = 0
        self.train()
        print()
        print()
        if data_loader.batch_size == 1:
            aggrigation_interval = 32
        else:
            aggrigation_interval = 1
        if self.mode == 'classification':
            loss_type = 'Binary crossentropy'
        else:
            loss_type = 'MSE'
        for self.epoch in range(epochs):
            loss = torch.FloatTensor([0.0]).cuda()
            mean_positive_prediction, mean_negative_prediction = 0., 0.
            n_positive, n_negative = 0, 0
            for self.batch, (x, y_true, filenames) in enumerate(
                    data_loader):

                x = self._process_inputs(x)
                y_true = self._get_y_true(y_true).cuda()
                y_pred = self(x).cuda()

                y_pred_np = torch.sigmoid(y_pred).cpu().detach().numpy()
                y_true_np = y_true.cpu().detach().numpy()
                positive_indices = np.where(y_true_np > 0.5)
                negative_indices = np.where(y_true_np < 0.5)
                n_positive += len(positive_indices[0])
                n_negative += len(negative_indices[0])
                mean_positive_prediction += np.mean(y_pred_np[positive_indices])
                mean_negative_prediction += np.mean(y_pred_np[negative_indices])

                loss += self._get_loss(y_true, y_pred)

                if not (self.batch + 1) % aggrigation_interval:
                    self.optimiser.zero_grad()
                    loss /= aggrigation_interval
                    reported_batch = (self.batch + 1) // aggrigation_interval  #
                    eta = get_eta(start_time, global_iter, total_iters)
                    time_elapsed = format_time(time.time() - start_time)
                    wandb_update_dict = {
                        'Time remaining (train)': eta,
                        '{} (train)'.format(loss_type): float(loss),
                        'Batch (train)':
                            (self.epoch * len(data_loader) + reported_batch),
                    }
                    if self.mode == 'classification':
                        mean_positive_prediction /= aggrigation_interval
                        mean_negative_prediction /= aggrigation_interval
                        wandb_update_dict.update({
                            'Mean positive prediction':
                                mean_positive_prediction,
                            'Mean negative prediction':
                                mean_negative_prediction
                        })
                        y_pred_info = 'Mean prediction (positive ({0}) | ' \
                                      'negative ({1})): {2:0.4f} | ' \
                                      '{3:0.4f}'.format(
                            n_positive, n_negative, mean_positive_prediction,
                            mean_negative_prediction)
                    else:
                        mean_pred_std_dev = np.std(y_pred_np)
                        mean_pred_mean = np.mean(y_pred_np)
                        mean_true_std_dev = np.std(y_true_np)
                        mean_true_mean = np.mean(y_true_np)
                        y_pred_info = 'True mean | std: {0:0.4f} | {1:0.4f}\t' \
                                      'Predicted mean | std: {2:0.4f} | ' \
                                      '{3:0.4f}'.format(
                            mean_true_mean, mean_true_std_dev, mean_pred_mean,
                            mean_pred_std_dev)
                    loss.backward()
                    loss = float(loss)
                    self.optimiser.step()
                    self.losses.append(loss)

                    if not (reported_batch + 1) % log_interval or \
                            self.batch == total_iters - 1:
                        self.save_loss(log_interval)
                    global_iter += 1

                    try:
                        wandb.log(wandb_update_dict)
                    except wandb.errors.error.Error:
                        pass  # wandb has not been initialised so ignore

                    print_with_overwrite(
                        (
                            'Epoch:',
                            '{0}/{1}'.format(self.epoch + 1, epochs),
                            '|', 'Batch:', '{0}/{1}'.format(
                                reported_batch, len(data_loader))),
                        (y_pred_info,),
                        ('Time elapsed:', time_elapsed, '|',
                         'Time remaining:', eta),
                        ('{0}: {1:.4f}'.format(loss_type, loss),)
                    )

                    mean_positive_prediction, mean_negative_prediction = 0., 0.
                    loss = 0.0
                    n_positive, n_negative = 0, 0

            # save after each epoch
            self.save()

    def save(self, save_path=None):
        """Save all network attributes, including internal states."""

        if save_path is None:
            fname = 'ckpt_epoch_{}.pt'.format(self.epoch + 1)
            save_path = self.save_path / 'checkpoints' / fname

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'learning_rate': self.lr,
            'weight_decay': self.weight_decay,
            'epoch': self.epoch,
            'losses': self.losses,
            'model_state_dict': self.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict()
        }, save_path)

    def save_loss(self, save_interval):
        """Save the loss information to disk.

        Arguments:
            save_interval: how often the loss is being recorded (in batches).
        """
        log_file = self.save_path / 'loss.log'
        start_idx = save_interval * (self.batch // save_interval)
        with open(log_file, 'a') as f:
            f.write('\n'.join(
                [str(idx + start_idx + 1) + ' ' + str(loss) for idx, loss in
                 enumerate(self.losses[-save_interval:])]) + '\n')

    @property
    def param_count(self):
        """Return the number of parameters in the network."""
        return sum(
            [torch.numel(t) for t in self.parameters() if t.requires_grad])

    @staticmethod
    def xavier_init(m):
        """Initialise network weights with xiavier initialisation."""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def colour_pdb(self, rec_fname, lig_fname, output_fname, radius=12):
        """Use model to set b-factor values for each atom to node label.

        This method can be used to visualise the output of point-cloud based
        atom labelling neural networks. The atom labels are stored in a PDB
        file which has the same entries as the receptor input PDB file, with
        the b-factor field filled in to the value given by each atom in the
        network. This can then be visualised in pymol using the commands:

            load <output_fname>
            spectrum b, red_white_blue

        Arguments:
            rec_fname: pdb file containing the protein input
            lig_fname: sdf or mol2 file containing the ligand input. This is
                only used to find the bounding box
            output_fname: where to store the output pdb file
            radius: radius of the bounding box, centred on the ligand

        Raises:
            RuntimeError if the ligand cannot be extracted from the input.
        """

        # RDKit and Openbabel are extremely verbose
        RDLogger.DisableLog('*')
        pybel.ob.obErrorLog.SetOutputLevel(0)

        # Extract bounding box location
        centre_coords = None
        if Path(lig_fname).suffix == '.sdf':
            for item in Chem.SDMolSupplier(str(Path(lig_fname).expanduser())):
                centre_coords = get_centre_coordinates(item)
                break
        else:
            for item in Chem.MolFromMol2File(str(Path(lig_fname).expanduser())):
                centre_coords = get_centre_coordinates(item)
                break
        if centre_coords is None:
            raise RuntimeError('Unable to extract ligand from {}'.format(
                lig_fname))

        dc = DistanceCalculator()

        # Extract openbabel and biopython protein objects. Openbabel must be
        # used for accurate atom typing, but only biopython can be used to set
        # b-factor information.
        rec_fname = Path(rec_fname).expanduser()
        output_fname = str(Path(output_fname).expanduser())
        receptor_bp = dc.read_file(rec_fname, False, read_type='biopython')
        receptor_ob = dc.read_file(rec_fname, False, read_type='openbabel')

        # Set up bookkeeping so we can go between openbabel and biopython,
        # because openbabel types and indices do not translate to biopython
        # so we have to use coordinates as indices...
        pos_to_idx = defaultdict(lambda: defaultdict(lambda: dict()))

        xs, ys, zs = [], [], []
        types, pdb_types = [], []
        for idx, ob_atom in enumerate(receptor_ob):
            smina_type = dc.obatom_to_smina_type(ob_atom)
            if smina_type == "NumTypes":
                smina_type_int = len(self.atom_type_data)
            else:
                smina_type_int = dc.atom_types.index(smina_type)
            type_int = dc.type_map[smina_type_int]

            ainfo = [i for i in ob_atom.coords]
            ainfo.append(type_int)
            str_coords = [str(i) for i in ob_atom.coords]
            pos_to_idx[str_coords[0]][str_coords[1]][str_coords[2]] = idx

            xs.append(ainfo[0] - centre_coords[0])
            ys.append(ainfo[1] - centre_coords[1])
            zs.append(ainfo[2] - centre_coords[2])
            types.append(ainfo[3])

        # Extract information about atoms in bounding box
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        df = pd.DataFrame()
        df['x'] = xs
        df['y'] = ys
        df['z'] = zs
        df['types'] = types
        df['atom_idx'] = np.arange(len(df))
        df['sq_vec'] = df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2
        df = df[df.sq_vec < radius ** 2].copy()

        # Openbabel and biopython do not use the same indexing so we have to be
        # sneaky...
        included_indices = df['atom_idx'].to_numpy()
        coords = torch.from_numpy(
            np.vstack([df['x'], df['y'], df['z']]).T).float()
        coords = repeat(coords, 'a b -> n a b', n=2)
        feats = F.one_hot(
            torch.from_numpy(df['types'].to_numpy()), num_classes=12).float()
        feats = repeat(feats, 'a b -> n a b', n=2)
        mask = torch.ones(2, feats.shape[1]).byte()

        # Obtain the atom labels from the network
        labels = self(
            (coords.cuda(),
             feats.cuda(),
             mask.cuda())
        ).cpu().detach().numpy()[0, :].squeeze()

        # Set probability of atoms in bounding box, all others set to zero
        all_labels = np.zeros((len(xs),))
        all_labels[included_indices] = labels

        # Finally we can set the b-factors using biopython
        for atom in receptor_bp.get_atoms():
            x, y, z = atom.get_coord()
            ob_idx = pos_to_idx[str(x)][str(y)][str(z)]
            atom.set_bfactor(all_labels[ob_idx])

        # Write modified PDB to disk
        io = PDB.PDBIO()
        io.set_structure(receptor_bp)
        io.save(output_fname)

        print()
        print('Receptor:', rec_fname)
        print('Ligand:', lig_fname)
        print('Output:', output_fname)