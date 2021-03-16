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
from eqv_transformer.eqv_attention import EquivariantTransformerBlock
from lie_conv.lieGroups import SE3
from lie_conv.utils import Pass
from rdkit import Chem, RDLogger
from torch import nn

from preprocessing.distance_calculator import DistanceCalculator, \
    get_centre_coordinates
from utils import get_eta, format_time, print_with_overwrite


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PointNeuralNetwork(nn.Module):

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

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @abstractmethod
    def _get_y_true(self, y):
        pass

    def _process_inputs(self, x):
        return x.cuda()

    def _get_loss(self, y_true, y_pred):
        loss = self.loss(y_pred, y_true)
        return loss

    def optimise(self, data_loader, epochs=1, opt_cycle=-1):
        """Train the network.

        Trains the neural network. Displays training information and plots the
        loss. All figures and logs are saved to save_path.

        Arguments:
            data_loader: pytorch DataLoader object for training
            epochs: number of complete training cycles
            opt_cycle: (for active learning): active learning cycle
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

    def test(self, data_loader, predictions_file=None):
        """Use trained network to perform inference on the test set.

        Uses the neural network (in Session.network), to perform predictions
        on the structures found in <test_data_root>, and saves this output
        to <save_path>/predictions_<test_data_root.name>.txt.

        Arguments:
            data_loader:
            predictions_file:
        """
        self.cuda()
        start_time = time.time()
        log_interval = 10
        decoy_mean_pred, active_mean_pred = 0.5, 0.5
        predictions = ''
        if predictions_file is None:
            predictions_file = self.predictions_file
        predictions_file = Path(predictions_file).expanduser()
        if predictions_file.is_file():
            predictions_file.unlink()
        self.eval()
        with torch.no_grad():
            for self.batch, (x, y_true, ligands, receptors) in enumerate(
                    data_loader):

                x = self._process_inputs(x)
                y_true = self._get_y_true(y_true).cuda()
                y_pred = self.forward_pass(x)
                y_true_np = y_true.cpu().detach().numpy()
                y_pred_np = nn.Softmax(dim=1)(y_pred).cpu().detach().numpy()

                active_idx = (np.where(y_true_np > 0.5), 1)
                decoy_idx = (np.where(y_true_np < 0.5), 1)

                scale = len(y_true) / len(data_loader)
                _ = self._get_loss(y_true, y_pred)

                eta = get_eta(start_time, self.batch, len(data_loader))
                time_elapsed = format_time(time.time() - start_time)

                wandb_update_dict = {
                    'Time remaining (validation)': eta,
                    'Binary crossentropy (validation)': self.bce_loss,
                    'Batch': self.batch + 1
                }

                if len(active_idx[0][0]):
                    active_mean_pred = np.mean(y_pred_np[active_idx])
                    wandb_update_dict.update({
                        'Mean active prediction (validation)': active_mean_pred
                    })
                if len(decoy_idx[0][0]):
                    decoy_mean_pred = np.mean(y_pred_np[decoy_idx])
                    wandb_update_dict.update({
                        'Mean decoy prediction (validation)': decoy_mean_pred,
                    })

                try:
                    wandb.log(wandb_update_dict)
                except wandb.errors.error.Error:
                    pass  # wandb has not been initialised so ignore

                print_with_overwrite(
                    ('Inference on: {}'.format(data_loader.dataset.base_path),
                     '|', 'Iteration:', '{0}/{1}'.format(
                        self.batch + 1, len(data_loader))),
                    ('Time elapsed:', time_elapsed, '|',
                     'Time remaining:', eta),
                    ('Loss: {0:.4f}'.format(self.bce_loss), '|',
                     'Mean active: {0:.4f}'.format(active_mean_pred), '|',
                     'Mean decoy: {0:.4f}'.format(decoy_mean_pred))
                )

                predictions += '\n'.join(['{0} | {1:.7f} {2} {3}'.format(
                    int(y_true_np[i]),
                    y_pred_np[i, 1],
                    receptors[i],
                    ligands[i]) for i in range(len(receptors))]) + '\n'

                # Periodically write predictions to disk
                if not (self.batch + 1) % log_interval or self.batch == len(
                        data_loader) - 1:
                    with open(predictions_file, 'a') as f:
                        f.write(predictions)
                        predictions = ''

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
        return sum(
            [torch.numel(t) for t in self.parameters() if t.requires_grad])

    @staticmethod
    def xavier_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def colour_pdb(self, rec_fname, lig_fname, output_fname, radius=12):
        RDLogger.DisableLog('*')
        pybel.ob.obErrorLog.SetOutputLevel(0)
        if Path(lig_fname).suffix == '.sdf':
            for item in Chem.SDMolSupplier(str(Path(lig_fname).expanduser())):
                centre_coords = get_centre_coordinates(item)
                break
        else:
            for item in Chem.MolFromMol2File(str(Path(lig_fname).expanduser())):
                centre_coords = get_centre_coordinates(item)
                break
        dc = DistanceCalculator()

        rec_fname = Path(rec_fname).expanduser()
        output_fname = str(Path(output_fname).expanduser())
        receptor_bp = dc.read_file(rec_fname, False, read_type='biopython')
        receptor_ob = dc.read_file(rec_fname, False, read_type='openbabel')

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
            pdb_types.append(
                ob_atom.residue.OBResidue.GetAtomID(ob_atom.OBAtom).strip())

        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        df = pd.DataFrame()
        df['x'] = xs
        df['y'] = ys
        df['z'] = zs
        df['types'] = types
        df['atom_idx'] = np.arange(len(df))
        df['sq_vec'] = df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2
        df = df[df.sq_vec < radius ** 2].copy()
        included_indices = df['atom_idx'].to_numpy()
        coords = torch.from_numpy(
            np.vstack([df['x'], df['y'], df['z']]).T).float()
        coords = repeat(coords, 'a b -> n a b', n=2)
        feats = F.one_hot(
            torch.from_numpy(df['types'].to_numpy()), num_classes=11).float()
        feats = repeat(feats, 'a b -> n a b', n=2)
        mask = torch.ones(2, feats.shape[1]).byte()
        labels = self(
            (coords.cuda(),
             feats.cuda(),
             mask.cuda())
        ).cpu().detach().numpy()[0, :].squeeze()
        all_labels = np.zeros((len(xs),))
        all_labels[included_indices] = labels

        for atom in receptor_bp.get_atoms():
            x, y, z = atom.get_coord()
            ob_idx = pos_to_idx[str(x)][str(y)][str(z)]
            atom.set_bfactor(all_labels[ob_idx])

        io = PDB.PDBIO()
        io.set_structure(receptor_bp)
        io.save(output_fname)


class LieTepid(PointNeuralNetwork):
    """Adapted from https://github.com/anonymous-code-0/lie-transformer"""

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return tuple([ten.cuda() for ten in x])

    def build_net(self, dim_input, dim_hidden, num_layers,
                  num_heads, group=SE3(0.2), liftsamples=1,
                  block_norm="layer_pre", kernel_norm="none", kernel_type="mlp",
                  kernel_dim=16, kernel_act="swish", mc_samples=0, fill=1.0,
                  attention_fn="norm_exp", feature_embed_dim=None,
                  max_sample_norm=None, lie_algebra_nonlinearity=None,
                  dropout=0):

        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * (num_layers + 1)

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers

        attention_block = lambda dim, n_head: EquivariantTransformerBlock(
            dim, n_head, group, block_norm=block_norm, kernel_norm=kernel_norm,
            kernel_type=kernel_type, kernel_dim=kernel_dim,
            kernel_act=kernel_act, mc_samples=mc_samples, fill=fill,
            attention_fn=attention_fn, feature_embed_dim=feature_embed_dim,
        )

        self.net = nn.Sequential(
            Pass(nn.Linear(dim_input, dim_hidden[0]), dim=1),
            *[
                attention_block(dim_hidden[i], num_heads[i])
                for i in range(num_layers)
            ],
            Pass(nn.Linear(dim_hidden[-1], 1), dim=1)
        )

        self.group = group
        self.liftsamples = liftsamples
        self.max_sample_norm = max_sample_norm

        self.lie_algebra_nonlinearity = lie_algebra_nonlinearity
        if lie_algebra_nonlinearity is not None:
            if lie_algebra_nonlinearity == "tanh":
                self.lie_algebra_nonlinearity = nn.Tanh()
            else:
                raise ValueError(
                    f'{lie_algebra_nonlinearity} is not a supported '
                    f'nonlinearity'
                )

    def forward(self, x):
        if self.max_sample_norm is None:
            lifted_data = self.group.lift(x, self.liftsamples)
        else:
            lifted_data = [
                torch.tensor(self.max_sample_norm * 2, device=x[0].device),
                0,
                0,
            ]
            while lifted_data[0].norm(dim=-1).max() > self.max_sample_norm:
                lifted_data = self.group.lift(x, self.liftsamples)

        if self.lie_algebra_nonlinearity is not None:
            lifted_data = list(lifted_data)
            pairs_norm = lifted_data[0].norm(dim=-1) + 1e-6
            lifted_data[0] = lifted_data[0] * (
                    self.lie_algebra_nonlinearity(pairs_norm / 7) / pairs_norm
            ).unsqueeze(-1)

        return self.net(lifted_data)[1][..., -1]
