import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch import nn

from tepid_invariance.utils import get_eta, format_time, print_with_overwrite

try:
    from openbabel import pybel
except (ModuleNotFoundError, ImportError):
    import pybel


class PointNeuralNetwork(nn.Module):
    """Base class for node classification and regression tasks."""

    def __init__(self, save_path, learning_rate, weight_decay=None,
                 mode='classification', silent=False, weighted_loss=False,
                 **model_kwargs):
        super().__init__()
        self.batch = 0
        self.epoch = 0
        self.losses = []
        self.save_path = Path(save_path).expanduser()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.save_path / 'predictions.txt'
        self.mode = mode
        self.weighted_loss = weighted_loss

        self.loss_plot_file = self.save_path / 'loss.png'

        self.lr = learning_rate
        self.weight_decay = weight_decay

        self.loss_log_file = self.save_path / 'loss.log'

        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        self.build_net(**model_kwargs)
        self.optimiser = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=weight_decay)

        if not silent:
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

    @staticmethod
    def _get_y_true(y):
        """Preprocessing for getting the true node labels."""
        return y.cuda()

    @staticmethod
    def _process_inputs(x):
        """Preprocessing for getting the inputs."""
        return x.cuda()

    def _get_loss_component(self, y_true, y_pred, indices):
        n_atoms = len(indices[0])
        if not n_atoms:
            return torch.FloatTensor([0])
        atoms_true = y_true[indices]
        atoms_pred = y_pred[indices]
        n_positives = int(torch.sum(atoms_true))

        n_positives = max(1., n_positives)  # div0 errors avoided
        N_ratio = (n_atoms - n_positives) / n_positives
        loss = self.loss(atoms_pred, atoms_true)
        loss[torch.where(atoms_true)] *= N_ratio
        loss = torch.mean(loss)
        return loss

    def _get_types_component(self, y_true, y_pred, n):
        indices = torch.where(self.types == n)
        n_atoms = len(indices[0])
        if not n_atoms:
            return torch.FloatTensor([0])
        atoms_true = y_true[indices]
        atoms_pred = y_pred[indices]
        n_positives = int(torch.sum(atoms_true))

        n_positives = max(1., n_positives)  # div0 errors avoided
        N_ratio = (n_atoms - n_positives) / n_positives
        loss = self.loss(atoms_pred, atoms_true)
        loss[torch.where(atoms_true)] *= N_ratio
        loss = torch.mean(loss)
        return loss

    def _get_loss(self, y_true, y_pred, weighted=False):
        """Process the loss."""
        if not weighted:
            return torch.mean(self.loss(y_pred, y_true))

        loss = torch.FloatTensor([0]).cuda()
        for n in range(1, 12):
            loss += self._get_types_component(y_true, y_pred, n).cuda()
        return loss / 11

        # Other experimental loss functions
        for component in [
            self.N_indices, self.O_indices, self.C_indices, self.F_indices]:
            loss += self._get_loss_component(y_true, y_pred, component).cuda()
        return loss / 4

        total_nodes = np.product(y_true.shape)
        positive_nodes = float(torch.sum(y_true))
        ratio = total_nodes / positive_nodes
        _loss = self.loss(y_pred, y_true)
        _loss[torch.where(y_true)] *= min(10, ratio)
        return torch.mean(_loss)

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
            for self.batch, (x, y_true, atomic_numbers) in enumerate(
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

                self.types = atomic_numbers
                self.C_indices = torch.where(atomic_numbers == 6)
                self.N_indices = torch.where(atomic_numbers == 7)
                self.O_indices = torch.where(atomic_numbers == 8)
                self.F_indices = torch.where(atomic_numbers == 9)

                loss += self._get_loss(y_true, y_pred, self.weighted_loss)

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
