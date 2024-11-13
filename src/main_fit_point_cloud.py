import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from tqdm import tqdm
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
from collections import OrderedDict
from enum import Enum
import numpy as np
import sys, os, csv
from prettytable import from_csv
from warnings import warn
import dataio

# imports specific to sdf
import igl, geometry

# print(plt.style.available)  # uncomment to view the available plot styles
plt.rcParams['text.usetex'] = False  # tex not necessary here and may cause error if not installed
plt.style.use("seaborn-white")  # if throws error, use "seaborn-white" or "seaborn-v0_8-white"

set_t = {
    'dtype': torch.float32,  # double precision for more accurate training
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

available_activations = [nn.ReLU, nn.ELU, nn.GELU, nn.Sigmoid]  # list of currently supported activation functions

to_numpy = lambda x : x.detach().cpu().numpy()  # converts tensors to numpy arrays

class SineLayer(nn.Module):
    """
    The SineLayer implementation given by the SIRENZ paper.

    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    hyperparameter.
    If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    """

    def __init__(self, in_features: int, out_features: int, bias=True,
                 is_first: bool=False, omega_0: float=30.):
        """

        :param in_features:
        :param out_features:
        :param bias:
        :param is_first:
        :param omega_0:
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        """
        :return:
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)


class Siren(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int, out_features: int,
                 lrate: float, outermost_linear: bool=False,
                 first_omega_0: int=30, hidden_omega_0: float=30., latent_dim: int = 0,
                 step_size: Optional[int] = None, gamma: Optional[float] = None):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.latent, self.modulator = None, None
        opt_parameters = []  # optimizable parameters
        if latent_dim > 0:
            self.modulator = Modulator(
                dim_in=latent_dim,
                dim_hidden=hidden_features,
                num_layers=hidden_layers
            )
            self.latent = nn.Parameter(torch.zeros(latent_dim).normal_(0, 1e-2))
            opt_parameters.extend([
                self.latent,
                *self.modulator.parameters(),
            ])

        self.model = []

        for i in range(hidden_layers):
            idx_str = f"{i:4d}_SineLayer"
            is_first = i == 0
            omega_0 = first_omega_0 if is_first else hidden_omega_0
            input_dim = in_features if is_first else hidden_features
            self.model.append(
                (idx_str, SineLayer(input_dim, hidden_features,
                                      is_first=i == 0, omega_0=omega_0))
            )

        # append last layer
        self.model.append(
            ("LastLayer", nn.Sequential(nn.Linear(hidden_features, out_features), nn.Tanh()))
        )

        self.model = nn.ModuleDict(OrderedDict(self.model))
        for l in self.model.values():
            opt_parameters.extend(l.parameters())
        self.optimizer = optim.Adam(opt_parameters, lr=lrate)
        self.loss_fn = nn.MSELoss(reduction='none')

        # set LR scheduler
        self.scheduler = None
        if step_size is not None and gamma is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size,
                                                       gamma=gamma)

    def forward(self, x):

        # create mods (simply tuple of Nones if not enabled)
        if self.latent is not None and self.modulator is not None:
            latent_input = self.latent
            mods = self.modulator(latent_input)
        else:
            mods = tuple([None] * self.hidden_layers)

        hidden_layers = tuple([l for k, l in self.model.items() if k.split('_')[-1] == 'SineLayer'])
        last_layer = self.model['LastLayer']
        for l, mod in zip(hidden_layers, mods):
            # pass through sine layer
            x = l(x)

            # apply mod if feature is enabled
            if mod is not None:
                x *= mod.unsqueeze(0)

        # apply last layer's output
        x = last_layer(x)

        return x

    def forward_with_coords(self, x):
        x = x.clone().detach().requires_grad_(
            True)  # allows to take derivative w.r.t. input
        output = x

        # create mods (simply tuple of Nones if not enabled)
        if self.latent is not None and self.modulator is not None:
            latent_input = self.latent
            mods = self.modulator(latent_input)
        else:
            mods = tuple([None] * self.hidden_layers)

        hidden_layers = tuple([l for k, l in self.model.items() if k.split('_')[-1] == 'SineLayer'])
        last_layer = self.model['LastLayer']
        for l, mod in zip(hidden_layers, mods):
            # pass through sine layer
            output = l(output)

            # apply mod if feature is enabled
            if mod is not None:
                output *= mod.unsqueeze(0)

        # apply last layer's output
        output = last_layer(output)

        return output, x

    def step(self, x: Tensor, y: Tensor, weights: Tensor) -> float:
        """
        Returns the loss of a single forward pass
        :param x:       (Batch, input size)
        :param y:       (Batch, output size)
        :param weights: (Batch, input size), weights to apply to input samples to correct class imbalance
        :return:        loss
        """
        # zero the gradients
        self.optimizer.zero_grad()

        # pass the batch through the model
        y_hat = self.forward(x)

        # compute the loss
        unweighted_loss = self.loss_fn(y_hat, y)
        loss = (unweighted_loss * weights).mean()

        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def step_eikonal(self, x: Tensor, y: Tensor, on_surface_mask: Tensor) -> Tuple[float, list[float]]:
        """
        Returns the loss of a single forward pass
        :param x:       (Batch, input size) -- batches of coordinates
        :param y:       (Batch, output size) -- the target normals; only meaningful for points on the surface
        :param weights: (Batch, input size), weights to apply to input samples to correct class imbalance
        :return:        loss
        """
        # penalization multipliers recommended by SIREN paper
        c1 = 5e1
        c2 = 3e3
        c3 = 1e2

        def _gradient(x, y, grad_outputs=None):
            if grad_outputs is None:
                grad_outputs = torch.ones_like(y)
            grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
            return grad

        def _psi(x, a: float = 100):
            exp_in = -a * x.abs()
            exp_out = torch.exp(exp_in).squeeze(1)
            return exp_out

        # don't need singleton dimension in the mask
        on_surface_mask = on_surface_mask.squeeze(1)

        # zero the gradients
        self.optimizer.zero_grad()

        # pass the batch through the model
        output, coords = self.forward_with_coords(x)

        # compute the graident of the output w.r.t. input
        grad_output = _gradient(coords, output)

        # eikonal loss
        eik_loss = c1*(1 - torch.linalg.vector_norm(grad_output, dim=1)).abs().unsqueeze(1)

        # on surface loss
        surface_output = output[on_surface_mask]
        surface_grad = grad_output[on_surface_mask]
        on_surface_loss = torch.zeros_like(output)
        target_normals = y[on_surface_mask]
        pre_on_surface_loss = c2 * surface_output.abs()
        pre_on_surface_loss += c3 * (1 - F.cosine_similarity(surface_grad, target_normals, dim=1)).unsqueeze(1)
        on_surface_loss[on_surface_mask, :] = pre_on_surface_loss

        # off surface loss
        off_surface_mask = torch.logical_not(on_surface_mask)
        off_surface_output = output[off_surface_mask]
        off_surface_loss = torch.zeros_like(output)
        off_surface_loss[off_surface_mask, :] = c2*_psi(off_surface_output).unsqueeze(1)

        # get loss description so that more info can be printed
        loss_desc = [to_numpy(eik_loss.clone()), to_numpy(on_surface_loss.clone()), to_numpy(off_surface_loss.clone())]
        loss_desc = [n.mean().item() for n in loss_desc]

        # add together for a total loss
        loss = eik_loss + on_surface_loss + off_surface_loss
        loss = loss.mean()

        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_desc

def plot_training_metrics(losses: list[float], save_path: Optional[str] = None, display: bool = False):
    """
    Displays and/or saves the metrics recorded during the training of the implicit surface.
    :param losses:          List of losses over epochs
    :param correct_fracs:   List of fraction of correct sign predictions of epochs
    :param save_path:       Path to save the plot to
    :param display:         If true, displays the plot
    :return:
    """
    if save_path is None and not display:
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title("Training Loss")
    ax1.grid()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if display:
        plt.show()
    else:
        plt.close()

def fit_model(
        NetObject: Siren,
        train_loader: DataLoader,
        epochs: int
) -> Tuple[list[float], Siren]:
    """
    Given a neurol network and train loader, fit the neural network to the training dataset and record the losses.
    The training heuristics that are returned are:

    * `losses` -- losses per epoch
    * `correct_counts` -- number of predictions that have predicted the correct sign
    * `correct_fracs` -- fraction of predictions that have predicted the correct sign

    :param NetObject:       Neural network object to train
    :param train_loader:    Training dataset
    :param epochs:          Number of epochs to run
    :return:                Training heuristics and trained `NetObject`
    """

    # send to device
    NetObject = NetObject.to(**set_t)

    # train and record losses
    losses, correct_fracs = [], []
    n_total = 0
    epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)
    for epoch in range(epochs):
        epoch_loss, eik_loss, on_surface_loss, off_surface_loss = [0.0] * 4
        for batch in train_loader:
            # load in batch_data
            inputs, labels = batch
            batch_x = inputs['coords'].squeeze(0)
            batch_normal = labels['normals'].squeeze(0)
            batch_surface_mask = labels['surface_mask'].squeeze(0)
            batch_x = batch_x.to(**set_t)
            batch_normal = batch_normal.to(**set_t)
            batch_surface_mask = batch_surface_mask.to(device=set_t['device'])  # should remain as bool

            n_total += len(batch_x)
            curr_epoch_loss, loss_desc = NetObject.step_eikonal(batch_x, batch_normal, batch_surface_mask)

            # update epoch losses
            [curr_eik_loss, curr_on_surface_loss, curr_off_surface_loss] = loss_desc
            epoch_loss += curr_epoch_loss
            eik_loss += curr_eik_loss
            on_surface_loss += curr_on_surface_loss
            off_surface_loss += curr_off_surface_loss

        # get the current learning rate
        if NetObject.scheduler is not None:
            NetObject.scheduler.step()
            current_lr = NetObject.scheduler.get_last_lr()[0]
        else:
            current_lr = NetObject.lr
        # calculate the fraction of correctly predicted signs
        # calculate the epoch loss and update progress bar
        train_loader_len = len(train_loader)
        epoch_loss /= train_loader_len
        eik_loss /= train_loader_len
        on_surface_loss /= train_loader_len
        off_surface_loss /= train_loader_len
        losses.append(epoch_loss)
        epoch_details = {
                'Epoch Loss': epoch_loss,
                'Eik Loss': eik_loss,
                'On Surface Loss': on_surface_loss,
                'Off Surface Loss': off_surface_loss,
                'Learning Rate': current_lr,
                'Train Loader Length': train_loader_len
            }
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix(epoch_details)

    # return metrics and trained network
    return losses, NetObject


def main(args: dict):

    print(f"Torch Settings: {set_t}")

    ##  unpack arguments

    # Build arguments
    input_file = args["input_file"]
    output_file = args["output_file"]
    if input_file is None or output_file is None:
        raise ValueError("input_file and/or output_file is None")
    # network
    n_layers = args["n_layers"]
    layer_width = args["layer_width"]
    # siren params
    siren_model = args["siren_model"]
    siren_latent_dim = args["siren_latent_dim"]
    siren_outermost_linear = args["siren_outermost_linear"]
    siren_first_omega_0 = args["siren_first_omega_0"]
    siren_hidden_omega_0 = args["siren_hidden_omega_0"]

    # loss / data
    fit_mode = args["fit_mode"]
    n_epochs = args["n_epochs"]
    # training
    lr = args["lr"]
    batch_size = args["batch_size"]
    lr_decay_every = args["lr_decay_every"]
    lr_decay_frac = args["lr_decay_frac"]
    # general options
    display_plots = args["display_plots"]

    print(f"Program Configuration: {args}")

    model_params = {
        'in_features': 3,
        'hidden_features': layer_width,
        'hidden_layers': n_layers,
        'out_features': 1,
        'lrate': lr,
        'outermost_linear': siren_outermost_linear,
        'first_omega_0': siren_first_omega_0,
        'hidden_omega_0': siren_hidden_omega_0,
        'latent_dim': siren_latent_dim,
        'step_size': lr_decay_every,
        'gamma': lr_decay_frac,
    }

    NetObject = Siren(**model_params)

    # load the dataset
    sdf_dataset = dataio.PointCloud(input_file, on_surface_points=batch_size)
    dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1)

    losses, NetObject = fit_model(NetObject, dataloader, n_epochs)

    NetObject.eval()  # set to evaluation mode

    # save the neural network in Torch format
    pth_file = output_file.replace('.xyz', '.pth')
    print(f"Saving model to {pth_file}...")
    pth_dict = {
        "state_dict": NetObject.state_dict(),
        "model_params": model_params,
        "is_siren": siren_model,
    }
    torch.save(pth_dict, pth_file)

    # display results
    plt_file = output_file.replace('.xyz', '.png')
    plot_training_metrics(losses, plt_file, display_plots)


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)

    # network
    parser.add_argument("--activation", type=str, default='elu')
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--layer_width", type=int, default=32)
    #positional arguments
    parser.add_argument("--positional_encoding", action='store_true')
    parser.add_argument("--positional_count", type=int, default=10)
    parser.add_argument("--positional_pow_start", type=int, default=-3)
    parser.add_argument("--positional_prepend", action='store_true')
    # siren arguments
    parser.add_argument("--siren_model", action='store_true')
    parser.add_argument("--siren_latent_dim", type=int, default=0)
    parser.add_argument("--siren_outermost_linear", action='store_true')
    parser.add_argument("--siren_first_omega_0", type=int, default=30)
    parser.add_argument("--siren_hidden_omega_0", type=int, default=30)

    # loss / data
    parser.add_argument("--fit_mode", type=str, default='sdf')
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1000000)
    parser.add_argument("--sample_ambient_range", type=float, default=1.25)
    parser.add_argument("--sample_weight_beta", type=float, default=20.)
    parser.add_argument("--sample_221", action='store_true')
    parser.add_argument('--show_sample_221', action='store_true')
    parser.add_argument("--sdf_max", type=float, default=0.1)

    # training
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr_decay_every", type=int, default=99999)
    parser.add_argument("--lr_decay_frac", type=float, default=.5)

    # general options
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--display_plots", action='store_true')
    parser.add_argument('--check_csv_table', type=str, default=None)

    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict

if __name__ == '__main__':
    # parse user arguments
    args_dict = parse_args()
    main(args_dict)