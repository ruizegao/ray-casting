from sched import scheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import os
from functools import partial
from warnings import warn
from torch.optim.lr_scheduler import LambdaLR
import dataio

USE_WANDB = bool(os.environ.get('USE_WANDB', 0))
WANDB_GROUP = None
if USE_WANDB:
    import wandb

# print(plt.style.available)  # uncomment to view the available plot styles
plt.rcParams['text.usetex'] = False  # tex not necessary here and may cause error if not installed
plt.style.use("seaborn-v0_8-white")  # if throws error, use "seaborn-white" or "seaborn-v0_8-white"

set_t = {
    'dtype': torch.float32,  # double precision for more accurate training
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

available_activations = [nn.ReLU, nn.ELU, nn.GELU, nn.Sigmoid]  # list of currently supported activation functions

to_numpy = lambda x : x.detach().cpu().numpy()  # converts tensors to numpy arrays

### Custom LR Schedulers
def linear_decay(epoch, initial_lr, final_lr, total_epochs):
    return 1 - epoch / total_epochs * (1 - final_lr / initial_lr)

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
                 siren_lrate: float, latent_lrate: float, num_epochs: int,
                 final_siren_lrate: Optional[float]=None, final_latent_lrate: Optional[float] = None,
                 first_omega_0: int=30, hidden_omega_0: float=30., latent_dim: int = 0,
                 step_size: Optional[int] = None, gamma: Optional[float] = None,
                 clip_gradient_norm: Optional[float] = None, scheduler_type: str='none'):
        """

        Initializes a Siren model for fitting weak signed distance functions. Latent variables are supported as
        well for modulation.

        :param in_features:         Input dimension
        :param hidden_features:     Hidden layer width
        :param hidden_layers:       Number of hidden layers
        :param out_features:        Output dimension
        :param siren_lrate:         Learning rate for Siren network
        :param latent_lrate:        Learning rate for latent variable parameters
        :param first_omega_0:       omega to use for first Siren layer
        :param hidden_omega_0:      omegas to use for intermediate Siren layers
        :param latent_dim:          Dimension of the latent variable
        :param step_size:           Number of steps before applying LR Scheduler
        :param gamma:               LR Scheduler Decay
        :param clip_gradient_norm:  Max norm to clip model gradients. Helps with stabilization when using latent variables.
        """
        super().__init__()
        if final_siren_lrate is None: final_siren_lrate = siren_lrate
        if final_latent_lrate is None: final_latent_lrate = latent_lrate
        self.hidden_layers = hidden_layers
        self.clip_gradient_norm = clip_gradient_norm
        self.latent, self.modulator = None, None
        self.has_latent = latent_dim > 0
        self.opt_latent_parameters = []  # optimizable parameters
        self.opt_siren_parameters = []
        if latent_dim > 0:
            self.modulator = Modulator(
                dim_in=latent_dim,
                dim_hidden=hidden_features,
                num_layers=hidden_layers
            )
            self.latent = nn.Parameter(torch.zeros(latent_dim).normal_(0, 1e-2))
            self.opt_latent_parameters.extend([
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
            self.opt_siren_parameters.extend(l.parameters())
        if self.has_latent:
            self.latent_optimizer = optim.Adam(self.opt_latent_parameters, lr=latent_lrate)
            self.latent_lrate = latent_lrate
        else:
            self.latent_optimizer = None
            self.latent_lrate = None
        self.siren_optimizer = optim.Adam(self.opt_siren_parameters, lr=siren_lrate)
        self.siren_lrate = siren_lrate
        self.loss_fn = nn.MSELoss(reduction='none')

        # set linear LR scheduler
        self.siren_scheduler, self.latent_scheduler = None, None
        if scheduler_type == 'step':
            assert step_size is not None and gamma is not None, "Must specify step size and gamma to use Step LR"
            self.siren_scheduler = optim.lr_scheduler.StepLR(self.siren_optimizer, step_size=step_size,
                                                       gamma=gamma)
            if self.has_latent:
                self.latent_scheduler = optim.lr_scheduler.StepLR(self.latent_optimizer, step_size=step_size,
                                                                 gamma=gamma)
        elif scheduler_type == 'linear':
            decay_func_siren = partial(linear_decay, initial_lr=siren_lrate, final_lr=final_siren_lrate,
                                       total_epochs=num_epochs)
            self.siren_scheduler = optim.lr_scheduler.LambdaLR(self.siren_optimizer, lr_lambda=decay_func_siren)
            if self.has_latent:
                decay_func_latent = partial(linear_decay, initial_lr=latent_lrate, final_lr=final_latent_lrate,
                                            total_epochs=num_epochs)
                self.latent_scheduler = optim.lr_scheduler.LambdaLR(self.latent_optimizer, lr_lambda=decay_func_latent)
        elif scheduler_type == 'none':
            pass
        else:
            raise ValueError(f"Scheduler type of {scheduler_type} is not recognized")



    def forward(self, x):

        # create mods (simply tuple of Nones if not enabled)
        if self.has_latent:
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
        if self.has_latent:
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
        self.siren_optimizer.step()
        if self.has_latent:
            self.latent_optimizer.step()

        return loss.item()

    def step_eikonal(self, x: Tensor, y: Tensor, on_surface_mask: Tensor) -> Tuple[float, list[float]]:
        """
        Returns the loss of a single forward pass
        :param x:       (Batch, input size) -- batches of coordinates
        :param y:       (Batch, output size) -- the target normals; only meaningful for points on the surface
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
        self.siren_optimizer.zero_grad()
        if self.has_latent:
            self.latent_optimizer.zero_grad()

        # pass the batch through the model
        output, coords = self.forward_with_coords(x)

        # compute the graident of the output w.r.t. input
        grad_output = _gradient(coords, output)

        # eikonal loss
        # Desc: The norm of the gradient should be constrained to be 1 everywhere
        eik_loss = c1*(1 - torch.linalg.vector_norm(grad_output, dim=1)).abs().unsqueeze(1)

        # on surface loss
        # Desc: We should have that:
        # 1) Points on the surfaces have an SDF of 0
        # 2) Points on the surface should have their normals align with the target normals from the dataset
        surface_output = output[on_surface_mask]
        surface_grad = grad_output[on_surface_mask]
        on_surface_loss = torch.zeros_like(output)
        target_normals = y[on_surface_mask]
        pre_on_surface_loss = c2 * surface_output.abs()
        pre_on_surface_loss += c3 * (1 - F.cosine_similarity(surface_grad, target_normals, dim=1)).unsqueeze(1)
        on_surface_loss[on_surface_mask, :] = pre_on_surface_loss

        # off surface loss
        # Desc: Points close to the surface should be penalized if their SDF is close to 0
        off_surface_mask = torch.logical_not(on_surface_mask)
        off_surface_output = output[off_surface_mask]
        off_surface_loss = torch.zeros_like(output)
        off_surface_loss[off_surface_mask, :] = c2*_psi(off_surface_output).unsqueeze(1)

        # return loss description for more details
        loss_desc = [to_numpy(eik_loss.clone()), to_numpy(on_surface_loss.clone()), to_numpy(off_surface_loss.clone())]
        loss_desc = [n.mean().item() for n in loss_desc]

        # add together for a total loss
        loss = eik_loss + on_surface_loss + off_surface_loss
        loss = loss.mean()

        # update model
        loss.backward()

        # perform gradient clipping
        if self.clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.opt_siren_parameters, max_norm=self.clip_gradient_norm)
            if self.has_latent:
                torch.nn.utils.clip_grad_norm_(self.opt_latent_parameters, max_norm=self.clip_gradient_norm)

        self.siren_optimizer.step()
        if self.has_latent:
            self.latent_optimizer.step()

        return loss.item(), loss_desc

    def scheduler_step(self) -> Tuple[Optional[float], Optional[float]]:

        # step with siren scheduler
        if self.siren_scheduler is not None:
            self.siren_scheduler.step()
            siren_lr = self.siren_scheduler.get_last_lr()[0]
        else:
            siren_lr = self.siren_lrate

        # step with latent scheduler
        if self.has_latent and self.latent_scheduler is not None:
            self.latent_scheduler.step()
            latent_lr = self.latent_scheduler.get_last_lr()[0]
        else:
            latent_lr = self.latent_lrate

        return siren_lr, latent_lr

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
    global USE_WANDB

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
        current_siren_lr, current_latent_lr = NetObject.scheduler_step()
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
                'Siren Learning Rate': current_siren_lr,
                'Latent Learning Rate': current_latent_lr,
                'Train Loader Length': train_loader_len
            }
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix(epoch_details)
        if USE_WANDB:
            wandb.log(epoch_details)

    # return metrics and trained network
    return losses, NetObject

def load_net_object(pth_file: str) -> Siren:
    """
    A helper function that retrieves a torch network from a .pth file
    :param pth_file:    .pth file to load network parameters and weights from.
    :return:    Network object
    """
    pth_dict = torch.load(pth_file, weights_only=False)
    state_dict = pth_dict["state_dict"]  # weights and biases
    model_params = pth_dict["model_params"]  # rest of the parameters
    NetObject = Siren(**model_params)
    NetObject.load_state_dict(state_dict)  # load in weights and biases
    NetObject.eval()  # set to evaluation mode

    return NetObject


def main(args: dict):
    global USE_WANDB, WANDB_GROUP

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
    clip_gradient_norm = args["clip_gradient_norm"]
    # siren params
    siren_model = args["siren_model"]
    siren_latent_dim = args["siren_latent_dim"]
    siren_first_omega_0 = args["siren_first_omega_0"]
    siren_hidden_omega_0 = args["siren_hidden_omega_0"]

    # loss / data
    fit_mode = args["fit_mode"]
    n_epochs = args["n_epochs"]
    # training
    siren_lr = args["siren_lr"]
    final_siren_lr = args["final_siren_lr"]
    latent_lr = args["latent_lr"]
    final_latent_lr = args["final_latent_lr"]
    scheduler_type = args["scheduler_type"]
    batch_size = args["batch_size"]
    lr_decay_every = args["lr_decay_every"]
    lr_decay_frac = args["lr_decay_frac"]
    # general options
    display_plots = args["display_plots"]

    print(f"Program Configuration: {args}")

    if USE_WANDB:
        if WANDB_GROUP is None:
            WANDB_GROUP = 'manuscript_' + wandb.util.generate_id()
        uniq_id = WANDB_GROUP.split('_')[-1]
        file_name = input_file.split('/')[-1].split('.obj')[0] + '_' + uniq_id

        # start a new wandb run to track this script
        tags = [fit_mode, 'siren_eik']
        if siren_latent_dim > 0:
            tags.append('latent_modulation')
        wandb.init(
            # set the wandb project and name where this run will be logged
            project="main_fit_implicit_point_cloud",
            name=file_name,
            # track hyperparameters and run metadata
            config=args_dict,
            # set group
            group=WANDB_GROUP,
            # set tags
            tags=tags
        )

    print(f"WANDB ENABLED: {USE_WANDB} | WANDB GROUP: {WANDB_GROUP}")

    model_params = {
        'in_features': 3,
        'hidden_features': layer_width,
        'hidden_layers': n_layers,
        'out_features': 1,
        'num_epochs': n_epochs,
        'siren_lrate': siren_lr,
        'final_siren_lrate': final_siren_lr,
        'latent_lrate': latent_lr,
        'final_latent_lrate': final_latent_lr,
        'scheduler_type': scheduler_type,
        'first_omega_0': siren_first_omega_0,
        'hidden_omega_0': siren_hidden_omega_0,
        'latent_dim': siren_latent_dim,
        'step_size': lr_decay_every,
        'gamma': lr_decay_frac,
        'clip_gradient_norm': clip_gradient_norm
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

    if USE_WANDB:
        wandb.finish()


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)

    # network
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--layer_width", type=int, default=32)
    parser.add_argument("--clip_gradient_norm", type=float, default=1.0)
    #positional arguments
    parser.add_argument("--positional_encoding", action='store_true')
    parser.add_argument("--positional_count", type=int, default=10)
    parser.add_argument("--positional_pow_start", type=int, default=-3)
    parser.add_argument("--positional_prepend", action='store_true')
    # siren arguments
    parser.add_argument("--siren_model", action='store_true')
    parser.add_argument("--siren_latent_dim", type=int, default=0)
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
    parser.add_argument("--siren_lr", type=float, default=1e-4)
    parser.add_argument("--final_siren_lr", type=float, default=None)
    parser.add_argument("--latent_lr", type=float, default=1e-2)
    parser.add_argument("--final_latent_lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr_decay_every", type=int, default=None)
    parser.add_argument("--lr_decay_frac", type=float, default=None)
    parser.add_argument('--scheduler_type', type=str, default='none')

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