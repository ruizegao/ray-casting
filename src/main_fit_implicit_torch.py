import torch
import torch.nn as nn
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
from siren_pytorch import SirenNet, SirenWrapper

# imports specific to sdf
import igl, geometry

USE_WANDB = bool(os.environ.get('USE_WANDB', 0))
WANDB_GROUP = None
if USE_WANDB:
    import wandb

# print(plt.style.available)  # uncomment to view the available plot styles
plt.rcParams['text.usetex'] = False  # tex not necessary here and may cause error if not installed

# Set plot style to seaborn white. If these options do not work, don't set the plot style or select from other
# available plot styles.
try:
    plt.style.use("seaborn-white")
except OSError as e:
    plt.style.use("seaborn-v0_8-white")

set_t = {
    'dtype': torch.float64,  # double precision for more accurate training
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

available_activations = [nn.ReLU, nn.ELU, nn.GELU, nn.Sigmoid]  # list of currently supported activation functions

to_numpy = lambda x : x.detach().cpu().numpy()  # converts tensors to numpy arrays

class MainApplicationMethod(Enum):
    """
    1) Default manner of training an implicit surface for a single .obj file
    2) Trains implicit surface for all available .obj files in the Thingi10K dataset
    3) Trains implicit surface for all .obj files in the Meshes Master dataset
    4) Trains implicit surface for all .obj files in the ShapeNetCore dataset; This is the largest dataset of them all
    5) Visualizes the .pth model of a neural implicit network
    """
    Default = 1
    TrainThingi10K = 2
    TrainMeshesMaster = 3
    ShapeNetCore = 4
    Visualize = 5

    @classmethod
    def get(cls, identifier, default_ret=None):
        # Check if the identifier is a valid name
        if isinstance(identifier, str):
            return cls.__members__.get(identifier, default_ret)
        # Check if the identifier is a valid value
        elif isinstance(identifier, int):
            for member in cls:
                if member.value == identifier:
                    return member
            return default_ret
        else:
            raise TypeError("Identifier must be a string (name) or an integer (value)")


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


class Siren(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int, out_features: int,
                 lrate: float, fit_mode: str, outermost_linear: bool=False,
                 first_omega_0: int=30, hidden_omega_0: float=30.,
                 step_size: Optional[int] = None, gamma: Optional[float] = None):
        super().__init__()

        self.model = []
        self.model.append(
            ('0000_SineLayer', SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))
        )

        for i in range(hidden_layers):
            idx_str = f"{i+1:4d}_SineLayer"
            self.model.append(
                (idx_str, SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
            )

        idx_str = f"{hidden_layers + 1:4d}_"
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.model.append((idx_str + "dense", final_linear))
        else:
            self.model.append(
                (idx_str + "SineLayer", SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
            )

        self.model = nn.Sequential(OrderedDict(self.model))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)
        if fit_mode == 'sdf':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif fit_mode == 'occupancy':
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise ValueError(f"fit_mode {fit_mode} is not valid. Select from ['sdf', 'occupancy']")


        # set LR scheduler
        self.scheduler = None
        if step_size is not None and gamma is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size,
                                                       gamma=gamma)

    def forward(self, coords):
        return self.model(coords)

    def forward_with_coords(self, coords):
        coords = coords.clone().detach().requires_grad_(
            True)  # allows to take derivative w.r.t. input
        output = self.model(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.model):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

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

        # clip the estimate
        # y_hat = torch.clip(y_hat, min=-self.sdf_max, max=self.sdf_max)

        # compute the loss
        unweighted_loss = self.loss_fn(y_hat, y)
        loss = (unweighted_loss * weights).mean()

        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def step_eikonal(self, x: Tensor, y: Tensor, weights: Tensor, on_surface_mask: Tensor):
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
            exp_in = -a * (x).abs()
            exp_out = torch.exp(exp_in).squeeze(1)
            return exp_out

        # zero the gradients
        self.optimizer.zero_grad()

        # pass the batch through the model
        output, coords = self.forward_with_coords(x)

        # compute the graident of the output w.r.t. input
        grad_output = _gradient(coords, output)

        # eikonal loss
        eik_loss = torch.linalg.vector_norm(grad_output, dim=1).unsqueeze(1)

        # on surface loss
        surface_output = output[on_surface_mask]
        surface_grad = grad_output[on_surface_mask]
        on_surface_loss = torch.zeros_like(on_surface_mask)
        on_surface_loss[on_surface_mask, :] = torch.linalg.vector_norm(surface_output, dim=1) + (
                    1 - torch.einsum('bi,bi->b', surface_grad, y))

        # off surface loss
        off_surface_mask = torch.logical_not(on_surface_mask)
        off_surface_output = output[off_surface_mask]
        off_surface_loss = torch.zeros_like(output)
        off_surface_loss[off_surface_mask, :] = _psi(off_surface_output)


        # add together for a total loss
        total_loss = c1*eik_loss + c2*on_surface_loss + c3*off_surface_loss

        # compute the loss
        unweighted_loss = self.loss_fn(total_loss, torch.zeros_like(total_loss))
        loss = (unweighted_loss * weights).mean()

        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item()

class PositionalEncodingLayer(nn.Module):
    """
    Manually implemented module that performs the following operations in order to produce
    positional encoding in a higher feature space:

    1) Unflatten the input to be (batches, 3) to (batches, 3, 1)
    2) Multiplies each input by its respective encoding coefficient
    3) [optionally] adds a shift so that every other element is a sin operation
    4) Pass the embedding through a cosine activation function
    5) Flatten the result to be (batches, 3*L(*2)) where (*2) means that the dimension is doubled if shift is enabled
    """

    def __init__(self, L: int, start_pow: int = 0, with_shift: bool = True, prepend: bool = False):
        """
        Initializes a layer that performs positional encoding.
        :param L:
        :param start_pow:
        :param with_shift:
        :param prepend:
        """
        super(PositionalEncodingLayer, self).__init__()
        pow_args = torch.arange(start=start_pow, end=start_pow + L - 1)
        pow_args = torch.concatenate((torch.zeros(1), pow_args))
        pows = torch.pow(2., pow_args)
        coeffs = pows * torch.pi
        self.prepend = prepend

        if with_shift:
            coeffs = coeffs.repeat_interleave(2)
            shift = torch.zeros_like(coeffs)
            shift[0::2] = torch.pi/2
            # reshape so that we can broadcast
            self.register_buffer("coeffs", coeffs.reshape(1, L * 2))
            self.register_buffer("shift", shift.reshape(1, L * 2))
        else:
            # reshape so that we can broadcast
            self.register_buffer("coeffs", coeffs.reshape(1, L))
            self.shift = None  # No shift buffer needed

        # TODO: Unflatten is not supported by CROWN, simply reshape the input
        # will unflatten input from (batches, 3) to (batches, 3, 1)
        self.unflatten = nn.Unflatten(1, (3, 1))
        # will flatten input from (batches, 3, L(*2)) to (batches, 3*L(*2))
        self.flatten = nn.Flatten(start_dim=1)


    def forward(self, x: Tensor):
        # reshape input
        x_reshape = self.unflatten(x)

        # apply encoding
        x_encoded = x_reshape * self.coeffs

        # add the shift so that every other is sin
        if self.shift is not None:
            x_encoded += self.shift

        # apply cos as activation function
        cos_output = torch.cos(x_encoded)
        flatten_output = self.flatten(cos_output)

        # prepend the position if specified
        if self.prepend:
            flatten_output = torch.concatenate((x, flatten_output), dim=1)

        return flatten_output


    @staticmethod
    def compute_output_dim(positional_count: int, with_shift: bool, prepend: bool):
        """
        Returns what the output dimension of this Module would be given these parameters
        """
        output_dim = 3 * positional_count
        if with_shift:
            output_dim *= 2

        if prepend:
            output_dim += 3

        return output_dim

class CubeRootActivation(nn.Module):
    def forward(self, x):
        return torch.sign(x) * torch.abs(x).pow(1/3)

class FitSurfaceModel(nn.Module):
    def __init__(self, lrate: float, fit_mode: str, activation:str='relu', n_layers:int=8,
                 layer_width:int=32, sdf_max: float=0.4,
                 use_positional_encoding: bool = False, positional_count: Optional[int] = None,
                 positional_power_start: Optional[int] = None, positional_prepend: bool = False,
                 with_shift: bool = True, step_size: Optional[int] = None, gamma: Optional[float] = None):
        """
        Constructs a neural network for fitting to an implicit surface. Layers are carefully named as to make it easier
        to convert the network into an .npz file that can be used for ray-casting.
        :param lrate:                       Learning rate
        :param fit_mode:                    If the neural network should be fit for occupancy or sdf
        :param activation:                  Activation function to use at nonlinear layers
        :param n_layers:                    Number of layers to use in the neural network
        :param layer_width:                 Number of neurons per hidden layer
        :param use_positional_encoding:     If True, does positional encoding
        :param positional_count:            New dimension after positional encoding
        :param positional_power_start:      Starting frequency in positional encoding
        :param with_shift:                  If true, doubles encoding size to incorporate sin
        :param step_size:                   Number of epochs before applying gamma in step scheduler
        :param gamma:                       Gamma to apply to LR after step_size epochs
        """
        super(FitSurfaceModel, self).__init__()

        # parse the specified activation
        activation_lower = activation.lower()
        if activation_lower == 'relu':
            activation_fn = nn.ReLU()
        elif activation_lower == 'elu':
            activation_fn = nn.ELU()
        elif activation_lower == 'gelu':
            activation_fn = nn.GELU()
        elif activation_lower == 'sigmoid':
            activation_fn = nn.Sigmoid()
        else:
            raise ValueError("Activation not recognized. If you wish to use a new activation function, "
                             "feel free to add it to the list in the constructor.")
        activation_fn_name = activation_fn.__class__.__name__.lower()

        ## create the network based on the specifications

        # first layers
        start_layer = 0
        layers = []
        mlp_input_dim = 3  # default input of 3D coordinate
        if use_positional_encoding:
            layers.append(
                ('0000_encoding',
                 PositionalEncodingLayer(positional_count, positional_power_start,
                                         with_shift, positional_prepend))
            )
            # MLP will now have start layer idx of 1
            start_layer = 1
            # Get new input dimension of MLP
            mlp_input_dim = PositionalEncodingLayer.compute_output_dim(positional_count,
                                                                       with_shift,
                                                                       positional_prepend)
        layers.extend([
            (f'{start_layer:04d}_dense', nn.Linear(mlp_input_dim, layer_width)),
            (f'{start_layer+1:04d}_{activation_fn_name}', activation_fn)
        ])
        # hidden layers
        for i in range(n_layers - 2):
            layer_count = len(layers)
            layer_count_formatted = f"{layer_count:04d}_"
            layer_count_formatted_plus_one = f"{layer_count + 1:04d}_"
            layers.extend([
                (layer_count_formatted + 'dense', nn.Linear(layer_width, layer_width)),
                (f'{layer_count_formatted_plus_one}{activation_fn_name}', activation_fn)
            ])
        # last layer
        layer_count = len(layers)
        layer_count_formatted = f"{layer_count:04d}_"
        layer_count_formatted_plus_one = f"{layer_count+1:04d}_"
        layers.extend([
            (layer_count_formatted + 'dense', nn.Linear(layer_width, 1)),
            (layer_count_formatted_plus_one + 'tanh', nn.Tanh())
        ])
        # layer_count = len(layers)
        # layer_count_formatted = f"{layer_count:04d}_"
        # layer_count_formatted_plus_one = f"{layer_count + 1:04d}_"
        # layers.extend([
        #     (layer_count_formatted + 'dense', nn.Linear(layer_width, 1)),
        #     (layer_count_formatted_plus_one + 'cbrt', CubeRootActivation())
        # ])
        # set the loss function
        if fit_mode == 'occupancy':
            # We will not apply Sigmoid. The raw logits will be passed to BCE which also applies sigmoid for
            # numerical stability (using the log-sum-exp trick)
            # As a note, we also do not want sigmoid because it can make bounds unnecessarily loose when we don't need
            # the output to be in the range (0, 1). We simply want to classify based on if the logit is >=0 or < 0.
            # Such an output aligns well with the SDF output and requires fewer changes in ray-casting
            # Reduction = 'None' allows us to manually apply weights to the loss to help correct class imbalance
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif fit_mode == 'sdf':
            # Reduction = 'None' but the weights that are passed will be all 1's
            self.loss_fn = nn.L1Loss(reduction='none')
            # self.loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError("fit_mode must be either 'occupancy' or 'sdf'")
        # convert layers to OrderedDict to retain custom layer names
        layer_dict = OrderedDict(layers)
        self.model = nn.Sequential(layer_dict)

        # set optimizer
        self.fit_mode = fit_mode
        self.lr = lrate
        self.sdf_max = sdf_max
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)

        # set LR scheduler
        self.scheduler = None
        if step_size is not None and gamma is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model
        :param x:   (Batch, input size)
        :return:    (Batch, output size)
        """
        return self.model(x)

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
        y_hat = self.forward(x) * self.sdf_max

        # compute the loss
        unweighted_loss = self.loss_fn(y_hat, y)
        loss = (unweighted_loss * weights).mean()

        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item()

def load_net_object(pth_file: str) -> Union[FitSurfaceModel, Siren]:
    """
    A helper function that retrieves a torch network from a .pth file
    :param pth_file:    .pth file to load network parameters and weights from.
    :return:    Network object
    """
    pth_dict = torch.load(pth_file, weights_only=True)
    state_dict = pth_dict["state_dict"]  # weights and biases
    model_params = pth_dict["model_params"]  # rest of the parameters
    is_siren = pth_dict.pop("is_siren", False)
    siren_latent_dim = pth_dict.get("siren_latent_dim", 0)
    if is_siren:
        NetObject = Siren(**model_params)
    elif siren_latent_dim == 0:
        NetObject = FitSurfaceModel(**model_params)  # initialize object
    else:
        NetObject = SirenWrapper(**model_params)
    NetObject.load_state_dict(state_dict)  # load in weights and biases
    NetObject.eval()  # set to evaluation mode

    return NetObject

class SampleDataset(Dataset):
    def __init__(
            self,
            mesh_input_file: str,
            fit_mode: str,
            n_samples: int,
            sdf_max: float,
            sample_weight_beta: float,
            sample_ambient_range: float,
            sample_221: bool = False,
            show_sample_221: bool = False,
            verbose=False
    ):
        """
        Creates a dataset of samples given a mesh input file of type .obj. Target label varies based on what type of
        implicit surface fitting is being done.
        :param mesh_input_file:         Mesh input to fit
        :param fit_mode:                Fit mode for the mesh with options ['occupancy', 'sdf', 'tanh']
        :param n_samples:               Number of samples to create
        :param sample_weight_beta:      The sample weight beta factor
        :param sample_ambient_range:
        :param sample_221:              Use 2-2-1 sampling
        :param verbose:                 If true, prints additional info during dataset creation
        """

        if verbose:
            print(f"Loading mesh {mesh_input_file}")
        V, F = igl.read_triangle_mesh(mesh_input_file)
        V = torch.from_numpy(V)
        F = torch.from_numpy(F)
        # preprocess (center and scale)
        if verbose:
            print("Normalizing position array")
        V = geometry.normalize_positions(V, method='bbox')


        if verbose:
            print(f"Collecting geometry samples. Is using sample_221? {sample_221}")
        if sample_221:
            samp, samp_SDF = geometry.sample_221(V, F, n_samples, sample_ambient_range, sdf_max, show_sample_221)
        else:
            samp, samp_SDF = geometry.sample_mesh_importance(V, F, n_samples, beta=sample_weight_beta,
                                                             ambient_range=sample_ambient_range, sdf_max=sdf_max,
                                                             show_surface=show_sample_221)

        if verbose:
            print(f"Formatting labels")
        if fit_mode == 'occupancy':
            # apply label and calculate sample weight to correct class imbalance
            samp_target = (samp_SDF > 0) * 1.0
            n_pos = np.sum(samp_target > 0)
            n_neg = samp_target.shape[0] - n_pos
            w_pos = n_neg / (n_pos + n_neg)
            w_neg = n_pos / (n_pos + n_neg)
            samp_weight = np.where(samp_target > 0, w_pos, w_neg)
        elif fit_mode in ['sdf', 'tanh']:
            # apply label and give all weights equal importance
            # since this is regression not classification based
            samp_target = samp_SDF
            samp_weight = np.ones_like(samp_target)
        else:
            raise ValueError(f"Fit mode {fit_mode} not recognized. Please select from ['occupancy', 'sdf', 'tanh'].")
        # in the original implicit file, the samp_weight array is not used at all so disregard it here as well

        # save inputs and labels
        if verbose:
            print(f"Saving samples and labels to the dataset")
        self.x = torch.from_numpy(samp)  # shape (n_samples, 3)
        self.y = torch.from_numpy(samp_target).reshape(n_samples, 1)
        self.weights = torch.from_numpy(samp_weight).reshape(n_samples, 1)

    def __len__(self) -> int:
            return len(self.x)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        return self.x[idx], self.y[idx], self.weights[idx]

def batch_count_correct(NetObject: Union[FitSurfaceModel, Siren], batch_x: Tensor, batch_y: Tensor,
                        fit_mode: str) -> Tensor:
    """
    For some batch of inputs and labels, return the number of predictions whose sign is correct.
    :param NetObject:   Neural network object to evaluate
    :param batch_x:     Batch of inputs
    :param batch_y:     Batch of labels
    :return:            Number of predictions whose sign is correct
    """
    prediction = NetObject.forward(batch_x)
    if fit_mode in 'occupancy':
        # labels are probabilities, they must be corrected
        is_correct_sign = torch.sign(prediction) == torch.sign(batch_y - 0.5)
    elif fit_mode == 'sdf':
        is_correct_sign = torch.sign(prediction) == torch.sign(batch_y)
    else:
        raise ValueError("fit_mode must be either 'occupancy' or 'sdf'")

    current_count = is_correct_sign.to(dtype=int).sum()
    return current_count


def fit_model(
        NetObject: Union[FitSurfaceModel, Siren],
        train_loader: DataLoader,
        fit_mode: str,
        epochs: int
) -> Tuple[list[float], list[int], list[float], Union[FitSurfaceModel, Siren]]:
    """
    Given a neurol network and train loader, fit the neural network to the training dataset and record the losses.
    The training heuristics that are returned are:

    * `losses` -- losses per epoch
    * `correct_counts` -- number of predictions that have predicted the correct sign
    * `correct_fracs` -- fraction of predictions that have predicted the correct sign

    :param NetObject:       Neural network object to train
    :param train_loader:    Training dataset
    :param fit_mode:        Neural network fitting mode (occupancy or sdf)
    :param epochs:          Number of epochs to run
    :return:                Training heuristics and trained `NetObject`
    """

    global USE_WANDB

    # send to device
    NetObject = NetObject.to(**set_t)

    # train and record losses
    losses, correct_counts, correct_fracs = [], [], []
    n_correct = 0
    n_total = 0
    epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch_x, batch_y, batch_weight = batch
            batch_x = batch_x.to(**set_t)
            batch_y = batch_y.to(**set_t)
            batch_weight = batch_weight.to(**set_t)
            n_total += len(batch_x)
            curr_epoch_loss = NetObject.step(batch_x, batch_y, batch_weight)
            epoch_loss += curr_epoch_loss
            with torch.no_grad():
                correct_count = batch_count_correct(NetObject, batch_x, batch_y, fit_mode).item()
                n_correct += correct_count
                correct_counts.append(correct_count)

        # get the current learning rate
        if NetObject.scheduler is not None:
            NetObject.scheduler.step()
            current_lr = NetObject.scheduler.get_last_lr()[0]
        else:
            current_lr = NetObject.lr
        # calculate the fraction of correctly predicted signs
        frac_correct= n_correct / n_total
        correct_fracs.append(frac_correct)
        # calculate the epoch loss and update progress bar
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        epoch_details = {
                'Epoch Loss': epoch_loss,
                'Correct Sign': f'{100*frac_correct:.2f}%',
                'Learning Rate': current_lr,
            }
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix(epoch_details)
        if USE_WANDB:
            epoch_details.update({'Correct Sign': 100 * frac_correct})
            wandb.log(epoch_details)

    # return metrics and trained network
    return losses, correct_counts, correct_fracs, NetObject

def plot_training_metrics(losses: list[float], correct_fracs: list[float], save_path: Optional[str] = None, display: bool = False):
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title("Training Loss")
    ax1.grid()

    ax2.plot(correct_fracs)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correct Sign %')
    ax2.set_title("Number of Correct Sign Predictions")
    ax2.grid()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if display:
        plt.show()
    else:
        plt.close()

def save_to_npz(NetObject: Union[FitSurfaceModel, Siren], npz_path: str, verbose: bool = False):
    """
    Saves the Torch model as a .npz file that can be loaded in by the other ray tracing scripts. Runs in 3 stages:

    1) Get all the optimizable layers which are simply the linear layers and format them
    2) Get all the activation functions and add them to the npz dictionary as well
    3) Finally, add a 'squeeze_last' parameter as the ray-tracing scripts rely on this parameter.

    :param NetObject:   Neural network object to save
    :param npz_path:    .npz file path to save the network to
    :param verbose:     If true, prints additional logging information
    :return:
    """
    npz_dict = {}  # holds network architecture
    if verbose:
        print("Adding optimizable parameters to the npz dictionary")
    for name, param in NetObject.named_parameters():
        split_name = name.split('.')
        new_base_name = split_name[1].replace('_', '.') + '.'
        is_weight = split_name[2] == 'weight'
        weight_name = 'A' if is_weight else 'b'
        new_key = new_base_name + weight_name

        # convert parameter to numpy array
        # if parameter is a weight, it must be permuted
        if len(param.shape) == 2:
            np_param = to_numpy(param.permute(1, 0))
        else:
            np_param = to_numpy(param)
        npz_dict[new_key] = np_param
        if verbose:
            print(f"Parameter Name: {name}")
            print(f"Parameter Value: {param}")
            print("-" * 30)

    if verbose:
        print("Adding activation functions to the npz dictionary")
    available_activation_names = [n().__class__.__name__.lower() for n in available_activations]
    for name, layer in NetObject.model._modules.items():
        split_name = name.split('_')
        base_name = split_name[1]
        if base_name in available_activation_names:
            new_key = name.replace('_', '.') + '._'
            np_param = np.empty(0)
            npz_dict[new_key] = np_param
        if verbose:
            print(f"Module Name: {name}")
            print(f"Module Value: {layer}")
            print("-" * 30)

    squeeze_last_idx = len(NetObject.model._modules.keys())
    if verbose:
        print(f"Adding squeeze_last at layer index {squeeze_last_idx}")
    squeeze_last_idx_formatted = f"{squeeze_last_idx:04d}.squeeze_last._"
    npz_dict[squeeze_last_idx_formatted] = np.empty(0)

    if verbose:
        print(f"Saving network in .npz format with path {npz_path} \nand dictionary with keys \n{npz_dict.keys()}")
    np.savez(npz_path, **npz_dict)

def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    # Program mode
    parser.add_argument("--program_mode", type=str, default=MainApplicationMethod.Default.name)

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

def main(args: dict):
    global USE_WANDB, WANDB_GROUP

    print(f"Torch Settings: {set_t}")

    ##  unpack arguments

    # Build arguments
    program_mode = args["program_mode"]
    input_file = args["input_file"]
    output_file = args["output_file"]
    if input_file is None or output_file is None:
        raise ValueError("input_file and/or output_file is None")
    # network
    activation = args["activation"]
    n_layers = args["n_layers"]
    layer_width = args["layer_width"]
    # positional encoding params
    positional_encoding = args["positional_encoding"]
    positional_count = args["positional_count"]
    positional_pow_start = args["positional_pow_start"]
    positional_prepend = args["positional_prepend"]
    # siren params
    siren_model = args["siren_model"]
    siren_latent_dim = args["siren_latent_dim"]
    siren_outermost_linear = args["siren_outermost_linear"]
    siren_first_omega_0 = args["siren_first_omega_0"]
    siren_hidden_omega_0 = args["siren_hidden_omega_0"]

    # loss / data
    fit_mode = args["fit_mode"]
    n_epochs = args["n_epochs"]
    n_samples = args["n_samples"]
    sample_ambient_range = args["sample_ambient_range"]
    sample_weight_beta = args["sample_weight_beta"]
    sample_221 = args["sample_221"]
    show_sample_221 = args["show_sample_221"]
    sdf_max = args["sdf_max"]
    # training
    lr = args["lr"]
    batch_size = args["batch_size"]
    lr_decay_every = args["lr_decay_every"]
    lr_decay_frac = args["lr_decay_frac"]
    # general options
    verbose = args["verbose"]
    display_plots = args["display_plots"]

    print(f"Program Configuration: {args}")

    if USE_WANDB:
        if WANDB_GROUP is None:
            WANDB_GROUP = program_mode + '_' + wandb.util.generate_id()
        uniq_id = WANDB_GROUP.split('_')[-1]
        file_name = input_file.split('/')[-1].split('.obj')[0] + '_' + uniq_id

        # start a new wandb run to track this script
        tags = [fit_mode, program_mode]
        if siren_model:
            tags += ['siren']
        elif positional_encoding:
            tags += ['positional_encoding']
        wandb.init(
            # set the wandb project and name where this run will be logged
            project="main_fit_implicit_torch",
            name=file_name,
            # track hyperparameters and run metadata
            config=args_dict,
            # set group
            group=WANDB_GROUP,
            # set tags
            tags=tags
        )

    print(f"WANDB ENABLED: {USE_WANDB} | WANDB GROUP: {WANDB_GROUP}")

    # validate some inputs
    if activation not in ['relu', 'elu', 'gelu', 'cos']:
        raise ValueError("unrecognized activation")
    if fit_mode not in ['occupancy', 'sdf']:
        raise ValueError("unrecognized activation")
    if not output_file.endswith('.npz'):
        raise ValueError("output file should end with .npz")

    # initialize the network
    if not siren_model:
        model_params = {
            'lrate': lr,
            'fit_mode': fit_mode,
            'activation': activation,
            'n_layers': n_layers,
            'layer_width': layer_width,
            'sdf_max': sdf_max,
            'use_positional_encoding': positional_encoding,
            'positional_count': positional_count,
            'positional_power_start': positional_pow_start,
            'positional_prepend': positional_prepend,
            'with_shift': True,
            'step_size': lr_decay_every,
            'gamma': lr_decay_frac,
        }
        NetObject = FitSurfaceModel(**model_params)
    elif siren_latent_dim == 0:
        model_params = {
            'in_features': 3,
            'hidden_features': layer_width,
            'hidden_layers': n_layers,
            'out_features': 1,
            'lrate': lr,
            'fit_mode': fit_mode,
            'outermost_linear': siren_outermost_linear,
            'first_omega_0': siren_first_omega_0,
            'hidden_omega_0': siren_hidden_omega_0,
            'step_size': lr_decay_every,
            'gamma': lr_decay_frac,
        }
        NetObject = Siren(**model_params)
        sample_ambient_range = 1.
        warn("'sample_ambient_range' was changed to 1. as inputs outside of the range [-1, 1] "
             "may cause training instability for a SIREN model")
    else:
        print(f"Using Siren Wrapper instead of Sirent with Latent dim {siren_latent_dim}")
        if siren_hidden_omega_0 != 1.:
            warn(f"'siren_hidden_omega_0' is {siren_hidden_omega_0}, for siren with latents, it is recommended that"
                 f"this value is 1.")
        siren_net_params = {
            'dim_in': 3,
            'dim_hidden': layer_width,
            'dim_out': 1,
            'num_layers': n_layers,
            'w0_initial': siren_first_omega_0,
            'w0': siren_hidden_omega_0,
            'use_bias': True,
            'sdf_max': sdf_max,
            'final_activation': nn.Tanh(),
            'dropout': 0,
        }
        net = SirenNet(**siren_net_params)

        model_params = {
            'net': net,
            'lrate': lr,
            'fit_mode': fit_mode,
            'latent_dim': siren_latent_dim,
            'step_size': lr_decay_every,
            'gamma': lr_decay_frac,
        }
        NetObject = SirenWrapper(**model_params)
        sample_ambient_range = 1.
        warn("'sample_ambient_range' was changed to 1. as inputs outside of the range [-1, 1] "
             "may cause training instability for a SIREN model")

    # initialize the dataset
    dataset_pararms = {
        'mesh_input_file': input_file,
        'fit_mode': fit_mode,
        'n_samples': n_samples,
        'sample_weight_beta': sample_weight_beta,
        'sample_ambient_range': 1. if siren_model else sample_ambient_range,
        'sample_221': sample_221,
        'show_sample_221': show_sample_221,
        'sdf_max': sdf_max,
        'verbose': verbose
    }
    train_dataset = SampleDataset(**dataset_pararms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # train the neural network
    losses, correct_counts, correct_fracs, NetObject = fit_model(NetObject, train_loader, fit_mode, n_epochs)
    NetObject.eval()  # set to evaluation mode

    # save the neural network in Torch format
    pth_file = output_file.replace('.npz', '.pth')
    print(f"Saving model to {pth_file}...")
    pth_dict = {
        "state_dict": NetObject.state_dict(),
        "model_params": model_params,
        "is_siren": siren_model,
    }
    torch.save(pth_dict, pth_file)

    # display results
    plt_file = output_file.replace('.npz', '.png')
    plot_training_metrics(losses, correct_fracs, plt_file, display_plots)

    # save the neural network in .npz format
    save_to_npz(NetObject, output_file, verbose)

    if USE_WANDB:
        wandb.finish()

def get_filename_descriptor(args: dict) -> str:
    """
    Formats the filename of the output file to be more descriptive with regard to the training
    parameters and architecture.
    :param args:
    :return:
    """
    is_siren = args['siren_model']
    first_omega_0 = args['siren_first_omega_0']
    hidden_omega_0 = args['siren_hidden_omega_0']
    outermost_linear = args['siren_outermost_linear']
    activation, nlayers, layerwidth, fit_mode = args['activation'], args['n_layers'], args[
        'layer_width'], args['fit_mode']
    pos, positional_count, positional_pow_start = args['positional_encoding'], args[
        'positional_count'], args['positional_pow_start']

    if is_siren:
        descriptor = f"_nlayers_{nlayers}_layerwidth_{layerwidth}_fitmode_{fit_mode}"
        descriptor += f"_siren_first_omega_{first_omega_0}_hidden_omega_{hidden_omega_0}_out_linear_{outermost_linear}"
    else:
        descriptor = f"_activation_{activation}_nlayers_{nlayers}_layerwidth_{layerwidth}_fitmode_{fit_mode}"
        if pos:
            descriptor += f"_pos_L_{positional_count}_pow_start_{positional_pow_start}"

    return descriptor

def TrainThingi10K_main(args: dict):
    """
    Main program for training implicit surfaces on the entire Thingi10K dataset
    :param args: Default main program arguments/configurations
    :return:
    """

    row_names = ['Obj Filename', 'Training Success', 'Training Failure']

    # TODO: Would be nice to load in a csv file that describes what objects have previously
    # check_csv_table: Optional['str'] = args.pop('check_csv_table', None)
    # prior_directory_dict = makehash()
    # if check_csv_table is not None:
    #     with open(check_csv_table, mode='r') as csvfile:
    #         reader = csv.DictReader(csvfile)
    #         for row in reader:
    #             row_entries = [row[i] for i in row_names]
    #             filename, success, error = row_entries
    #             if success == 'y' and error == 'n':
    #                 exists = True
    #             elif success == 'n' and error == 'y':
    #                 exists = False
    #             else:
    #                 raise ValueError("File cannot have been trained successfully AND had an error.")
    #             prior_directory_dict[filename]['exists'] = exists

    input_directory = "Thingi10K/raw_meshes/"
    output_directory = "sample_inputs/Thingi10K_inputs/"
    file_names = [f for f in os.listdir(input_directory) if f.endswith('.obj')]
    os.makedirs(output_directory, exist_ok=True)
    descriptor = get_filename_descriptor(args)
    input_files = [input_directory + f for f in file_names]
    output_files = [output_directory + f.replace(".obj", descriptor + ".npz") for f in file_names]

    csv_file, success, error = [], [], []

    for i, in_file, out_file in zip([p for p in range(len(input_files))], input_files, output_files):
        args.update({
            'input_file': in_file,
            'output_file': out_file,
        })

        # update table
        csv_file.append(file_names[i])

        # See above TODO
        # # if has been trained before, continue
        # if check_csv_table:
        #     exists = prior_directory_dict[file_names[i]].get('exists', False)
        #     if exists:
        #         success.append('y')
        #         error.append('n')
        #         continue

        try:
            main(args)
            success.append('y')
            error.append('n')
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            warn(
                f"Could not fit implicit surface to {in_file}. Received exception:\n{exc_type}, {fname}, {exc_tb.tb_lineno}\n{str(e)}",
            stacklevel=1)

            success.append('n')
            error.append('y')

    csv_path = output_directory + 'summary.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_names)

        for row in zip(csv_file, success, error):
            writer.writerow(row)

    # read the csv and print to the console
    with open(csv_path, mode='r') as file:
        table = from_csv(file)  # renders the table in pretty print format
        print(table)



def TrainMeshesMaster_main(args: dict):
    """
    Main program for training implicit surfaces on the entire Meshes Master dataset
    :param args: Default main program arguments/configurations
    :return:
    """

    row_names = ['Directory', 'Obj Filename', 'Training Success', 'Training Failure']

    # TODO: Would be nice to load in a csv file that describes what objects have previously
    # check_csv_table: Optional['str'] = args.pop('check_csv_table', None)
    # prior_directory_dict = makehash()
    # if check_csv_table is not None:
    #     with open(check_csv_table, mode='r') as csvfile:
    #         reader = csv.DictReader(csvfile)
    #         for row in reader:
    #             row_entries = [row[i] for i in row_names]
    #             directory, filename, success, error = row_entries
    #             if success == 'y' and error == 'n':
    #                 exists = True
    #             elif success == 'n' and error == 'y':
    #                 exists = False
    #             else:
    #                 raise ValueError("File cannot have been trained successfully AND had an error.")
    #             prior_directory_dict[directory][filename]['exists'] = exists

    input_directory = "meshes-master/objects/"
    sub_names = [name + '/' for name in os.listdir(input_directory)
                      if os.path.isdir(os.path.join(input_directory, name))]
    subdirectories = [input_directory + name for name in sub_names]
    output_directory = "sample_inputs/meshes-master_inputs/"
    os.makedirs(output_directory, exist_ok=True)
    descriptor = get_filename_descriptor(args)

    # create a table that views the output
    csv_subdir, csv_file, success, error = [], [], [], []

    for i, sub in enumerate(subdirectories):
        file_names = [f for f in os.listdir(sub) if f.endswith('.obj')]
        input_files = [sub + f for f in file_names]
        output_files = [output_directory + f.replace(".obj", descriptor + ".npz") for f in file_names]
        for j, in_file, out_file in zip([p for p in range(len(input_files))], input_files, output_files):
            args.update({
                'input_file': in_file,
                'output_file': out_file,
            })

            # update table
            csv_subdir.append(sub_names[i])
            csv_file.append(file_names[j])

            # See above TODO
            # # if has been trained before, continue
            # if check_csv_table:
            #     exists = prior_directory_dict[sub_names[i]][file_names[j]].get('exists', False)
            #     if exists:
            #         success.append('y')
            #         error.append('n')
            #         continue

            # try:
            main(args)
            success.append('y')
            error.append('n')
            # except Exception as e:
            #     exc_type, exc_obj, exc_tb = sys.exc_info()
            #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            #     warn(f"Could not fit implicit surface to {in_file}. Received exception:\n{exc_type}, {fname}, {exc_tb.tb_lineno}\n{str(e)}",
            #          stacklevel=1)
            #     success.append('n')
            #     error.append('y')

    csv_path = output_directory + 'summary.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_names)

        for row in zip(csv_subdir, csv_file, success, error):
            writer.writerow(row)

    # read the csv and print to the console
    with open(csv_path, mode='r') as file:
        table = from_csv(file)  # renders the table in pretty print format
        print(table)


def ShapeNetCore_main(args: dict):
    """
    Main program for training implicit surfaces on the entire ShapeNetCore dataset
    :param args: Default main program arguments/configurations
    :return:
    """

    row_names = ['Directory', 'Sub Directory', 'Obj Filename', 'Training Success', 'Training Failure']

    # TODO: Would be nice to load in a csv file that describes what objects have previously
    # been rendered in the data successfully/unsuccessfully
    # check_csv_table: Optional['str'] = args.pop('check_csv_table', None)
    # prior_directory_dict = makehash()
    # if check_csv_table is not None:
    #     with open(check_csv_table, mode='r') as csvfile:
    #         reader = csv.DictReader(csvfile)
    #         for row in reader:
    #             row_entries = [row[i] for i in row_names]
    #             directory, subdirectory, filename, success, error = row_entries
    #             if success == 'y' and error == 'n':
    #                 exists = True
    #             elif success == 'n' and error == 'y':
    #                 exists = False
    #             else:
    #                 raise ValueError("File cannot have been trained successfully AND had an error.")
    #             prior_directory_dict[directory][subdirectory][filename]['exists'] = exists

    input_directory = "ShapeNetCore/"
    sub_names = [name + '/' for name in os.listdir(input_directory)
                      if os.path.isdir(os.path.join(input_directory, name))]
    subdirectories = [input_directory + name for name in sub_names]
    output_directory = "sample_inputs/ShapeNetCore_inputs/"
    descriptor = get_filename_descriptor(args)
    # create a table that views the output
    csv_subdir, csv_subbdir, csv_file, success, error = [], [], [], [], []

    for i, sub in enumerate(subdirectories):

        subb_names = [name + '/models/' for name in os.listdir(sub)
                      if os.path.isdir(os.path.join(sub, name))]
        subbdirectories = [sub + name for name in subb_names]

        for j, subb in enumerate(subbdirectories):

            # ShapeNet has too many objects, organize the output into directories
            curr_output_dir = output_directory + sub_names[i] + subb_names[j]
            os.makedirs(curr_output_dir, exist_ok=True)

            file_names = [f for f in os.listdir(subb) if f.endswith('.obj')]
            input_files = [subb + f for f in file_names]
            output_files = [curr_output_dir + f.replace(".obj", descriptor + ".npz") for f in file_names]
            for k, in_file, out_file in zip([p for p in range(len(input_files))], input_files, output_files):
                args.update({
                    'input_file': in_file,
                    'output_file': out_file,
                })

                # update table
                csv_subdir.append(sub_names[i])
                csv_subbdir.append(subb_names[j])
                csv_file.append(file_names[k])

                # See above TODO
                # # if has been trained before, continue
                # if check_csv_table:
                #     exists = prior_directory_dict[sub_names[i]][subb_names[j]][file_names[k]].get('exists', False)
                #     if exists:
                #         success.append('y')
                #         error.append('n')
                #         continue

                try:
                    main(args)
                    success.append('y')
                    error.append('n')
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    warn(
                        f"Could not fit implicit surface to {in_file}. Received exception:\n{exc_type}, {fname}, {exc_tb.tb_lineno}\n{str(e)}",
                    stacklevel=1)
                    success.append('n')
                    error.append('y')

    csv_path = output_directory + 'summary.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_names)

        for row in zip(csv_subdir, csv_subbdir, csv_file, success, error):
            writer.writerow(row)

    # read the csv and print to the console
    with open(csv_path, mode='r') as file:
        table = from_csv(file)  # renders the table in pretty print format
        print(table)

def Visualize_main(args: dict):
    import polyscope as ps
    from skimage import measure

    ps.init()
    # parse user arguments
    input_file = args["input_file"]
    NetObject = load_net_object(input_file)

    # check that the network runs
    sample = torch.rand((5, 3))
    output = NetObject(sample)
    print(output)

    grid_res = 128
    ax_coords = torch.linspace(-1., 1., grid_res)
    grid_x, grid_y, grid_z = torch.meshgrid(ax_coords, ax_coords, ax_coords, indexing='ij')
    grid = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=-1)
    delta = (grid[1, 2] - grid[0, 2]).item()
    sdf_vals = NetObject(grid)
    sdf_vals = sdf_vals.reshape(grid_res, grid_res, grid_res)
    bbox_min = grid[0, :]
    verts, faces, normals, values = measure.marching_cubes(to_numpy(sdf_vals), level=0.,
                                                           spacing=(delta, delta, delta))
    verts = torch.from_numpy(verts).to('cpu')
    verts = verts + bbox_min[None, :]
    ps.register_surface_mesh("coarse shape preview", to_numpy(verts), faces)

    print(
        "REMEMBER: All routines will be slow on the first invocation due to JAX kernel compilation. Subsequent calls will be fast.")

    # Hand off control to the main callback
    ps.show()


if __name__ == '__main__':
    # parse user arguments
    args_dict = parse_args()
    program_mode_name = args_dict['program_mode']
    program_mode = MainApplicationMethod.get(program_mode_name, None)

    # run the specified program
    if program_mode == MainApplicationMethod.Default:
        main(args_dict)
    elif program_mode == MainApplicationMethod.TrainThingi10K:
        TrainThingi10K_main(args_dict)
    elif program_mode == MainApplicationMethod.TrainMeshesMaster:
        TrainMeshesMaster_main(args_dict)
    elif program_mode == MainApplicationMethod.ShapeNetCore:
        ShapeNetCore_main(args_dict)
    elif program_mode == MainApplicationMethod.Visualize:
        Visualize_main(args_dict)
    else:
        raise ValueError(f"Invalid program_mode of {program_mode_name}")
