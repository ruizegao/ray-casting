"""
Defines the different neural network architectures that are commonly used for fitting onto SDF or occupancy
based functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from functools import partial
from neural_building_blocks import *

set_t = {
    'dtype': torch.float32,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

to_numpy = lambda x : x.detach().cpu().numpy()

### Custom LR Schedulers
def linear_decay(epoch, initial_lr, final_lr, total_epochs, last_decay):
    """

    :param epoch:
    :param initial_lr:
    :param final_lr:
    :param total_epochs:
    :param last_decay:
    :return:
    """
    # FIXME: Currently hard-coded to a tailored configuration that shows stable convergence. The parameters should
    # be modified instead of hard-coded.
    if epoch < 5000:
        total_epochs = 5000
        return 1 - epoch / total_epochs * (1 - final_lr / initial_lr)
    else:
        return last_decay

class MLP(nn.Module):
    """
    A regular MLP neural network for fitting onto an SDF. Such networks typically use activations such as ReLU, GeLU,
    elu, and tanh.
    """
    def __init__(self, input_dim: int, lrate: float, fit_mode: str, activation:str='relu', n_layers:int=8,
                 layer_width:int=32, sdf_max: float = 1.0, optimizer: str = 'adam',
                 use_positional_encoding: bool = False, positional_count: Optional[int] = None,
                 positional_power_start: Optional[int] = None, positional_prepend: bool = False,
                 with_shift: bool = True, step_size: Optional[int] = None, gamma: Optional[float] = None,
                 weight_decay: Union[float, int] = 0):
        """
        Constructs a neural network for fitting to an implicit surface. Layers are carefully named as to make it easier
        to convert the network into an .npz file that can be used for ray-casting.
        :param input_dim:                   The input dimension of the network
        :param lrate:                       Learning rate
        :param fit_mode:                    If the neural network should be fit for occupancy or sdf
        :param activation:                  Activation function to use at nonlinear layers
        :param n_layers:                    Number of layers to use in the neural network
        :param layer_width:                 Number of neurons per hidden layer
        :param use_positional_encoding:     If True, does positional encoding
        :param positional_count:            New dimension after positional encoding
        :param positional_power_start:      Starting frequency in positional encoding
        :param positional_prepend:          Prepends the input coordinate to the output of the positional encoding layer
        :param with_shift:                  If true, doubles encoding size to incorporate sin
        :param step_size:                   Number of epochs before applying gamma in step scheduler
        :param gamma:                       Gamma to apply to LR after step_size epochs
        :param weight_decay:                Weight decay to use for training.
        """
        super(MLP, self).__init__()

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
        mlp_input_dim = input_dim  # default input of 3D coordinate
        if use_positional_encoding:
            # prepend a positional encoding layer to the network
            layers.append(
                ('0000_encoding',
                 PositionalEncodingLayer(input_dim, positional_count, positional_power_start,
                                         with_shift, positional_prepend))
            )
            # MLP will now have start layer idx of 1
            start_layer = 1
            # Get new input dimension of MLP from the output of the positional encoding layer
            mlp_input_dim = PositionalEncodingLayer.compute_output_dim(input_dim, positional_count, with_shift,
                                                                       positional_prepend)
        layers.extend([
            (f'{start_layer:04d}_dense', nn.Linear(mlp_input_dim, layer_width)),
            (f'{start_layer+1:04d}_{activation_fn_name}', activation_fn)
        ])
        # creates all hidden layers
        for i in range(n_layers - 2):
            layer_count = len(layers)
            layer_count_formatted = f"{layer_count:04d}_"
            layer_count_formatted_plus_one = f"{layer_count + 1:04d}_"
            layers.extend([
                (layer_count_formatted + 'dense', nn.Linear(layer_width, layer_width)),
                (f'{layer_count_formatted_plus_one}{activation_fn_name}', activation_fn)
            ])
        # create the last layer
        layer_count = len(layers)
        layer_count_formatted = f"{layer_count:04d}_"
        layer_count_formatted_plus_one = f"{layer_count+1:04d}_"
        layers.extend([
            (layer_count_formatted + 'dense', nn.Linear(layer_width, 1)),
            (layer_count_formatted_plus_one + 'tanh', nn.Tanh())
        ])
        # set the loss function
        if fit_mode == 'occupancy':
            # We will not apply Sigmoid. The raw logits will be passed to BCE which also applies sigmoid for
            # numerical stability (using the log-sum-exp trick)
            # As a note, we also do not want sigmoid because it can make bounds unnecessarily loose when we don't need
            # the output to be in the range (0, 1). We simply want to classify based on if the logit is >=0 or < 0.
            # Such an output aligns well with the SDF output and requires fewer changes in ray-casting
            # Reduction = 'None' allows us to manually apply weights to the loss to help correct class imbalance
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            sdf_max = 1.0
        elif fit_mode == 'sdf':
            # Reduction = 'None' but the weights that are passed will be all 1's
            self.loss_fn = nn.L1Loss(reduction='none')
            # self.loss_fn = nn.MSELoss(reduction='none')
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
        optimizer = optimizer.lower()
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lrate, weight_decay=weight_decay)
        elif optimizer == 'lbfgs':
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=lrate)
        else:
            raise ValueError(f"Optimizer {optimizer} not recognized.")

        # set LR scheduler
        self.scheduler = None
        if step_size is not None and gamma is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif gamma is not None:
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model
        :param x:   (Batch, input size)
        :return:    (Batch, output size)
        """
        return self.model(x)

    def forward_with_coords(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """

        Before the forward pass, clone the input and enable its gradient. Returning this cloned input allows the
        output of the network to be differentiated w.r.t. the input.

        :param x: (batches, input_dim)
        :return:
        """
        x = x.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        output = self.forward(x)

        return output, x

    def step(self, x: Tensor, y: Tensor, weights: Tensor) -> float:
        """
        Returns the loss of a single forward pass
        :param x:       (Batch, input size)
        :param y:       (Batch, output size)
        :param weights: (Batch, input size), weights to apply to input samples to correct class imbalance
        :return:        loss
        """

        if isinstance(self.optimizer, optim.LBFGS):
            loss = self.optimizer.step(lambda: self._step_closure(x, y, weights))
        else:
            loss = self._step_closure(x, y, weights)
            self.optimizer.step()

        return loss.item()

    def _step_closure(self, x: Tensor, y: Tensor, weights: Tensor):
        """
        The actual step optimization is placed into this closure method to enable
        support with the LBFGS optimizer.
        :param x:
        :param y:
        :param weights:
        :return:
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

        return loss

class Siren(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int, out_features: int,
                 siren_lrate: float, latent_lrate: float, num_epochs: int,
                 final_siren_lrate: Optional[float] = None, final_latent_lrate: Optional[float] = None,
                 first_omega_0: int = 30, hidden_omega_0: float = 30., latent_dim: int = 0,
                 step_size: Optional[int] = None, gamma: Optional[float] = None,
                 c1: float = 5e1, c2: float = 3e3, c3: float = 1e2,
                 clip_gradient_norm: Optional[float] = None, scheduler_type: str = 'none'):
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
        :param c1:                  First penalization parameter for Eikonal loss function (reference Siren paper for more details)
        :param c2:                  Second penalization parameter for Eikonal loss function (reference Siren paper for more details)
        :param c3:                  Third penalization parameter for Eikonal loss function (reference Siren paper for more details)
        :param clip_gradient_norm:  Max norm to clip model gradients. Helps with stabilization when using latent variables.
        """
        super().__init__()
        if final_siren_lrate is None: final_siren_lrate = siren_lrate
        if final_latent_lrate is None: final_latent_lrate = latent_lrate
        self.hidden_layers = hidden_layers
        self.clip_gradient_norm = clip_gradient_norm
        self.latent, self.modulator = None, None
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.has_latent = latent_dim > 0
        # optimizable parameters
        self.opt_latent_parameters = []
        self.opt_siren_parameters = []

        # If using modulation, instantiate a modulator network and an optimizable latent tensor
        # The modulator network and latent tensor are optimized separately from the Siren network for more
        # fine-grained control
        if latent_dim > 0:
            self.modulator = Modulator(
                dim_in=latent_dim,
                dim_hidden=hidden_features,
                num_layers=hidden_layers
            )
            # initialize latent input to the modulator network which will also be optimizable
            self.latent = nn.Parameter(torch.zeros(latent_dim).normal_(0, 1e-2))
            # append all optimizable parameters in the latent input and modulator network
            self.opt_latent_parameters.extend([
                self.latent,
                *self.modulator.parameters(),
            ])

        # append first layer and all hidden layers
        self.model = []
        for i in range(hidden_layers):
            idx_str = f"{i:4d}_SineLayer"
            is_first = i == 0
            omega_0 = first_omega_0 if is_first else hidden_omega_0
            input_dim = in_features if is_first else hidden_features
            self.model.append(
                (idx_str, SineLayer(input_dim, hidden_features,
                                    is_first=is_first, omega_0=omega_0))
            )

        # append last layer
        self.model.append(
            ("LastLayer", nn.Sequential(nn.Linear(hidden_features, out_features), nn.Tanh()))
        )

        # ModuleDict makes it easier to get layers by name
        self.model = nn.ModuleDict(OrderedDict(self.model))
        # get all Siren optimizable parameters as a list for the Siren Adam Optimizer
        for l in self.model.values():
            self.opt_siren_parameters.extend(l.parameters())
        # if using modulation, initialize its optimizer and save its learning rate
        if self.has_latent:
            self.latent_optimizer = optim.Adam(self.opt_latent_parameters, lr=latent_lrate)
            self.latent_lrate = latent_lrate
        else:
            self.latent_optimizer = None
            self.latent_lrate = None

        self.siren_optimizer = optim.Adam(self.opt_siren_parameters, lr=siren_lrate)
        self.siren_lrate = siren_lrate
        self.loss_fn = nn.MSELoss(reduction='none')  # only used for 'step_naive' method

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
                                       total_epochs=num_epochs, last_decay=1e-3)
            self.siren_scheduler = optim.lr_scheduler.LambdaLR(self.siren_optimizer, lr_lambda=decay_func_siren)
            if self.has_latent:
                decay_func_latent = partial(linear_decay, initial_lr=latent_lrate, final_lr=final_latent_lrate,
                                            total_epochs=num_epochs, last_decay=1e-2)
                self.latent_scheduler = optim.lr_scheduler.LambdaLR(self.latent_optimizer, lr_lambda=decay_func_latent)
        elif scheduler_type == 'none':
            pass
        else:
            raise ValueError(f"Scheduler type of {scheduler_type} is not recognized")

    def _get_mods(self) -> Tuple[Union[None, Tensor], ...]:
        # create mods (simply tuple of Nones if not enabled)
        if self.has_latent:
            latent_input = self.latent
            mods = self.modulator(latent_input)
        else:
            mods = tuple([None] * self.hidden_layers)
        return mods

    def forward(self, x: Tensor) -> Tensor:
        """
        Simple forward pass of the network
        :param x: (batches, 3)
        :return:
        """

        # get mods
        mods = self._get_mods()

        hidden_layers = tuple([l for (k, l) in self.model.items() if k.split('_')[-1] == 'SineLayer'])
        last_layer = self.model['LastLayer']
        for l, mod in zip(hidden_layers, mods):
            # pass through sine layer
            x = l(x)

            # apply mod if feature is enabled
            if mod is not None:
                x *= mod.unsqueeze(0)  # singleton allows mod to be broadcast to all batches

        # apply output layer
        x = last_layer(x)

        return x

    def forward_with_coords(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """

        Before the forward pass, clone the input and enable its gradient. Returning this cloned input allows the
        output of the network to be differentiated w.r.t. the input.

        :param x: (batches, input_dim)
        :return:
        """
        x = x.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        output = self.forward(x)

        return output, x

    def step_naive(self, x: Tensor, y: Tensor, weights: Tensor) -> float:
        """

        Step method for fitting sdf output to a target label via the MSE or BinaryCrossEntropy loss depending
        on whether the user is fitting a strong sdf or occupancy based network.
        Weights are accepted to help with class imbalance for occupancy based networks, otherwise weights
        should be a tensor of all 1's of the appropriate size.

        :param x:       (Batch, input size)
        :param y:       (Batch, output size)
        :param weights: (Batch, input size), weights to apply to input samples to correct class imbalance
        :return: loss
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

        Step method for fitting a weak sdf using the Eikonal constraints. The loss function here is described
        by the Siren paper which ensures that:

        1) The normal is constrained to have norm 1 everywhere
        2) Points on the surface have an SDF of 0
        3) Points on the surface should have their normals align with the target normals from the dataset
        4) Points off the surface should have an sdf output with large magnitude

        :param x:               (Batch, input size) -- batches of coordinates
        :param y:               (Batch, output size) -- the target normals; only meaningful for points on the surface
        :param on_surface_mask  (batch, output size) -- T/F mask where T -> batch sample is on the surface, F -> else
        :return: loss
        """
        # penalization multipliers recommended by SIREN paper
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3

        # function to calculate gradients of y w.r.t. x
        def _gradient(x: Tensor, y: Tensor, grad_outputs=None):
            if grad_outputs is None:
                grad_outputs = torch.ones_like(y)
            grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
            return grad

        # loss function for encouraging off-surface samples to have larger magnitude
        # reference siren for explicit formula and details
        def _psi(x: Tensor, a: float = 100):
            exp_in = -a * x.abs()
            exp_out = torch.exp(exp_in).squeeze(1)
            return exp_out

        # don't need singleton dimension in the mask, only need batch indices
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
        eik_loss = c1 * (1 - torch.linalg.vector_norm(grad_output, dim=1)).abs().unsqueeze(1)

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
        off_surface_loss[off_surface_mask, :] = c2 * _psi(off_surface_output).unsqueeze(1)

        # return loss description for more details
        loss_desc = [to_numpy(eik_loss.clone()), to_numpy(on_surface_loss.clone()), to_numpy(off_surface_loss.clone())]
        loss_desc = [n.mean().item() for n in loss_desc]

        # add together for a total loss
        loss = eik_loss + on_surface_loss + off_surface_loss
        loss = loss.mean()

        # perform backward gradient calculations
        loss.backward()

        # perform gradient clipping
        # typically recommended for stable training
        if self.clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.opt_siren_parameters, max_norm=self.clip_gradient_norm)
            if self.has_latent:
                torch.nn.utils.clip_grad_norm_(self.opt_latent_parameters, max_norm=self.clip_gradient_norm)

        # optimize all parameters
        self.siren_optimizer.step()
        if self.has_latent:
            self.latent_optimizer.step()

        return loss.item(), loss_desc

    def scheduler_step(self) -> Tuple[float, Optional[float]]:
        """
        Steps the siren and latent schedulers if they are being used.
        In addition, returns the siren learning rate (should always exist) and latent learning rate (optionally exists)
        before the scheduling step.
        :return:
        """

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