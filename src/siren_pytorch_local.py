import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out

# siren network

class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0 = 1.,
        w0_initial = 30.,
        use_bias = True,
        final_activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            )

            self.layers.append(layer)

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)

# modulatory feed forward

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

# wrapper

class SirenWrapper(nn.Module):
    def __init__(self, net, lrate, latent_dim = None,
                 step_size = None, gamma = None):
        super().__init__()
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        self.net = net
        parameters = [*self.net.parameters()]

        self.modulator = None
        self.latent = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )
            self.latent = nn.Parameter(torch.zeros(latent_dim).normal_(0, 1e-2))
            parameters.extend([
                self.latent,
                *self.modulator.parameters(),
            ])

        # print(f"SirenWrapper has parameters: \n{parameters}")
        self.optimizer = optim.Adam(parameters, lr=lrate)
        self.loss_fn = nn.MSELoss(reduction='none')

        # set LR scheduler
        self.scheduler = None
        if step_size is not None and gamma is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size,
                                                       gamma=gamma)

    def forward(self, x):
        modulate = exists(self.modulator)
        latent = exists(self.latent)
        assert not (modulate ^ latent), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(self.latent) if modulate else None

        out = self.net(x, mods)

        return out

    def step(self, x, y, weights):
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