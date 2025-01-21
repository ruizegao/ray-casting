"""
Additional building blocks that are unique in the SDF/occupancy neural network setting. It is more convenient to define
these building blocks as separate Torch modules that can be imported and used in neural_sdf.py.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Union
import numpy as np

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

    def __init__(self, input_dim: int, L: int, start_pow: int = 0, with_shift: bool = True, prepend: bool = False):
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
        # will unflatten input from (batches, input_dim) to (batches, input_dim, 1)
        self.unflatten = nn.Unflatten(1, (input_dim, 1))
        # will flatten input from (batches, input_dim, L(*2)) to (batches, input_dim*L(*2))
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
    def compute_output_dim(input_dim: int, positional_count: int, with_shift: bool, prepend: bool):
        """
        Returns what the output dimension of this Module would be given these parameters
        """
        output_dim = input_dim * positional_count
        if with_shift:
            output_dim *= 2

        if prepend:
            output_dim += 3

        return output_dim

class SineLayer(nn.Module):
    """
    The SineLayer implementation given by the Siren paper.

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
        :param in_features:     Input dimension of layer
        :param out_features:    Output dimension of layer
        :param bias:            If true, adds a bias vector
        :param is_first:        If true, this is the first layer of the network
        :param omega_0:         Angular frequency multiplied to the input before applying sine activation
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights such that the input is distributed w.r.t. the uniform distribution.
        This special initialization was recommended by the Siren paper for better training stabilization.
        :return:
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))

class Modulator(nn.Module):
    """
    The modulator network implementation.

    See the paper by Adobe which combines modulation with Siren. Modulation's primary purpose in the paper is to allow
    the model to be generalizable to different images, but can be used to reduce noise in SDF (if carefully tuned).

    """
    def __init__(self, dim_in: int, dim_hidden: int, num_layers: int):
        """
        :param dim_in:      Input dimension of the network
        :param dim_hidden:  Output dimension of the network
        :param num_layers:  Numer of layers in the network
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        # Creates simple ReLU network with skip connections.
        # Skip connections brings the latent input tensor to all layers of the network.
        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z: Tensor) -> Tuple[Tensor, ...]:
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)