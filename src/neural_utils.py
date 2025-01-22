"""
Addtional utility methods for the SDF/occupancy based neural network architectures.
"""
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple

from neural_sdf import MLP, Siren

available_activations = [nn.ReLU, nn.ELU, nn.GELU, nn.Sigmoid]  # list of currently supported activation functions

to_numpy = lambda x : x.detach().cpu().numpy()

def load_net_object(pth_file: str, model_type: str) -> Union[MLP, Siren]:
    """
    A helper function that retrieves a torch network from a .pth file. As a note, it is wise to save the model type
    in the name of the .pth file since it is a required parameter to load the neural network. This is because the
    MLP, Siren, and nglod architectures are very different that it is easier to define them in separate classes rather
    than try to consolidate them into a single class.
    :param pth_file:    .pth file to load network parameters and weights from.
    :param model_type:  The type of neural network architecture the .pth file is associated with.
    :return:    Network object
    """
    model_type = model_type.lower()
    pth_dict = torch.load(pth_file, weights_only=True)
    state_dict = pth_dict["state_dict"]  # weights and biases
    model_params = pth_dict["model_params"]  # rest of the parameters
    # initialize the NN model based on its type of architecture
    if model_type == 'mlp':
        net_object = MLP(**model_params)
    elif model_type == 'siren':
        net_object = Siren(**model_params)
    elif model_type == 'nglod':
        raise NotImplementedError('nglod models not yet supported.')
    else:
        raise ValueError('Invalid model type')


    net_object.load_state_dict(state_dict)  # load in weights and biases
    net_object.eval()  # set to evaluation mode

    return net_object

def save_net_object(net_object: Union[MLP, Siren], losses: list[float], model_params: dict, output_file: str, 
                    verbose: bool = False):
    """
    Saves the model in a .pth file, generates a .png plot of the losses, and saves the model weights and layers
    to an .npz file that is compatible with the Jax ray tracing scripts.
    :param net_object: 
    :param losses: 
    :param model_params: 
    :param output_file:     Path to save the model. The file extension is expected to be .npz.
    :param verbose: 
    :return: 
    """
    net_object.eval()  # set to evaluation mode

    # save the neural network in Torch format
    # TODO: Probably cleaner to not depend on the output_file to end with .npz and could leave more general
    pth_file = output_file.replace('.npz', '.pth')
    print(f"Saving model to {pth_file}...")
    pth_dict = {
        "state_dict": net_object.state_dict(),
        "model_params": model_params,
    }
    torch.save(pth_dict, pth_file)

    # display results
    plt_file = output_file.replace('.npz', '.png')
    plot_training_metrics(losses, None, plt_file, False)

    # save the neural network in .npz format
    save_to_npz(net_object, output_file, verbose)

def plot_training_metrics(losses: list[float], correct_fracs: Optional[list[float]] = None, save_path: Optional[str] = None, display: bool = False):
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

    num_subplots = 1 if correct_fracs is None else 2
    fig, axes = plt.subplots(1, num_subplots, figsize=(10, 5))
    axes = np.ravel(axes)

    ax1 = axes[0]
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title("Training Loss")
    ax1.grid()

    if correct_fracs is not None:
        ax2 = axes[1]
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

def save_to_npz(net_object, npz_path: str, verbose: bool = False):
    """
    Saves the Torch model as a .npz file that can be loaded in by the other ray tracing scripts. Runs in 3 stages:

    1) Get all the optimizable layers which are simply the linear layers and format them
    2) Get all the activation functions and add them to the npz dictionary as well
    3) Finally, add a 'squeeze_last' parameter as the ray-tracing scripts rely on this parameter.

    :param net_object:   Neural network object to save
    :param npz_path:    .npz file path to save the network to
    :param verbose:     If true, prints additional logging information
    :return:
    """
    npz_dict = {}  # holds network architecture
    if verbose:
        print("Adding optimizable parameters to the npz dictionary")
    for name, param in net_object.named_parameters():
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
    for name, layer in net_object.model._modules.items():
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

    squeeze_last_idx = len(net_object.model._modules.keys())
    if verbose:
        print(f"Adding squeeze_last at layer index {squeeze_last_idx}")
    squeeze_last_idx_formatted = f"{squeeze_last_idx:04d}.squeeze_last._"
    npz_dict[squeeze_last_idx_formatted] = np.empty(0)

    if verbose:
        print(f"Saving network in .npz format with path {npz_path} \nand dictionary with keys \n{npz_dict.keys()}")
    np.savez(npz_path, **npz_dict)

def batch_count_correct(net_object, batch_x: Tensor, batch_y: Tensor,
                        fit_mode: str) -> Tensor:
    """
    For some batch of inputs and labels, return the number of predictions whose sign is correct.
    :param net_object:   Neural network object to evaluate
    :param batch_x:     Batch of inputs
    :param batch_y:     Batch of labels
    :return:            Number of predictions whose sign is correct
    """
    prediction = net_object.forward(batch_x)
    if fit_mode in 'occupancy':
        # labels are probabilities, they must be corrected
        is_correct_sign = torch.sign(prediction) == torch.sign(batch_y - 0.5)
    elif fit_mode == 'sdf':
        is_correct_sign = torch.sign(prediction) == torch.sign(batch_y)
    else:
        raise ValueError("fit_mode must be either 'occupancy' or 'sdf'")

    current_count = is_correct_sign.to(dtype=int).sum()
    return current_count