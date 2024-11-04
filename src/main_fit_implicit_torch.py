import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from tqdm import tqdm
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import os

# imports specific to sdf
import igl, geometry

print(plt.style.available)
plt.rcParams['text.usetex'] = True
plt.style.use("seaborn-white")

set_t = {
    'dtype': torch.float32,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

to_numpy = lambda x : x.detach().cpu().numpy()  # converts tensors to numpy arrays

class FitSurfaceModel(nn.Module):
    def __init__(self, lrate: float, loss_fn, fit_mode: str, activation:str='relu', n_layers:int=8, layer_width:int=32):
        """

        :param lrate:       Learning rate
        :param loss_fn:     Loss function use to udpated optimizable parameters
        :param fit_mode:    If the neural network should be fit for occupancy or sdf
        :param activation:  Activation function to use at nonlinear layers
        :param n_layers:    Number of layers to use in the neural network
        :param layer_width: Number of neurons per hidden layer
        """
        super(FitSurfaceModel, self).__init__()

        # parse the specified activation
        activation_lower = activation.lower()
        if activation_lower == 'relu':
            activation_fn = nn.ReLU()
        elif activation_lower == 'elu':
            activation_fn = nn.ELU()
        elif activation_lower == 'cos':
            activation_fn = torch.cos
        else:
            raise ValueError("Activation not recognized. If you wish to use a new activation function, "
                             "feel free to add it to the list in the constructor.")

        # create the network based on the specifications
        layers = [nn.Linear(3, layer_width), activation_fn]
        for i in range(n_layers - 2):
            layers.extend([nn.Linear(layer_width, layer_width), activation_fn])
        if fit_mode == 'occupancy':
            # binary classification, should be a probability in range (0, 1)
            layers.extend([nn.Linear(layer_width, 1), nn.Sigmoid()])
        elif fit_mode == 'sdf':
            # regression, last layer is fine being linear
            layers.append(nn.Linear(layer_width, 1))
        else:
            raise ValueError("fit_mode must be either 'occupancy' or 'sdf'")
        self.model = nn.Sequential(*layers)

        # set optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)

        # set loss function
        self.loss_fn = loss_fn

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model
        :param x:   (Batch, input size)
        :return:    (Batch, output size)
        """
        return self.model(x)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Returns the loss of a single forward pass
        :param x:   (Batch, input size)
        :param y:   (Batch, output size)
        :return:    loss
        """
        # zero the gradients
        self.optimizer.zero_grad()

        # pass the batch through the model
        y_hat = self.forward(x)

        # compute the loss
        loss = self.loss_fn(y_hat, y)

        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item()

class SampleDataset(Dataset):
    def __init__(
            self,
            mesh_input_file: str,
            fit_mode: str,
            n_samples: int,
            sample_weight_beta: float,
            sample_ambient_range: float,
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
            print("Collecting geometry samples")
        samp, samp_SDF = geometry.sample_mesh_importance(V, F, n_samples, beta=sample_weight_beta, ambient_range=sample_ambient_range)

        if verbose:
            print(f"Formatting labels")
        if fit_mode == 'occupancy':
            samp_target = (samp_SDF > 0) * 1.0
            n_pos = np.sum(samp_target > 0)
            n_neg = samp_target.shape[0] - n_pos
            w_pos = n_neg / (n_pos + n_neg)
            w_neg = n_pos / (n_pos + n_neg)
            samp_weight = np.where(samp_target > 0, w_pos, w_neg)
        elif fit_mode in ['sdf', 'tanh']:
            samp_target = samp_SDF
            samp_weight = np.ones_like(samp_target)
        else:
            raise ValueError(f"Fit mode {fit_mode} not recognized. Please select from ['occupancy', 'sdf', 'tanh'].")
        # in the original implicit file, the samp_weight array is not used at all so disregard it here as well

        # save inputs and labels
        if verbose:
            print(f"Saving samples and labels to the dataset")
        self.x = torch.from_numpy(samp)
        self.y = torch.from_numpy(samp_target).reshape(n_samples, 1)

    def __len__(self) -> int:
            return len(self.x)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]

def batch_count_correct(NetObject: FitSurfaceModel, batch_x: Tensor, batch_y: Tensor, fit_mode: str) -> Tensor:
    """
    For some batch of inputs and labels, return the number of predictions whose sign is correct.
    :param NetObject:   Neural network object to evaluate
    :param batch_x:     Batch of inputs
    :param batch_y:     Batch of labels
    :return:            Number of predictions whose sign is correct
    """
    prediction = NetObject.forward(batch_x)
    if fit_mode == 'occupancy':
        is_correct_sign = torch.sign(prediction) == torch.sign(batch_y - 0.5)
    elif fit_mode == 'sdf':
        is_correct_sign = torch.sign(prediction) == torch.sign(batch_y)
    else:
        raise ValueError("fit_mode must be either 'occupancy' or 'sdf'")

    current_count = is_correct_sign.to(dtype=int).sum()
    return current_count


def fit_model(
        NetObject: FitSurfaceModel,
        train_loader: DataLoader,
        fit_mode: str,
        epochs: int
) -> Tuple[list[float], list[int], list[float], FitSurfaceModel]:
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
            batch_x, batch_y = batch
            batch_x = batch_x.to(**set_t)
            batch_y = batch_y.to(**set_t)
            n_total += len(batch_x)
            curr_epoch_loss = NetObject.step(batch_x, batch_y)
            epoch_loss += curr_epoch_loss
            losses.append(curr_epoch_loss)
            with torch.no_grad():
                correct_count = batch_count_correct(NetObject, batch_x, batch_y, fit_mode).item()
                n_correct += correct_count
                correct_counts.append(correct_count)

        frac_correct= n_correct / n_total
        correct_fracs.append(frac_correct)
        epoch_loss /= len(train_loader)
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix(
            {
                'Epoch Loss': epoch_loss,
                'Correct Sign': f'{100*frac_correct:.2f}%'
            }
        )

    # return losses and trained network
    return losses, correct_counts, correct_fracs, NetObject

def main(
        input_file: Optional[str] = None,
        output_file: Optional[str] = None
):
    assert (input_file is None and output_file is None) or (
                isinstance(input_file, str) and isinstance(output_file, str))

    print(f"Torch Settings: {set_t}")

    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)

    # network
    parser.add_argument("--activation", type=str, default='elu')
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--layer_width", type=int, default=32)
    parser.add_argument("--positional_encoding", action='store_true')
    parser.add_argument("--positional_count", type=int, default=10)
    parser.add_argument("--positional_pow_start", type=int, default=-3)

    # loss / data
    parser.add_argument("--fit_mode", type=str, default='sdf')
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1000000)
    parser.add_argument("--sample_ambient_range", type=float, default=1.25)
    parser.add_argument("--sample_weight_beta", type=float, default=20.)

    # training
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr_decay_every", type=int, default=99999)
    parser.add_argument("--lr_decay_frac", type=float, default=.5)

    # jax options
    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    print(f"Program Configuration: {args}")

    input_file = args.input_file if input_file is None else input_file
    output_file = args.output_file if output_file is None else output_file

    # validate some inputs
    if args.activation not in ['relu', 'elu', 'cos']:
        raise ValueError("unrecognized activation")
    if args.fit_mode not in ['occupancy', 'sdf']:
        raise ValueError("unrecognized activation")
    if not output_file.endswith('.npz'):
        raise ValueError("output file should end with .npz")

    # initialize the network
    if args.fit_mode == 'occupancy':
        loss_fn = nn.BCELoss()
    elif args.fit_mode == 'sdf':
        loss_fn = nn.L1Loss()
    else:
        raise ValueError("unrecognized fit_mode")
    model_params = {
        'lrate': args.lr,
        'loss_fn': loss_fn,
        'fit_mode': args.fit_mode,
        'activation': args.activation,
        'n_layers': args.n_layers,
        'layer_width': args.layer_width,
    }
    NetObject = FitSurfaceModel(**model_params)

    # initialize the dataset
    dataset_pararms = {
        'mesh_input_file': input_file,
        'fit_mode': args.fit_mode,
        'n_samples': args.n_samples,
        'sample_weight_beta': args.sample_weight_beta,
        'sample_ambient_range': args.sample_ambient_range,
        'verbose': True
    }
    train_dataset = SampleDataset(**dataset_pararms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # train the neural network
    losses, correct_counts, correct_fracs, NetObject = fit_model(NetObject, train_loader, args.fit_mode, args.n_epochs)

    # save the neural network in Torch format

    # display results
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
    plt.show()

    # save neural network to the specified path
    # Iterate over model parameters
    for name, param in NetObject.named_parameters():
        print(f"Parameter Name: {name}")
        print(f"Parameter Value: {param}")
        print("-" * 30)

    # save the neural network in .npz format


if __name__ == '__main__':
    main()
    # use_predefined_files = True
    # input_directory = "/home/jorgejc2/Documents/Research/ray-casting/Thingi10K/raw_meshes/"
    # output_directory = "/home/jorgejc2/Documents/Research/ray-casting/sample_inputs/Thingi10K_inputs/"
    # os.makedirs(output_directory, exist_ok=True)
    # file_names = [f for f in os.listdir(input_directory) if f.endswith('.obj')]
    # input_files = [input_directory + f for f in file_names]
    # output_files = [output_directory + f.replace(".obj", ".npz") for f in file_names]
    # main_args = {
    #     "input_files": input_files,
    #     "output_files": output_files
    # }
    # if use_predefined_files:
    #     for in_file, out_file in zip(input_files, output_files):
    #         main(in_file, out_file)
    # else:
    #     main()