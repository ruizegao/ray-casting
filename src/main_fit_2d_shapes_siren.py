import torch
import argparse

from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from tqdm import tqdm
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

from main_fit_point_cloud import Siren, fit_model
from src.neural_rendering import init_circle_sdf_torch

input_shapes = ['circle']

# print(plt.style.available)  # uncomment to view the available plot styles
plt.rcParams['text.usetex'] = False  # tex not necessary here and may cause error if not installed

# Set plot style to seaborn white. If these options do not work, don't set the plot style or select from other
# available plot styles.
try:
    plt.style.use("seaborn-white")
except OSError as e:
    plt.style.use("seaborn-v0_8-white")

set_t = {
    'dtype': torch.float32,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

class BasicShapeDataset(Dataset):
    def __init__(self, shape_dict: dict, on_surface_points: int, keep_aspect_ratio: bool = True,
                 model: str = 'siren', fit_mode: str = 'sdf'):
        super().__init__()
        shape = shape_dict.get('shape', None)
        print(f"Creating dataset for simple sdf shape {shape}")
        if shape == 'circle':
            # extract parameters
            center = shape_dict.get('center', (0.,0.))
            radius = shape_dict.get('radius', 0.1)
            print(f"Creating dataset for circle sdf with center {center}, radius {radius}")
            # generate coordinates that lie on the surface of the circle
            theta_samples = np.random.uniform(low=0., high=2 * np.pi, size=on_surface_points)
            xv = (radius * np.cos(theta_samples)) + center[0]
            yv = (radius * np.sin(theta_samples)) + center[1]
            xy_samples = np.stack([xv, yv], axis=1)
            coords = torch.from_numpy(xy_samples).to(**set_t)
            # initialize the exact sdf for a parameterized circle and get the distances and normals
            circle_sdf = init_circle_sdf_torch(center, radius)
            self._exact_sdf = circle_sdf
            # no need to save the distances since they will be 0 for points on the surface
            _, normals = circle_sdf(coords)
            # send normals to cpu and save; coords will be further manipulated before being saved
            self._normals = normals.cpu()
        else:
            raise NotImplementedError(f"Shape {shape} not implemented")

        # normalize the coordinates and send them to the CPU
        # self._coords = self._coords_normalization(coords, keep_aspect_ratio).to(device='cpu')
        coords = coords.cpu()  # shape (on_surface_samples, 2)
        self._coords = coords

        self._on_surface_points = on_surface_points
        self._model = model

        ### for mlp
        off_surface_points = on_surface_points
        n_samples = on_surface_points + off_surface_points
        off_surface_coords = torch.from_numpy(np.random.uniform(-1, 1, size=(off_surface_points, 2))).to(
            device=set_t['device']
        )
        samp_SDF, _ = self._exact_sdf(off_surface_coords, device=set_t['device'])
        samp_SDF = samp_SDF.cpu()

        if fit_mode == 'occupancy':
            # apply label and calculate sample weight to correct class imbalance
            samp_target = (samp_SDF > 0) * 1.0
            n_pos = torch.sum(samp_target > 0)
            n_neg = samp_target.shape[0] - n_pos
            w_pos = n_neg / (n_pos + n_neg)
            w_neg = n_pos / (n_pos + n_neg)
            samp_weight = torch.where(samp_target > 0, w_pos, w_neg)
        elif fit_mode == 'sdf':
            # apply label and give all weights equal importance
            # since this is regression not classification based
            samp_target = samp_SDF
            samp_weight = np.ones_like(samp_target)
        else:
            raise ValueError(f"Fit mode {fit_mode} not recognized. Please select from ['occupancy', 'sdf'].")
        # in the original implicit file, the samp_weight array is not used at all so disregard it here as well

        # save inputs and labels
        coords = torch.concatenate((coords, off_surface_coords.cpu()), dim=0)
        samp_target = samp_target.repeat(2, 1)
        samp_target[:on_surface_points, :] = 0.
        self.x = coords  # shape (n_samples, 2)
        self.y = samp_target.cpu().reshape(n_samples, 1)
        self.weights = samp_weight.cpu().reshape(n_samples, 1)

    def _coords_normalization(self, coords: Tensor, keep_aspect_ratio: bool):
        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= torch.mean(coords, dim=0, keepdim=True)
        if keep_aspect_ratio:
            coord_max = torch.amax(coords)
            coord_min = torch.amin(coords)
        else:
            coord_max = torch.amax(coords, dim=0, keepdim=True)
            coord_min = torch.amin(coords, dim=0, keepdim=True)
        new_coords = (coords - coord_min) / (coord_max - coord_min)
        new_coords -= 0.5
        new_coords *= 2.
        return new_coords

    def __len__(self):
        return self._coords.shape[0] // self._on_surface_points

    def __getitem__(self, idx):
        if self._model == 'siren':
            return self._siren_getitem(idx)
        elif self._model == 'mlp':
            return self._mlp_get_item(idx)
        else:
            raise NotImplementedError(f"Model {self._model} not implemented")

    def _mlp_get_item(self, idx):
        return self.x[idx], self.y[idx], self.weights[idx]

    def _siren_getitem(self, idx):
        point_cloud_size = self._coords.shape[0]

        off_surface_samples = self._on_surface_points  # **2
        total_samples = self._on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = torch.from_numpy(np.random.choice(point_cloud_size, size=self._on_surface_points))

        on_surface_coords = self._coords[rand_idcs, :]
        on_surface_normals = self._normals[rand_idcs, :]

        off_surface_coords = torch.from_numpy(np.random.uniform(-1, 1, size=(off_surface_samples, 2))).to(
            dtype=set_t['dtype'])
        off_surface_normals = torch.from_numpy(np.ones((off_surface_samples, 2)) * -1).to(dtype=set_t['dtype'])

        # Set sdf to 0 for on surface points and calculate sdf for off-surface points; Exact sdf calculation for
        # off-surface points is perhaps not necessary but is possible for primitive shapes.
        # FIXME: Since coordinates are normalized, the exact sdf calculation could be incorrect
        sdf = torch.zeros((total_samples, 1), dtype=set_t['dtype'])  # on-surface = 0
        off_dist, _ = self._exact_sdf(off_surface_coords, device=set_t['device'])
        sdf[self._on_surface_points:, :] = off_dist.reshape(off_surface_samples, 1).cpu()
        # sdf[self._on_surface_points:, :] = -1  # off-surface = -1

        coords = torch.concatenate((on_surface_coords, off_surface_coords), dim=0)
        normals = torch.concatenate((on_surface_normals, off_surface_normals), dim=0)

        surface_mask = torch.zeros((total_samples, 1), dtype=torch.bool)
        surface_mask[0:self._on_surface_points, :] = True

        return {'coords': coords}, {'sdf': sdf, 'normals': normals, 'surface_mask': surface_mask}

def plot_training_metrics(losses: list[float], save_path: Optional[str] = None, display: bool = False):
    """
    Displays and/or saves the metrics recorded during the training of the implicit surface.
    :param losses:          List of losses over epochs
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

def main(args: dict):
    print(f"Torch Settings: {set_t}")

    ##  unpack arguments

    # Build arguments
    input_shape = args["input_shape"]
    output_file = args["output_file"]
    sdf_png_path = args["sdf_png_path"]
    assert input_shape in input_shapes, f"input_shape '{input_shapes}' not supported"
    if output_file is None:
        raise ValueError("output_file is None")
    # network
    n_layers = args["n_layers"]
    layer_width = args["layer_width"]
    clip_gradient_norm = args["clip_gradient_norm"]
    # siren params
    siren_latent_dim = args["siren_latent_dim"]
    siren_first_omega_0 = args["siren_first_omega_0"]
    siren_hidden_omega_0 = args["siren_hidden_omega_0"]
    siren_c1 = args["siren_c1"]
    siren_c2 = args["siren_c2"]
    siren_c3 = args["siren_c3"]

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

    # build the neural network with the specified configuration
    model_params = {
        'in_features': 2,
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
        'c1': siren_c1,
        'c2': siren_c2,
        'c3': siren_c3,
        'clip_gradient_norm': clip_gradient_norm
    }
    net_object = Siren(**model_params)

    # load the dataset
    sdf_dataset = BasicShapeDataset({'shape': input_shape, 'sdf_png_path': sdf_png_path},
                                    on_surface_points=batch_size)
    dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1)

    # train the neural network
    losses, net_object = fit_model(net_object, dataloader, n_epochs)

    net_object.eval()  # set to evaluation mode

    # save the neural network in Torch format
    pth_file = output_file.replace('.xyz', '.pth')
    print(f"Saving model to {pth_file}...")
    pth_dict = {
        "state_dict": net_object.state_dict(),
        "model_params": model_params
    }
    torch.save(pth_dict, pth_file)

    # display results
    plt_file = output_file.replace('.xyz', '.png')
    plot_training_metrics(losses, plt_file, display_plots)

def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("--input_shape", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--sdf_png_path", type=str, default=None)

    # network
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--layer_width", type=int, default=32)
    parser.add_argument("--clip_gradient_norm", type=float, default=1.0)
    # positional arguments
    parser.add_argument("--positional_encoding", action='store_true')
    parser.add_argument("--positional_count", type=int, default=10)
    parser.add_argument("--positional_pow_start", type=int, default=-3)
    parser.add_argument("--positional_prepend", action='store_true')
    # siren arguments
    parser.add_argument("--siren_latent_dim", type=int, default=0)
    parser.add_argument("--siren_first_omega_0", type=int, default=30)
    parser.add_argument("--siren_hidden_omega_0", type=int, default=30)
    parser.add_argument("--siren_c1", type=float, default=5e1)
    parser.add_argument("--siren_c2", type=float, default=3e3)
    parser.add_argument("--siren_c3", type=float, default=1e2)

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

if __name__ == "__main__":
    # parse user arguments
    args_dict = parse_args()
    main(args_dict)