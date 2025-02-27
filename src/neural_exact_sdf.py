"""
Describes exact SDFs to use that can be defined algebraically.
This script also contains a main method demonstrating how the exact sdf visually looks
and can be used to help debug that your sdf is working as intended.
"""
import argparse
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
from warnings import warn

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

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

class BaseExactSDF(ABC):
    """
    Abstract Base Class with bare minimum methods that an exact SDF object should implement.
    """
    def __init__(self):
        pass

    @abstractmethod
    def generate_vertices(self, num_samples: int, *args, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Should generate a set of samples on the zero level-set
        """
        ...

    @abstractmethod
    def query_sdf(self, pts: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Should return the signed distance for a batch of vertices and potentially their normals.
        """
        ...

    def __call__(self, pts: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        A forward call on the object for a set of points.
        :param pts:         Points to query SDF with.
        :return:            The signed distance results.
        """
        return self.query_sdf(pts)

class StarExactSDF(BaseExactSDF):
    def __init__(self, R0: float = 1.0, R1: float = 0.6, k: int = 4, sharpness: int = 1):
        """
        Instantiates a Star SDF object.
        :param R0:          Radius of the base circle.
        :param R1:          Radius of each spike.
        :param k:           2*k is the number of spikes used in the star.
        :param sharpness:   Smaller values render spikier stars.
        """
        super().__init__()
        self._R0 = R0
        self._R1 = R1
        self._k = k
        self._sharpness = sharpness

    def generate_vertices(self, num_samples: int, *args, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Generates a set of random vertices on the zero level-set.
        :param num_samples:
        :return:
        """
        device = kwargs.get('device', set_t['device'])
        # Step 1: Sample random angles in [0, 2π]
        theta = torch.rand(num_samples, device=device) * 2 * torch.pi  # Uniformly distributed

        # Step 2: Compute the exact radius at each angle
        r_star = self._star_sdf_equation(theta)

        # Step 3: Convert to Cartesian coordinates
        x = r_star * torch.cos(theta)
        y = r_star * torch.sin(theta)

        return torch.stack([x, y], dim=-1), None  # Shape: (num_samples, 2)

    def query_sdf(self, pts: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """

        :param pts:
        :return:
        """
        x, y = pts[:, 0], pts[:, 1]

        # Convert to polar coordinates
        r = torch.sqrt(x ** 2 + y ** 2)  # Euclidean distance
        theta = torch.atan2(y, x)  # Angle in radians

        # Compute the star boundary in polar coordinates
        r_star = self._star_sdf_equation(theta)

        # Compute signed distance
        sdf = r / r_star - 1  # Normalize distance to shape

        return sdf, None

    def _star_sdf_equation(self, theta: Tensor) -> Tensor:
        """
        The exact equation used for the sdf calculation for a star.
        :param theta:
        :return:
        """
        return self._R0 + self._R1 * (1 - torch.abs(torch.cos(self._k * theta)) ** self._sharpness)


def render_exact_sdf(figure, ax, sdf_obj: BaseExactSDF, num_levelset_samples: int, num_off_samples: int,
                     display_normals: bool = False):
    """
    Given an exact sdf object, renders an image.
    :param figure:                  Matplotlib figure object to display the results to.
    :param ax:                      Matplotlib axes object to display the results to.
    :param sdf_obj:                 ExactSDFObject to aid in generating level-set and off level samples.
    :param num_levelset_samples:    The number of samples to draw from the zero level-set.
    :param num_off_samples:         The number of random points to draw to show their signed distance.
    :param display_normals:         If True and supported, the normals will also be displayed on the zero level-set.
    :return:
    """

    levelset_samples, levelset_normals = sdf_obj.generate_vertices(num_levelset_samples)
    levelset_samples_np = to_numpy(levelset_samples)
    levelset_normals_np = to_numpy(levelset_normals)

    ax.scatter(levelset_samples_np[:, 0], levelset_samples_np[:, 1])
    if display_normals:
        if levelset_normals is None:
            warn("'display_normals' was set but this SDF object does not return normals.")
        else:
            ax.quiver(levelset_samples_np[:, 0], levelset_samples_np[:, 1], levelset_normals_np[:, 0], levelset_normals_np[:, 1], scale=1)

    R_max = 1.6  # Slightly larger than star boundary
    query_pts = (torch.rand(num_off_samples, 2) * 2 - 1) * R_max  # Uniformly sample in [-R_max, R_max]^2
    query_pts = query_pts.to(**set_t)
    offlevel_sdf, _ = sdf_obj(query_pts)
    offlevel_sdf_np = to_numpy(offlevel_sdf)
    off_x = to_numpy(query_pts[:, 0])
    off_y = to_numpy(query_pts[:, 1])
    scat_obj = ax.scatter(off_x, off_y, c=offlevel_sdf_np, cmap="coolwarm", edgecolors="k", s=30, label="Query Points",
                          vmin=-1, vmax=1)
    figure.colorbar(scat_obj, ax=ax, label="Signed Distance")

def get_sdf_choice(option: str) -> BaseExactSDF:
    if option == 'star':
        return StarExactSDF()
    else:
        raise ValueError(f"Unknown option: {option}")

def main(args: dict):
    # extract parsed arguments
    exact_sdf_method = args['exact_sdf_method']
    output_file = args['output_file']
    num_levelset_samples = args['num_levelset_samples']
    num_random_pts = args['num_random_pts']
    display_normals = args['display_normals']

    sdf_obj = get_sdf_choice(exact_sdf_method)

    # create plot handler and display to it
    figure, ax = plt.subplots(1, 1)
    render_exact_sdf(figure, ax, sdf_obj, num_levelset_samples, num_random_pts, display_normals)

    # additional formatting
    ax.set_title(f"Rendering {exact_sdf_method} Object")
    ax.set_aspect('equal')
    ax.grid(True)

    # save and show the plot
    if output_file is not None:
        print(f"Saving results to {output_file}...")
        plt.savefig(output_file)
    plt.show()


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("exact_sdf_method", type=str,
                        choices=['star'],
                        help="The path to the .pth model from the root directory.")
    parser.add_argument("--output_file", type=str,
                        help="The path to save the result rendered image of the exact sdf.")
    parser.add_argument("--num_levelset_samples", type=int, default=1000,
                        help="The number of samples to draw from the zero level-set.")
    parser.add_argument("--num_random_pts", type=int, default=1000,
                        help="The number of random points to draw to show their signed distance.")
    parser.add_argument("--display_normals", action='store_true',
                        help="If true, the normals will also be displayed on the zero level-set.")

    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict

if __name__ == "__main__":
    print(f"set_t: {set_t}")
    parsed_args = parse_args()
    main(parsed_args)