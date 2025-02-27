"""
Main script for taking a pretrained SDF/occupancy based neural network and visualizing their output.
"""
import argparse
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from neural_sdf import MLP, Siren
from neural_utils import load_net_object

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
gpu_id = torch.cuda.current_device()

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

def init_circle_sdf(center, r):
    def circle_sdf(pts):
        num_pts = pts.shape[0]
        pts_x, pts_y = pts[:, 0], pts[:, 1]
        [center_x, center_y] = center
        pts_xo = pts_x - center_x
        pts_yo = pts_y - center_y
        dist_x = np.power(pts_xo, 2)
        dist_y = np.power(pts_yo, 2)
        l2_norm = np.sqrt(dist_x + dist_y)
        dist = l2_norm - r
        normals = np.stack([pts_xo, pts_yo], axis=1) / l2_norm.reshape(num_pts, 1)
        return dist, normals
    return circle_sdf

def init_circle_sdf_torch(center: Tuple[float, float], r: float):
    """
    With the given circle parameters, returns an exact sdf associated with the circle.
    :param center:
    :param r:
    :return:
    """
    def circle_sdf(pts: Tensor, device: Optional[torch.device]=None) -> Tuple[Tensor, Tensor]:
        """
        Calculates the exact sdf for a batch of points and normals. Normal calculation is meaningless for points
        that do not lie on the surface.
        :param pts:     Batch of points
        :param device:  The device to perform the calculation on, i.e. pts may reside on CPU, but we can transfer them
                        to GPU solely for this exact sdf calculation
        :return:
        """
        if device is not None:
            pts = pts.to(device)
        num_pts = pts.shape[0]
        pts_x, pts_y = pts[:, 0], pts[:, 1]
        [center_x, center_y] = center
        pts_xo = pts_x - center_x
        pts_yo = pts_y - center_y
        dist_x = torch.pow(pts_xo, 2)
        dist_y = torch.pow(pts_yo, 2)
        l2_norm = torch.sqrt(dist_x + dist_y)
        dist = l2_norm - r
        normals = torch.stack([pts_xo, pts_yo], dim=1) / l2_norm.reshape(num_pts, 1)
        return dist, normals
    return circle_sdf

def parametric_curve(t):
    """
    Define the parametric curve here.
    For example, an ellipse:
        x(t) = a * cos(t)
        y(t) = b * sin(t)
    """
    a, b = 2.0, 1.0  # Ellipse parameters
    x = a * np.cos(t)
    y = b * np.sin(t)
    return np.stack([x, y], axis=1)

def render_sdf_image(width, height, curve_fn, t_samples, line_thickness=1, scale=1., rgb_mode=False):
    if rgb_mode:
        # Create a blank black RGB image
        image = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        # Create a blank black grayscale image
        image = np.zeros((height, width), dtype=np.uint8)

    distances, normals = curve_fn(t_samples)
    x_samples, y_samples = t_samples[:, 0], t_samples[:, 1]
    nx, ny = normals[:, 0], normals[:, 1]
    eps = 0.01
    dist_mask = np.isclose(np.abs(distances), 0., atol=eps)
    # print(f"Distances: ")
    # for d in distances:
    #     print(f"{d:.2f}")
    print(f"dist_mask sum: {dist_mask.sum()}")
    print(f"normals shape: {normals.shape}")

    # normalize the (x,y) coordinates
    x_min, x_max = x_samples.min(), x_samples.max()
    y_min, y_max = y_samples.min(), y_samples.max()
    x_samples_normalized = x_samples / (x_max - x_min)
    y_samples_normalized = y_samples / (y_max - y_min)

    offset_x = width // 2
    offset_y = height // 2

    curve_x = (x_samples_normalized * scale * (width - 1)).astype(int)
    curve_y = (y_samples_normalized * scale * (height - 1)).astype(int)

    if rgb_mode == False:
        white_x = curve_x[dist_mask] + offset_x
        white_y = curve_y[dist_mask] + offset_y
        image[white_y, white_x] = 255

    # also save the matplotlib to show the normals

    plt.scatter(x_samples[dist_mask], y_samples[dist_mask], color="blue", label="Points")
    # Plot the normal vectors using quiver
    plt.quiver(x_samples[dist_mask], y_samples[dist_mask], nx[dist_mask], ny[dist_mask], angles="xy", scale_units="xy", scale=10, color="red")

    # Add labels and a legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot with Normal Vectors")
    plt.axis("equal")  # Equal scaling for x and y axes
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.savefig("../parametric_renderings/sdf_plot.png")
    plt.close()

    return image

def render_parametric_curve_image(width, height, curve_fn, t_samples, line_thickness=1, scale=1., rgb_mode=False):
    """
    Renders a grayscale image displaying a parametric curve.

    Args:
        width (int): Width of the output image in pixels.
        height (int): Height of the output image in pixels.
        curve_fn (function): Function defining the parametric curve.
        t_samples (np.ndarray): Discrete parameter samples (e.g., np.linspace).
        line_thickness (int): Thickness of the curve in pixels.

    Returns:
        np.ndarray: Grayscale image array with the parametric curve.
    """
    if rgb_mode:
        # Create a blank black RGB image
        image = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        # Create a blank black grayscale image
        image = np.zeros((height, width), dtype=np.uint8)

    # Generate curve points in the parametric space
    curve_points = curve_fn(t_samples)  # Shape (N, 2)

    # Normalize curve points to fit within the image dimensions
    curve_min = curve_points.min(axis=0)
    curve_max = curve_points.max(axis=0)
    # curve_points_normalized = (curve_points - curve_min) / (curve_max - curve_min)
    curve_points_normalized = (curve_points) / (curve_max - curve_min)
    curve_points_normalized *= scale
    offset_x = width // 2
    offset_y = height // 2

    # Scale to image dimensions
    curve_pixels = (curve_points_normalized * [width - 1, height - 1]).astype(int)

    # # Draw the curve on the image
    # for px, py in curve_pixels:
    #     # Draw a point with optional thickness
    #     for dx in range(-line_thickness, line_thickness + 1):
    #         for dy in range(-line_thickness, line_thickness + 1):
    #             x, y = px + dx, py + dy
    #             x += offset_x
    #             y += offset_y
    #             if 0 <= x < width and 0 <= y < height:
    #                 image[y, x] = 255  # Set pixel to white

    # Draw the curve on the image
    for px, py in curve_pixels:
        # Draw a point with optional thickness
        for dx in range(-line_thickness, line_thickness + 1):
            for dy in range(-line_thickness, line_thickness + 1):
                x, y = px + dx, py + dy
                x += offset_x
                y += offset_y
                if 0 <= x < width and 0 <= y < height:
                    if rgb_mode:
                        image[y, x] = [0, 0, 0]  # Black border for the curve
                    else:
                        image[y, x] = 255  # Set pixel to white

    if rgb_mode:
        # Add signed distance coloring
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        x_normalized = xx / (width - 1) * (curve_max[0] - curve_min[0]) + curve_min[0]
        y_normalized = yy / (height - 1) * (curve_max[1] - curve_min[1]) + curve_min[1]

        # Compute signed distance to curve
        grid_points = np.stack([x_normalized, y_normalized], axis=-1)
        distances = np.linalg.norm(grid_points[:, :, None] - curve_points[None, None, :], axis=-1).min(axis=-1)

        # Map distances to red (negative) and blue (positive)
        max_distance = distances.max()
        distances_normalized = distances / max_distance
        image[:, :, 0] = (255 * distances_normalized).astype(np.uint8)  # Red channel
        image[:, :, 2] = (255 * (1 - distances_normalized)).astype(np.uint8)  # Blue channel


    return image

def sample_model(net: Union[MLP, Siren], save_path: str, show_normals: bool = False, normal_scale: float = 1.0,
                 l_range: float = -0.55,u_range: float = 0.55, generate_n_random_samples: Optional[int] = None,
                 dim_samples: int = 1000, normal_samples: int = 1000):
    """
    Generates a heat map plot of a neural SDF
    :param net:             SDF Net object
    :param save_path:       Path to save plot to
    :param show_normals:    If true, also displays the normals of the points on the surface
    :param normal_scale:    Scale the normals by the given value after they have been normalized
    :param dim_samples:     Number of evenly spaced samples along each dimension to form the mesh grid
    :param normal_samples:     Number of samples along the surface for plotting their normals
    :return:
    """
    # from matplotlib.patches import Circle
    x_np = np.linspace(l_range, u_range, dim_samples)  # 100 points along the x-axis
    y_np = np.linspace(l_range, u_range, dim_samples)  # 100 points along the y-axis
    X_np, Y_np = np.meshgrid(x_np, y_np)
    X_np = X_np.flatten()
    Y_np = Y_np.flatten()
    coords_np = np.stack((X_np, Y_np), axis=1)
    coords = torch.from_numpy(coords_np).to(**set_t)
    dist = net(coords)
    dist_np = to_numpy(dist)

    # Reshape distances back to 2D for plotting
    dist_2d = dist_np.reshape((dim_samples,)*2)

    # Sample directly on the surface if we also want to display the normals of this SDF
    if show_normals:
        # function to calculate gradients of y w.r.t. x
        def _gradient(x: Tensor, y: Tensor, grad_outputs=None):
            if grad_outputs is None:
                grad_outputs = torch.ones_like(y)
            grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
            return grad

        # FIXME: This logic is not correct, but it shows that we could handle inputs in an arbitrary range
        # if isinstance(net, MLP) and not net.truncate_output:
        #     xl, xu = dist_2d[:, 0].min(), dist_2d[:, 0].max()
        #     yl, yu = dist_2d[:, 1].min(), dist_2d[:, 1].max()
        #     x_range = xu - xl
        #     y_range = yu - yu
        #     scale = torch.tensor([x_range, y_range], **set_t).reshape(1, 2)
        #     offset = torch.tensor([xl, yu], **set_t).reshape(1, 2)
        #     _generate_samples = lambda : torch.rand((dim_samples * 100, 2), **set_t) * scale + offset
        # else:
        # generate square number of samples to speed up the process of finding samples on the surface level-set
        gs = generate_n_random_samples if generate_n_random_samples is not None else normal_samples
        _generate_samples = lambda : torch.rand((gs, 2), **set_t) * (u_range - l_range) + l_range
        if isinstance(net, MLP) and net.fit_mode == 'occupancy':
            _generate_mask = lambda x : torch.logical_and((x >= 0.5), (x <= 0.5 + 1e-3))
        else:
            _generate_mask = lambda x : torch.logical_and((x >= 0. + 1e-6), (x <= 2e-6))

        # Initialize a tensor to hold samples on the levelset of the SDF
        levelset_samples = torch.empty((0, 2), dtype=set_t['dtype'], device=torch.device('cpu'))
        levelset_normals = torch.empty((0, 2), dtype=set_t['dtype'], device=torch.device('cpu'))
        num_left = normal_samples

        print("'show_normals' set to True, starting to randomly sample SDF until enough level-set samples "
              "have been acquired.")
        num_left_progress_bar = tqdm(range(normal_samples), desc="Surface samples", leave=True)
        while num_left > 0:
            # Run indefinitely until we have acquired enough samples on the level-set surface
            samples = _generate_samples()

            # 'forward_with_coords' method allows us to compute the gradients using PyTorch Autograd
            distances, samples = net.forward_with_coords(samples)
            # detach since we don't need to retain the computation graph otherwise we quickly use up a lot of GPU memory
            normals = _gradient(samples, distances).detach()

            # Use the distances to create a mask that only retain samples and their normals if they are close
            # to the surface
            distances = distances.squeeze(1)
            mask = _generate_mask(distances)
            m_samples = samples[mask]
            m_normals = normals[mask]
            m_samples = m_samples[:min(num_left, m_samples.shape[0]), :]
            m_normals = m_normals[:m_samples.shape[0], :]

            # Append the samples and normals
            levelset_samples = torch.concatenate((levelset_samples, m_samples.cpu()), dim=0)
            levelset_normals = torch.concatenate((levelset_normals, m_normals.cpu()), dim=0)

            # final updates
            num_left -= m_samples.shape[0]
            num_left_progress_bar.update(m_samples.shape[0])
            num_left_progress_bar.set_postfix({'num_left': num_left})

    # Create the plot
    plt.figure(figsize=(8, 6))
    # adjust vmin and vmax to be equal in magnitude so that white contours represent the zero level-set in the plot
    max_abs = np.abs(dist_2d).max()
    vmin = -max_abs
    vmax = max_abs
    plt.pcolormesh(x_np, y_np, dist_2d, vmin=vmin, vmax=vmax, cmap='seismic', shading='auto')
    plt.colorbar(label="Distance")
    if show_normals:
        np_samples = to_numpy(levelset_samples)
        x_samples, y_samples = np_samples[:, 0], np_samples[:, 1]
        np_normals = to_numpy(levelset_normals)
        # Normalize and scale the normals:
        norms = np.linalg.norm(np_normals, axis=1, keepdims=True)
        np_normals_normalized = np_normals / (norms + 1e-8)
        np_normals_scaled = normal_scale * np_normals_normalized
        nx, ny = np_normals_scaled[:, 0], np_normals_scaled[:, 1]
        plt.scatter(x_samples, y_samples, color="blue", label="Points")
        # Plot the normal vectors using quiver
        plt.quiver(x_samples, y_samples, nx, ny, angles="xy",
                   scale_units="xy", scale=1, color="green")
    # Ensure equal aspect ratio
    plt.axis("equal")
    plt.title("2D Distance Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(save_path)
    plt.close()

def plot_model_with_bounds(ax, net, save_path: str, rows: int, cols: int, bl_coord: Tuple[float, float],
                           ur_coord: Tuple[float, float], crown_mode='CROWN', bound_opts: Optional[dict] = None):
    """

    :param ax:
    :param net:
    :param save_path:
    :param rows:
    :param cols:
    :param bl_coord:
    :param ur_coord:
    :return:
    """
    default_bound_opts = {
        'optimize_bound_args':
            {
                'iteration': 30,
                'lr_alpha': 1e-1,
                'keep_best': False,
                'early_stop_patience': 1e6,
                'lr_decay': 1,
                'save_loss_graphs': True}
    }
    if crown_mode.lower() == 'alpha-crown':
        default_bound_opts = {
            'optimize_bound_args':
                {
                    'iteration': 30,
                    'lr_alpha': 1e-1,
                    'keep_best': False,
                    'early_stop_patience': 1e6,
                    'lr_decay': 1,
                    'save_loss_graphs': True}
        }
        reuse_alpha = True
        bounded_net = BoundedModule(net, torch.empty((1, 2)), bound_opts=bound_opts if bound_opts else default_bound_opts)
    else:
        reuse_alpha = False
        bounded_net = BoundedModule(net, torch.empty((1, 2)))  # , bound_opts={'relu': 'same-slope'})
    needed_A_dict = defaultdict(set)
    output_name = bounded_net.output_name[0]
    input_name = bounded_net.input_name[0]
    needed_A_dict[output_name].add(input_name)

    # unpack the bottom left and upper right coordinates
    x_min, y_min = bl_coord
    x_max, y_max = ur_coord

    # Calculate step size for each grid cell
    x_step = (x_max - x_min) / cols
    y_step = (y_max - y_min) / rows

    # Draw the grid
    for row in range(rows + 1):
        y = y_min + row * y_step
        ax.axhline(y, color='black', linewidth=0.5)
    for col in range(cols + 1):
        x = x_min + col * x_step
        ax.axvline(x, color='black', linewidth=0.5)

    # Add diagonal lines to each cell using custom coordinates
    for row in range(rows):
        for col in range(cols):
            # TODO: This function is not finished as of yet. This section of code should iterate through each
            # cell, generate a hyperplane on this cell, and display it in the cell if the plane intersects.
            x_start = x_min + col * x_step
            x_end = x_start + x_step
            y_start = y_min + row * y_step
            y_end = y_start + y_step
            x_L = torch.tensor([x_start, y_start])
            x_U = torch.tensor([x_end, y_end])
            box_center = (x_U + x_L)/2
            ptb = PerturbationLpNorm(x_L=x_L, x_U=x_U)
            bounded_x = BoundedTensor(box_center, ptb)
            result = bounded_net.compute_bounds(x=(bounded_x,), method=crown_mode, needed_A_dict=needed_A_dict,
                                                bound_lower=True, bound_upper=True,
                                                return_A=True, reuse_alpha=reuse_alpha)  # dynamic forward
            may_lower, may_upper, A_dict = result
            lA = A_dict[output_name][input_name]['lA']
            lbias = A_dict[output_name][input_name]['lbias']
            uA = A_dict[output_name][input_name]['uA']
            ubias = A_dict[output_name][input_name]['ubias']
            # Example diagonal: bottom-left to top-right
            ax.plot([x_start, x_end], [y_start, y_end], color='red', linewidth=0.7)

    # Set axis limits and aspect ratio
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')  # Ensures square cells

    # Align tick marks and values with grid lines
    x_ticks = np.arange(x_min, x_max + x_step, x_step)
    y_ticks = np.arange(y_min, y_max + y_step, y_step)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Add labels for clarity (optional)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Optionally, add tick labels
    ax.set_xticklabels([f"{x:.1f}" for x in x_ticks])
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])

    return ax

def main(args: dict):
    # extract parsed arguments
    input_file = args['input_file']
    output_file = args['output_file']
    model_type = args['model_type']
    dim_samples = args['dim_samples']
    normal_samples = args['normal_samples']
    display_normals = args['display_normals']
    normal_scale = args['normal_scale']
    l_range = args['l_range']
    u_range = args['u_range']
    generate_n_random_samples = args['generate_n_random_samples']
    rows = args['rows']
    cols = args['cols']
    x_L = tuple(args['x_L'])
    x_U = tuple(args['x_U'])
    crown_mode = args['crown_mode']

    # load in the model
    net = load_net_object(input_file, model_type, device=set_t['device'])
    net = net.to(device=set_t['device'])

    # sample the model and generate a 2D plot
    sample_model_args = {
        'net': net,
        'save_path': output_file,
        'show_normals': display_normals,
        'normal_scale': normal_scale,
        'dim_samples': dim_samples,
        'normal_samples': normal_samples,
        'l_range': l_range,
        'u_range': u_range,
        'generate_n_random_samples': generate_n_random_samples
    }
    sample_model(**sample_model_args)

    # TODO: Finish the plot_model_with_bounds function
    # second_output_file = output_file.split('.png')[0] + '_bounded.png'
    # fig, ax = plt.subplots(figsize=(8, 8))
    # plot_model_with_bounds(ax, net, second_output_file, rows, cols, x_L, x_U, crown_mode)

    return

    # # Example usage
    # width, height = 500, 500
    # num_samples = 1000
    # t_samples = np.linspace(0, 2 * np.pi, num_samples)
    # center = (0.5, 0.5)
    # radius = 1.
    # theta_samples = np.random.uniform(low=0., high=2 * np.pi, size=num_samples)
    # xv = (radius * np.cos(theta_samples)) + center[0]
    # yv = (radius * np.sin(theta_samples)) + center[1]
    # xy_samples = np.stack([xv, yv], axis=1)
    # print(f"shape xy_samples {xy_samples.shape}")
    # circle_sdf = init_circle_sdf(center, radius)
    #
    # # # Render the parametric image
    # # image = render_parametric_curve_image(width, height, circle_sdf, t_samples, line_thickness=2, scale=0.45,
    # #                                       rgb_mode=False)
    # # img = Image.fromarray(image)
    # # img.save("../parametric_renderings/parametric_curve.png")
    # # image = render_parametric_curve_image(width, height, circle_sdf, t_samples, line_thickness=2, scale=0.45,
    # #                                       rgb_mode=True)
    # # img = Image.fromarray(image)
    # # img.save("../parametric_renderings/parametric_curve_rgb.png")
    #
    # # Render the sdf image
    # image = render_sdf_image(width, height, circle_sdf, xy_samples, line_thickness=2, scale=0.45,
    #                          rgb_mode=False)
    # img = Image.fromarray(image)
    # img.save("../parametric_renderings/sdf_curve.png")
    # # image = render_sdf_image(width, height, circle_sdf, xy_samples, line_thickness=2, scale=0.45,
    # #                                       rgb_mode=True)
    # # img = Image.fromarray(image)
    # # img.save("../parametric_renderings/sdf_curve_rgb.png")

def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str,
                        help="The path to the .pth model from the root directory.")
    parser.add_argument("output_file", type=str,
                        help="The path to save the images rendered images to.")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Must specify if the model is one of the following: [mlp, siren].")
    parser.add_argument("--dim_samples", type=int, default=1000,
                        help="The number of evenly spaced samples to draw from the model along each dimension "
                             "to create a mesh grid.")
    parser.add_argument("--normal_samples", type=int, default=1000,
                        help="The number of samples to draw from the model that lie on the surface to plot their "
                             "normals.")
    parser.add_argument("--generate_n_random_samples", type=int,
                        help="The number of samples to randomly generate per iteration to help gather the normal"
                             "samples. This helps with more flexibility if you do not want many surface normal samples"
                             "but you need to generally sample a lot to get surface level samples.")
    parser.add_argument("--display_normals", action="store_true",
                        help="Will sample the SDF on the zero level-set and calculate its normals to display in the "
                             "plot.")
    parser.add_argument("--normal_scale", type=float, default=1.0,
                        help="If normals are displayed, then their magnitudes are normalized and multiplied by this "
                             "scaling factor. This is to help make the normals appear visually clear in the plot.")
    parser.add_argument("--rows", type=int, default=8,
                        help="Number of rows to slice the input region for bounding a neural SDF.")
    parser.add_argument("--cols", type=int, default=8,
                        help="Number of columns to slice the input region for bounding a neural SDF.")
    parser.add_argument("--l_range", type=float, default=-0.55,
                        help="Lower range to use for uniform sampling.")
    parser.add_argument("--u_range", type=float, default=0.55,
                        help="Upper range to use for uniform sampling.")
    parser.add_argument("--x_L", type=float, nargs='+', default=[-1., -1.],
                        help="Bottom left point of the input bounding box.")
    parser.add_argument("--x_U", type=float, nargs='+', default=[1., 1.],
                        help="Upper right point of the input bounding box.")
    parser.add_argument("--crown_mode", type=str, default='CROWN',
                        help="Bounding method to use on the neural SDF.")

    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict

if __name__ == "__main__":
    print(f"set_t: {set_t}")
    parsed_args = parse_args()
    main(parsed_args)