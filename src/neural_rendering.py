"""
Main script for taking a pretrained SDF/occupancy based neural network and visualizing their output.
"""
import argparse
import tqdm
import numpy as np
import torch
from sympy import andre
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
from neural_sdf import MLP, Siren
from neural_utils import load_net_object
import crown
import mlp
import kd_tree
from shapely.ops import split, unary_union
import shapely
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import copy
from matplotlib.patches import PathPatch
from matplotlib.path import Path

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

to_numpy = lambda x : x.detach().cpu().numpy()

class BaseExactSDF:
    def get_contour_coordinates(self):
        pass

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
                 dim_samples: int = 500):
    """
    Generates a heat map plot of a neural SDF
    :param net:             SDF Net object
    :param save_path:       Path to save plot to
    :param show_normals:    If true, also displays the normals of the points on the surface
    :param normal_scale:    Scale the normals by the given value after they have been normalized
    :param dim_samples:     Number of samples along each dimension
    :return:
    """
    # from matplotlib.patches import Circle
    x_np = np.linspace(-0.53, 0.53, dim_samples)  # 100 points along the x-axis
    y_np = np.linspace(-0.53, 0.53, dim_samples)  # 100 points along the y-axis
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

        _generate_samples = lambda : torch.rand((dim_samples**2, 2), **set_t) - 0.5

        # Initialize a tensor to hold samples on the levelset of the SDF
        levelset_samples = torch.empty((0, 2), **set_t)
        levelset_normals = torch.empty((0, 2), **set_t)
        num_left = dim_samples

        print("'show_normals' set to True, starting to randomly sample SDF until enough levelset samples "
              "have been acquired.")
        num_left_progress_bar = tqdm(range(dim_samples), desc="Surface samples", leave=True)
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
            mask = torch.logical_and((distances >= 0.), (distances <= 1e-6))
            m_samples = samples[mask]
            m_normals = normals[mask]
            m_samples = m_samples[:min(num_left, m_samples.shape[0]), :]
            m_normals = m_normals[:m_samples.shape[0], :]

            # Append the samples and normals
            levelset_samples = torch.concatenate((levelset_samples, m_samples), dim=0)
            levelset_normals = torch.concatenate((levelset_normals, m_normals), dim=0)

            # final updates
            num_left -= m_samples.shape[0]
            num_left_progress_bar.update(m_samples.shape[0])
            num_left_progress_bar.set_postfix({'num_left': num_left})

    # Create the plot
    plt.figure(figsize=(8, 8))
    # adjust vmin and vmax to be equal in magnitude so that white contours represent the zero level-set in the plot
    max_abs = np.abs(dist_2d).max()
    vmin = -max_abs
    vmax = max_abs
    plt.pcolormesh(x_np, y_np, dist_2d, vmin=vmin, vmax=vmax, cmap='seismic', shading='auto')
    # plt.colorbar(label="Distance")
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
    # plt.axis("equal")

    x = np.linspace(-0.53, 0.53, 1250)
    y = np.linspace(-0.53, 0.53, 1250)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Compute SDF values for the grid points
    sdf_values = net(torch.from_numpy(grid_points).float().cuda()).detach().cpu().numpy().flatten()

    # Select points where |SDF| < epsilon
    near_surface = np.abs(sdf_values) < 0.0005
    surface_points = grid_points[near_surface]

    # Plot the points
    plt.scatter(surface_points[:, 0], surface_points[:, 1], color='gray', alpha=0.8, s=0.5, label='SDF ≈ 0')

    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def project_line_onto_square(a1, a2, b, x1_min, x1_max, x2_min, x2_max):
    # Define the bounding box (square)
    square = shapely.geometry.box(x1_min, x2_min, x1_max, x2_max)

    # Define the line equation a1*x1 + a2*x2 + b = 0 in explicit form
    if a2 != 0:
        # Express x2 as a function of x1
        line = shapely.geometry.LineString([
            (x1_min, (-a1*x1_min - b) / a2),
            (x1_max, (-a1*x1_max - b) / a2)
        ])
    else:
        # Vertical line case: x1 = constant
        x1 = -b / a1
        line = shapely.geometry.LineString([(x1, x2_min), (x1, x2_max)])

    # Intersect the line with the square
    segment = line.intersection(square)

    return segment


def carve(ax, net: MLP, deep=False, smoothify=False):
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    lower = torch.tensor([-0.53, -0.53])
    upper = torch.tensor([0.53,0.53])
    func = crown.CrownImplicitFunction(mlp.func_from_spec(mode='default'), net, crown_mode='crown', input_dim=2)
    if deep:
        lowers, uppers, lAs, lbs, uAs, ubs, pos_lowers, pos_uppers, neg_lowers, neg_uppers = kd_tree.construct_hybrid_unknown_tree(
            func, net, lower, upper, base_depth=12, max_depth=15, node_dim=2, include_pos_neg=True)
    else:
        lowers, uppers, lAs, lbs, uAs, ubs, pos_lowers, pos_uppers, neg_lowers, neg_uppers = kd_tree.construct_hybrid_unknown_tree(
            func, net, lower, upper, base_depth=9, max_depth=11, node_dim=2, include_pos_neg=True)
    lowers = lowers.detach().cpu().numpy()
    uppers = uppers.detach().cpu().numpy()
    lAs = lAs.detach().cpu().numpy()
    lbs = lbs.detach().cpu().numpy()
    uAs = uAs.detach().cpu().numpy()
    ubs = ubs.detach().cpu().numpy()
    pos_lowers = pos_lowers.detach().cpu().numpy()
    pos_uppers = pos_uppers.detach().cpu().numpy()
    neg_lowers = neg_lowers.detach().cpu().numpy()
    neg_uppers = neg_uppers.detach().cpu().numpy()

    # polygon_list = []
    outer_shell = shapely.geometry.Polygon([(-0.53, -0.53), (-0.53, 0.53), (0.53, 0.53), (0.53, -0.53)])
    inner_shell = shapely.geometry.Polygon([(-0.53, -0.53), (-0.53, 0.53), (0.53, 0.53), (0.53, -0.53)])
    for p_l, p_u in zip(pos_lowers, pos_uppers):
        # patch = matplotlib.patches.Polygon([p_l, (p_l[0], p_u[1]), p_u, (p_u[0], p_l[1])], edgecolor='grey',
        #                                    facecolor='none', linestyle='--', linewidth=0.5)
        # ax.add_patch(patch)
        outer_shell = outer_shell.difference(shapely.geometry.Polygon([p_l, (p_l[0], p_u[1]), p_u, (p_u[0], p_l[1])]))
        inner_shell = inner_shell.difference(shapely.geometry.Polygon([p_l, (p_l[0], p_u[1]), p_u, (p_u[0], p_l[1])]))


    squares = []
    outer_segments = []
    outer_segments_lAs = []
    outer_segments_lbs = []
    inner_segments = []
    outer_polygons = []
    inner_polygons = []
    inner_segments_uAs = []
    inner_segments_ubs = []

    for l, u, lA, lb, uA, ub in zip(lowers, uppers, lAs, lbs, uAs, ubs):
        patch = matplotlib.patches.Polygon([l, (l[0], u[1]), u, (u[0], l[1])], edgecolor='grey', facecolor='none',
                                           linestyle='--', linewidth=0.5)
        ax.add_patch(patch)
        square = shapely.geometry.Polygon([l, (l[0], u[1]), u, (u[0], l[1])])
        squares.append(square)
        outer_line = project_line_onto_square(lA[0], lA[1], lb, -0.53, 0.53, -0.53, 0.53)
        inner_line = project_line_onto_square(uA[0], uA[1], ub, -0.53, 0.53, -0.53, 0.53)


        # For each node and its outer_line segment, get its neighbors that the outer_line segment also intersect with
        outer_segment = shapely.intersection(square, outer_line)
        inner_segment = shapely.intersection(square, inner_line)
        if len(np.array(outer_segment.coords)) == 2:
            outer_segments.append(outer_segment)
            outer_segments_lAs.append(lA)
            outer_segments_lbs.append(lb)
        if len(np.array(inner_segment.coords)) == 2:
            inner_segments.append(inner_segment)
            inner_segments_uAs.append(uA)
            inner_segments_ubs.append(ub)

        # print(outer_qualified_neighbors)
        slices1 = split(square, outer_line)
        slices2 = split(square, inner_line)

        for g in slices1.geoms:
            if g.geom_type == 'Polygon':
                c = shapely.centroid(g)
                c = np.array([c.x, c.y])
                cls = np.dot(lA, c) + lb
                if cls > 0.:
                    outer_shell = outer_shell.difference(g)
                else:
                    outer_polygons.append(g)

        for g in slices2.geoms:
            if g.geom_type == 'Polygon':
                c = shapely.centroid(g)
                c = np.array([c.x, c.y])
                cls = np.dot(uA, c) + ub
                if cls > 0.:
                    inner_shell = inner_shell.difference(g)
                    inner_polygons.append(g)

        for g1 in slices1.geoms:
            for g2 in slices2.geoms:
                intersection = shapely.intersection(g1, g2)
                if intersection.geom_type == 'Polygon':
                    c = shapely.centroid(intersection)
                    if not len(list(c.coords)) == 0:
                        c = np.array([c.x, c.y])
                        cls1 = np.dot(lA, c) + lb
                        cls2 = np.dot(uA, c) + ub
                        # if cls1 < 0 and cls2 > 0:
                        #     if intersection.area > 0.00005:
                        #         patch = matplotlib.patches.Polygon(intersection.exterior.coords, edgecolor='none',
                        #                                            facecolor='lightblue', linewidth=2)
                        #         ax.add_patch(patch)

    outer_qualified_neighbors = []
    outer_contact_points = []

    for outer_segment in outer_segments:
        neighbors_buffer = []
        points_buffer = []
        for outer_polygon, lA, lb in zip(outer_polygons, lAs, lbs):
            segment_polygon_intersection = shapely.intersection(outer_polygon, outer_segment)
            if segment_polygon_intersection.geom_type == 'Point':
                p = np.array([segment_polygon_intersection.x, segment_polygon_intersection.y])
                cls = np.dot(lA, p) + lb
                if cls <= 0:
                    neighbors_buffer.append(outer_polygon)
                    points_buffer.append(segment_polygon_intersection)

        outer_qualified_neighbors.append(neighbors_buffer)
        outer_contact_points.append(points_buffer)

    inner_qualified_neighbors = []
    inner_contact_points = []

    for inner_segment in inner_segments:
        neighbors_buffer = []
        points_buffer = []
        for inner_polygon, uA, ub in zip(inner_polygons, uAs, ubs):
            segment_polygon_intersection = shapely.intersection(inner_polygon, inner_segment)
            if segment_polygon_intersection.geom_type == 'Point':
                p = np.array([segment_polygon_intersection.x, segment_polygon_intersection.y])
                cls = np.dot(uA, p) + ub
                if cls >= 0:
                    neighbors_buffer.append(inner_polygon)
                    points_buffer.append(segment_polygon_intersection)

        inner_qualified_neighbors.append(neighbors_buffer)
        inner_contact_points.append(points_buffer)

    outer_added_polygons = []
    for outer_segment, neighbors_buffer, points_buffer, lA, lb in zip(outer_segments, outer_qualified_neighbors,
                                                                      outer_contact_points, outer_segments_lAs,
                                                                      outer_segments_lbs):
        endpoint_A = list(outer_segment.coords)[0]
        endpoint_B = list(outer_segment.coords)[1]
        if len(neighbors_buffer) == 2:
            poly_A = neighbors_buffer[0]
            poly_B = neighbors_buffer[1]
            point_A = points_buffer[0]
            point_B = points_buffer[1]
            if (point_A.x != endpoint_A[0]) and (point_A.y != endpoint_A[1]) and (point_A.x != endpoint_B[0]) and \
                (point_A.y != endpoint_B[1]) and (point_B.x != endpoint_A[0]) and (point_B.y != endpoint_A[1]) and \
                    (point_B.x != endpoint_B[0]) and (point_B.y != endpoint_B[1]):
                continue
            else:
                vertices_A = list(poly_A.exterior.coords)
                vertices_B = list(poly_B.exterior.coords)
                for v_A in vertices_A:
                    if point_A.x == v_A[0] or point_A.y == v_A[1]:
                        if np.dot(lA, v_A) + lb > 0.:
                            point_A_new = shapely.geometry.Point(v_A)
                for v_B in vertices_B:
                    if point_B.x == v_B[0] or point_B.y == v_B[1]:
                        if np.dot(lA, v_B) + lb > 0.:
                            point_B_new = shapely.geometry.Point(v_B)

                added_poly = shapely.geometry.Polygon(
                    ((point_A.x, point_A.y), (point_B.x, point_B.y),
                     (point_B_new.x, point_B_new.y), (point_A_new.x, point_A_new.y))
                )
                outer_added_polygons.append(added_poly)
        elif len(neighbors_buffer) == 1:
            poly_A = neighbors_buffer[0]
            point_A = points_buffer[0]
            if (point_A.x != endpoint_A[0]) and (point_A.y != endpoint_A[1]) and (point_A.x != endpoint_B[0]) and \
                    (point_A.y != endpoint_B[1]):
                continue
            else:
                vertices_A = list(poly_A.exterior.coords)
                for v_A in vertices_A:
                    if point_A.x == v_A[0] or point_A.y == v_A[1]:
                        if np.dot(lA, v_A) + lb > 0.:
                            point_A_new = shapely.geometry.Point(v_A)
                unchanged_point = outer_segment.boundary.geoms[0] if shapely.equals(point_A,
                                                                                    outer_segment.boundary.geoms[1]) else \
                outer_segment.boundary.geoms[1]
                added_poly = shapely.geometry.Polygon(
                    ((unchanged_point.x, unchanged_point.y), (point_A.x, point_A.y), (point_A_new.x, point_A_new.y))
                )
                outer_added_polygons.append(added_poly)

    inner_added_polygons = []
    for inner_segment, neighbors_buffer, points_buffer, uA, ub in zip(inner_segments, inner_qualified_neighbors,
                                                                      inner_contact_points, inner_segments_uAs,
                                                                      inner_segments_ubs):
        if len(neighbors_buffer) == 2:
            poly_A = neighbors_buffer[0]
            poly_B = neighbors_buffer[1]
            point_A = points_buffer[0]
            point_B = points_buffer[1]
            vertices_A = list(poly_A.exterior.coords)
            vertices_B = list(poly_B.exterior.coords)
            for v_A in vertices_A:
                if point_A.x == v_A[0] or point_A.y == v_A[1]:
                    if np.dot(uA, v_A) + ub < 0.:
                        point_A_new = shapely.geometry.Point(v_A)
            for v_B in vertices_B:
                if point_B.x == v_B[0] or point_B.y == v_B[1]:
                    if np.dot(uA, v_B) + ub < 0.:
                        point_B_new = shapely.geometry.Point(v_B)

            added_poly = shapely.geometry.Polygon(
                ((point_A.x, point_A.y), (point_B.x, point_B.y),
                 (point_B_new.x, point_B_new.y), (point_A_new.x, point_A_new.y))
            )
            inner_added_polygons.append(added_poly)
        elif len(neighbors_buffer) == 1:
            poly_A = neighbors_buffer[0]
            point_A = points_buffer[0]
            vertices_A = list(poly_A.exterior.coords)
            for v_A in vertices_A:
                if point_A.x == v_A[0] or point_A.y == v_A[1]:
                    if np.dot(uA, v_A) + ub < 0.:
                        point_A_new = shapely.geometry.Point(v_A)
            unchanged_point = inner_segment.boundary.geoms[0] if shapely.equals(point_A,
                                                                                inner_segment.boundary.geoms[1]) else \
                inner_segment.boundary.geoms[1]
            added_poly = shapely.geometry.Polygon(
                ((unchanged_point.x, unchanged_point.y), (point_A.x, point_A.y), (point_A_new.x, point_A_new.y))
            )
            inner_added_polygons.append(added_poly)

    if smoothify:
        for poly in outer_added_polygons:
            # patch = matplotlib.patches.Polygon(poly.exterior.coords, edgecolor='none', facecolor='lightblue',
            #                                    linewidth=2)
            # ax.add_patch(patch)
            outer_shell = outer_shell.union(poly)


        for poly in inner_added_polygons:
            # patch = matplotlib.patches.Polygon(poly.exterior.coords, edgecolor='none', facecolor='lightblue',
            #                                    linewidth=2)
            # ax.add_patch(patch)
            inner_shell = inner_shell.difference(poly)

    if outer_shell.geom_type == 'Polygon':
        patch = matplotlib.patches.Polygon(outer_shell.exterior.coords, edgecolor='purple', facecolor='none', linewidth=2)
        ax.add_patch(patch)
        for hole in outer_shell.interiors:
            patch = matplotlib.patches.Polygon(hole.coords, edgecolor='purple', facecolor='none',
                                               linewidth=2)
            ax.add_patch(patch)
            outer_shell = outer_shell.union(hole)
    elif outer_shell.geom_type == 'MultiPolygon':
        for poly in outer_shell.geoms:
            patch = matplotlib.patches.Polygon(poly.exterior.coords, edgecolor='purple', facecolor='none',
                                               linewidth=2)
            ax.add_patch(patch)
            for hole in poly.interiors:
                patch = matplotlib.patches.Polygon(hole.coords, edgecolor='purple', facecolor='none',
                                                   linewidth=2)
                ax.add_patch(patch)
    else:
        raise NotImplementedError("Plotting of other geometries not implemented.")

    inner_shell_holes = []
    if inner_shell.geom_type == 'Polygon':
        patch = matplotlib.patches.Polygon(inner_shell.exterior.coords, edgecolor='green', facecolor='none', linewidth=2)
        ax.add_patch(patch)
        for hole in inner_shell.interiors:
            patch = matplotlib.patches.Polygon(hole.coords, edgecolor='green', facecolor='none',
                                               linewidth=2)
            ax.add_patch(patch)
    elif inner_shell.geom_type == 'MultiPolygon':
        for poly in inner_shell.geoms:
            patch = matplotlib.patches.Polygon(poly.exterior.coords, edgecolor='green', facecolor='none',
                                               linewidth=2)
            ax.add_patch(patch)
            for hole in poly.interiors:
                inner_shell_holes.append(hole)
            #     hole_area = shapely.Polygon(hole.coords).area
            #     if hole_area < 0.005:
            #         patch = matplotlib.patches.Polygon(hole.coords, edgecolor='gray', facecolor='white',
            #                                            linewidth=2)
            #         ax.add_patch(patch)
    else:
        raise NotImplementedError("Plotting of other geometries not implemented.")


    filled_region = outer_shell.difference(inner_shell)
    if filled_region.geom_type == 'MultiPolygon':
        filled_region = max(filled_region.geoms, key=lambda g: g.area)
    # Ensure we always work with a MultiPolygon for iteration
    if filled_region.geom_type == 'Polygon':
        polygons = [filled_region]
    elif filled_region.geom_type == 'MultiPolygon':
        polygons = list(filled_region.geoms)
    else:
        raise TypeError(f"Unexpected geometry type: {type(filled_region)}")

    for poly in polygons:
        paths = []

        # Exterior ring
        exterior = np.asarray(poly.exterior.coords)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(exterior) - 2) + [Path.CLOSEPOLY]
        paths.append(Path(exterior, codes))

        # Interior rings (holes)
        for interior in poly.interiors:
            # if interior.area > 0.00005:
            ring = np.asarray(interior.coords)
            codes = [Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY]
            paths.append(Path(ring, codes))

        compound_path = Path.make_compound_path(*paths)
        patch = PathPatch(compound_path, facecolor='lightblue', edgecolor='none', alpha=0.5)
        ax.add_patch(patch)

    for p_l, p_u in zip(pos_lowers, pos_uppers):
        patch = matplotlib.patches.Polygon([p_l, (p_l[0], p_u[1]), p_u, (p_u[0], p_l[1])], edgecolor='grey',
                                           facecolor='none', linestyle='--', linewidth=0.5)
        ax.add_patch(patch)

    for n_l, n_u in zip(neg_lowers, neg_uppers):
        patch = matplotlib.patches.Polygon([n_l, (n_l[0], n_u[1]), n_u, (n_u[0], n_l[1])], edgecolor='grey',
                                           facecolor='none', linestyle='--', linewidth=0.5)
        ax.add_patch(patch)

    x = np.linspace(-0.53, 0.53, 1250)
    y = np.linspace(-0.53, 0.53, 1250)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Compute SDF values for the grid points
    sdf_values = net(torch.from_numpy(grid_points).float().cuda()).detach().cpu().numpy().flatten()

    # Select points where |SDF| < epsilon
    near_surface = np.abs(sdf_values) < 0.0005
    surface_points = grid_points[near_surface]

    # Plot the points
    ax.scatter(surface_points[:, 0], surface_points[:, 1], color='gray', alpha=0.8, s=0.5, label='SDF ≈ 0')

    # axins.set_xlim(0.3, 0.4)
    # axins.set_ylim(0.3, 0.4)

    # Indicate the zoom region
    # ax.indicate_inset_zoom(axins, edgecolor="black")

    return ax

def fill(ax, net: MLP, deep=False):
    lower = torch.tensor([-0.53, -0.53])
    upper = torch.tensor([0.53, 0.53])
    func = crown.CrownImplicitFunction(mlp.func_from_spec(mode='default'), net, crown_mode='crown', input_dim=2)
    if deep:
        lowers, uppers, lAs, lbs, uAs, ubs, pos_lowers, pos_uppers, neg_lowers, neg_uppers = kd_tree.construct_hybrid_unknown_tree(
            func, net, lower, upper, base_depth=9, max_depth=12, node_dim=2, include_pos_neg=True)
    else:
        lowers, uppers, lAs, lbs, uAs, ubs, pos_lowers, pos_uppers, neg_lowers, neg_uppers = kd_tree.construct_hybrid_unknown_tree(
            func, net, lower, upper, base_depth=9, max_depth=11, node_dim=2, include_pos_neg=True)
    lowers = lowers.detach().cpu().numpy()
    uppers = uppers.detach().cpu().numpy()
    lAs = lAs.detach().cpu().numpy()
    lbs = lbs.detach().cpu().numpy()
    uAs = uAs.detach().cpu().numpy()
    ubs = ubs.detach().cpu().numpy()
    pos_lowers = pos_lowers.detach().cpu().numpy()
    pos_uppers = pos_uppers.detach().cpu().numpy()
    neg_lowers = neg_lowers.detach().cpu().numpy()
    neg_uppers = neg_uppers.detach().cpu().numpy()

    # polygon_list = []
    outer_shell = shapely.geometry.Polygon([(-0.53, -0.53), (-0.53, 0.53), (0.53, 0.53), (0.53, -0.53)])
    inner_shell = shapely.geometry.Polygon([(-0.53, -0.53), (-0.53, 0.53), (0.53, 0.53), (0.53, -0.53)])
    for p_l, p_u in zip(pos_lowers, pos_uppers):
        patch = matplotlib.patches.Polygon([p_l, (p_l[0], p_u[1]), p_u, (p_u[0], p_l[1])], edgecolor='grey',
                                           facecolor='none', linestyle='--', linewidth=0.5)
        ax.add_patch(patch)
        outer_shell = outer_shell.difference(shapely.geometry.Polygon([p_l, (p_l[0], p_u[1]), p_u, (p_u[0], p_l[1])]))
        inner_shell = inner_shell.difference(shapely.geometry.Polygon([p_l, (p_l[0], p_u[1]), p_u, (p_u[0], p_l[1])]))

    for n_l, n_u in zip(neg_lowers, neg_uppers):
        patch = matplotlib.patches.Polygon([n_l, (n_l[0], n_u[1]), n_u, (n_u[0], n_l[1])], edgecolor='grey',
                                           facecolor='none', linestyle='--', linewidth=0.5)
        ax.add_patch(patch)


    for l, u in zip(lowers, uppers):
        patch = matplotlib.patches.Polygon([l, (l[0], u[1]), u, (u[0], l[1])], edgecolor='grey', facecolor='lightblue',
                                           linestyle='--', linewidth=0.5, alpha=0.5)
        ax.add_patch(patch)
        inner_shell = inner_shell.difference(shapely.geometry.Polygon([l, (l[0], u[1]), u, (u[0], l[1])]))


    if outer_shell.geom_type == 'Polygon':
        patch = matplotlib.patches.Polygon(outer_shell.exterior.coords, edgecolor='purple', facecolor='none', linewidth=2)
        ax.add_patch(patch)
        for hole in outer_shell.interiors:
            patch = matplotlib.patches.Polygon(hole.coords, edgecolor='purple', facecolor='none',
                                               linewidth=2)
            ax.add_patch(patch)
    elif outer_shell.geom_type == 'MultiPolygon':
        for poly in outer_shell.geoms:
            patch = matplotlib.patches.Polygon(poly.exterior.coords, edgecolor='purple', facecolor='none',
                                               linewidth=2)
            ax.add_patch(patch)
            for hole in poly.interiors:
                patch = matplotlib.patches.Polygon(hole.coords, edgecolor='purple', facecolor='none',
                                                   linewidth=2)
                ax.add_patch(patch)
    else:
        raise NotImplementedError("Plotting of other geometries not implemented.")

    if inner_shell.geom_type == 'Polygon':
        patch = matplotlib.patches.Polygon(inner_shell.exterior.coords, edgecolor='green', facecolor='none', linewidth=2)
        ax.add_patch(patch)
        for hole in inner_shell.interiors:
            patch = matplotlib.patches.Polygon(hole.coords, edgecolor='green', facecolor='none',
                                               linewidth=2)
            ax.add_patch(patch)
    elif inner_shell.geom_type == 'MultiPolygon':
        for poly in inner_shell.geoms:
            patch = matplotlib.patches.Polygon(poly.exterior.coords, edgecolor='green', facecolor='none',
                                               linewidth=2)
            ax.add_patch(patch)
            for hole in poly.interiors:
                patch = matplotlib.patches.Polygon(hole.coords, edgecolor='green', facecolor='none',
                                                   linewidth=2)
                ax.add_patch(patch)
    else:
        raise NotImplementedError("Plotting of other geometries not implemented.")


    x = np.linspace(-0.53, 0.53, 1250)
    y = np.linspace(-0.53, 0.53, 1250)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Compute SDF values for the grid points
    sdf_values = net(torch.from_numpy(grid_points).float().cuda()).detach().cpu().numpy().flatten()

    # Select points where |SDF| < epsilon
    near_surface = np.abs(sdf_values) < 0.0005
    surface_points = grid_points[near_surface]

    # Plot the points
    ax.scatter(surface_points[:, 0], surface_points[:, 1], color='gray', alpha=0.8, s=0.5, label='SDF ≈ 0')
    return ax

def dilation_erosion(ax, net: MLP, resolution=2**4+1, threshold=0.0, delta=0.1):
    x = np.linspace(-0.53, 0.53, resolution)
    y = np.linspace(-0.53, 0.53, resolution)
    xx, yy = np.meshgrid(x, y)
    sdf_values = np.zeros_like(xx)

    # Evaluate the SDF function at each grid point
    for i in range(resolution):
        for j in range(resolution):
            sdf_values[i, j] = net(
                torch.tensor((xx[i, j], yy[i, j])).unsqueeze(0).float().cuda()).detach().cpu().numpy().flatten()

    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Use matplotlib's built-in contour to extract and plot the mesh
    contour = ax.contour(
        xx, yy, sdf_values, levels=[threshold-delta, threshold+delta], colors=['green', 'purple'], linewidths=2
    )

    # --- Fill region between contours ---
    # Assumes contour.collections[0] is inner (green), [1] is outer (purple)
    if len(contour.collections) >= 2:
        inner_paths = contour.collections[0].get_paths()
        outer_paths = contour.collections[1].get_paths()

        # For now, just use the largest path from each (usually the first one)
        if inner_paths and outer_paths:
            inner = inner_paths[0].vertices
            outer = outer_paths[0].vertices

            # Combine the paths (outer + reversed inner)
            verts = np.vstack([outer, inner[::-1]])
            codes = (
                    [Path.MOVETO] + [Path.LINETO] * (len(outer) - 1) +
                    [Path.MOVETO] + [Path.LINETO] * (len(inner) - 1)
            )
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor='lightblue', edgecolor='none', alpha=0.5)
            ax.add_patch(patch)

    x_ = np.linspace(-0.53, 0.53, 1250)
    y_ = np.linspace(-0.53, 0.53, 1250)
    xx_, yy_ = np.meshgrid(x_, y_)
    grid_points = np.stack([xx_.ravel(), yy_.ravel()], axis=-1)

    # Compute SDF values for the grid points
    sdf_values_ = net(torch.from_numpy(grid_points).float().cuda()).detach().cpu().numpy().flatten()

    # Select points where |SDF| < epsilon
    near_surface = np.abs(sdf_values_) < 0.0005
    surface_points = grid_points[near_surface]

    # Plot the points
    ax.scatter(surface_points[:, 0], surface_points[:, 1], color='gray', alpha=0.8, s=0.5, label='SDF ≈ 0')

    return contour, x, y

def main(args: dict):
    # extract parsed arguments
    input_file = args['input_file']
    output_file = args['output_file']
    model_type = args['model_type']
    dim_samples = args['dim_samples']
    display_normals = args['display_normals']
    normal_scale = args['normal_scale']
    rows = args['rows']
    cols = args['cols']
    x_L = tuple(args['x_L'])
    x_U = tuple(args['x_U'])
    crown_mode = args['crown_mode']
    deep = args['deep']
    print(deep)

    # load in the model
    net = load_net_object(input_file, model_type, device=set_t['device'])
    # net = star_sdf
    # net = net.to(device=set_t['device'])

    # sample the model and generate a 2D plot
    # sample_model_args = {
    #     'net': net,
    #     'save_path': output_file,
    #     'show_normals': display_normals,
    #     'normal_scale': normal_scale,
    #     'dim_samples': dim_samples,
    # }
    # sample_model(**sample_model_args)

    # TODO: Finish the plot_model_with_bounds function
    # second_output_file = output_file.split('.pdf')[0] + '_CROWN.pdf'
    # fig, ax = plt.subplots(figsize=(8, 8))
    # carve(ax, net, deep, smoothify=True)
    # plt.xlim(-0.53, 0.53)
    # plt.ylim(-0.53, 0.53)
    #
    # if True:
    #     axins = zoomed_inset_axes(ax, 8, loc=4)
    #     for patch in ax.patches:
    #         patch_cpy = copy.copy(patch)
    #         # cut the umbilical cord the hard way
    #         patch_cpy.axes = None
    #         patch_cpy.figure = None
    #         patch_cpy.set_transform(axins.transData)
    #         axins.add_patch(patch_cpy)
    #
    #     for collection in ax.get_children():
    #         if isinstance(collection, matplotlib.collections.PathCollection):  # This ensures it's a scatter plot
    #             offsets = collection.get_offsets()
    #             colors = collection.get_facecolors()
    #             sizes = collection.get_sizes() * 8
    #             axins.scatter(offsets[:, 0], offsets[:, 1],
    #                           color=colors, s=sizes)
    #
    #     axins.set_xlim(0.06, 0.11)
    #     axins.set_ylim(0.025, 0.075)
    #     mark_inset(ax, axins, loc1=1, loc2=3)
    # plt.xticks(visible=False)
    # plt.yticks(visible=False)
    # plt.tight_layout()
    # plt.savefig(second_output_file)

    third_output_file = output_file.split('.pdf')[0] + '_AA.pdf'
    fig, ax = plt.subplots(figsize=(8, 8))
    fill(ax, net, deep)
    plt.xlim(-0.53, 0.53)
    plt.ylim(-0.53, 0.53)
    if True:
        axins = zoomed_inset_axes(ax, 8, loc=4)
        for patch in ax.patches:
            patch_cpy = copy.copy(patch)
            # cut the umbilical cord the hard way
            patch_cpy.axes = None
            patch_cpy.figure = None
            patch_cpy.set_transform(axins.transData)
            axins.add_patch(patch_cpy)

        for collection in ax.get_children():
            if isinstance(collection, matplotlib.collections.PathCollection):  # This ensures it's a scatter plot
                offsets = collection.get_offsets()
                colors = collection.get_facecolors()
                sizes = collection.get_sizes() * 8
                axins.scatter(offsets[:, 0], offsets[:, 1],
                              color=colors, s=sizes)

        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_xlim(0.06, 0.11)
        axins.set_ylim(0.025, 0.075)
        mark_inset(ax, axins, loc1=1, loc2=3)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(third_output_file)

    # sixth_output_file = output_file.split('.pdf')[0] + '_DE.pdf'
    # fig, ax = plt.subplots(figsize=(8, 8))
    # if deep:
    #     contour, xticks, yticks = dilation_erosion(ax, net, resolution=2**12+1, delta=0.01)
    # else:
    #     contour, xticks, yticks = dilation_erosion(ax, net, resolution=2**5+1, delta=0.01)
    # plt.xlim(-0.53, 0.53)
    # plt.ylim(-0.53, 0.53)
    # if True:
    #     axins = zoomed_inset_axes(ax, 8, loc=4)
    #     for patch in ax.patches:
    #         patch_cpy = copy.copy(patch)
    #         # cut the umbilical cord the hard way
    #         patch_cpy.axes = None
    #         patch_cpy.figure = None
    #         patch_cpy.set_transform(axins.transData)
    #         axins.add_patch(patch_cpy)
    #
    #     for collection in contour.collections:
    #         for path in collection.get_paths():
    #             patch = PathPatch(path,
    #                               facecolor='none',
    #                               edgecolor=collection.get_edgecolor()[0],  # First color tuple
    #                               lw=collection.get_linewidths()[0])
    #             patch.set_transform(axins.transData)
    #             axins.add_patch(patch)
    #
    #     for collection in ax.get_children():
    #         if isinstance(collection, matplotlib.collections.PathCollection):  # scatter
    #             offsets = collection.get_offsets()
    #             colors = collection.get_facecolors()
    #             sizes = collection.get_sizes() * 8
    #
    #             if sizes.size == 0:
    #                 sizes = np.full(len(offsets), 36.0)  # Default size used by matplotlib
    #             elif sizes.size == 1:
    #                 sizes = np.full(len(offsets), sizes[0])
    #             elif sizes.size != len(offsets):
    #                 raise ValueError(f"Size mismatch: got {len(sizes)} sizes for {len(offsets)} points.")
    #
    #             axins.scatter(offsets[:, 0], offsets[:, 1],
    #                           color=colors, s=sizes)
    #
    #     axins.set_xticks(xticks)
    #     axins.set_yticks(yticks)
    #     axins.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
    #     axins.set_xticklabels([])
    #     axins.set_yticklabels([])
    #     axins.set_xlim(0.06, 0.11)
    #     axins.set_ylim(0.025, 0.075)
    #
    #     mark_inset(ax, axins, loc1=1, loc2=3)
    #
    # plt.xticks(visible=False)
    # plt.yticks(visible=False)
    # plt.tight_layout()
    # plt.savefig(sixth_output_file)
    return


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str,
                        help="The path to the .pth model from the root directory.")
    parser.add_argument("output_file", type=str,
                        help="The path to save the images rendered images to.")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Must specify if the model is one of the following: [mlp, siren].")
    parser.add_argument("--dim_samples", type=int, default=500,
                        help="The number of samples to draw from the model along each dimension.")
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
    parser.add_argument("--x_L", type=float, nargs='+', default=[-1., -1.],
                        help="Bottom left point of the input bounding box.")
    parser.add_argument("--x_U", type=float, nargs='+', default=[1., 1.],
                        help="Upper right point of the input bounding box.")
    parser.add_argument("--crown_mode", type=str, default='CROWN',
                        help="Bounding method to use on the neural SDF.")
    parser.add_argument("--deep", default=False, action='store_true')
    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict

if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)