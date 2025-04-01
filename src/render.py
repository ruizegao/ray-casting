import os
import sys
import gc
from typing import Optional, Tuple

import functorch
import torch
import scipy
import numpy as np
from functools import partial
from functorch import vmap
from crown import CrownImplicitFunction
import imageio
from PIL import Image
import geometry
import queries
from utils import *
import affine
import trimesh
import matplotlib.pyplot as plt
import sys, os, time, math
import jax
import jax.numpy as jnp

os.environ['OptiX_INSTALL_DIR'] = '/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'

from triro.ray.ray_optix import RayMeshIntersector  # FIXME: Should be uncommented when rendering meshes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


HIT_EPS = 0.001
FD_OFFSET = torch.tensor((
                (+HIT_EPS, -HIT_EPS, -HIT_EPS),
                (-HIT_EPS, -HIT_EPS, +HIT_EPS),
                (-HIT_EPS, +HIT_EPS, -HIT_EPS),
                (+HIT_EPS, +HIT_EPS, +HIT_EPS),
            ))

# theta_x/y should be
def camera_ray(look_dir, up_dir, left_dir, fov_deg_x, fov_deg_y, theta_x, theta_y):
    ray_image_plane_pos = look_dir \
                          + left_dir * (theta_x * torch.tan(
        torch.deg2rad(torch.tensor(fov_deg_x, device=look_dir.device)) / 2)) \
                          + up_dir * (theta_y * torch.tan(
        torch.deg2rad(torch.tensor(fov_deg_y, device=look_dir.device)) / 2))

    ray_dir = geometry.normalize(ray_image_plane_pos)

    return ray_dir

def generate_camera_rays(eye_pos, look_dir, up_dir, res=1024, fov_deg=30.):
    D = res  # image dimension
    R = res * res  # number of rays

    ## Generate rays according to a pinhole camera

    # Image coords on [-1,1] for each output pixel
    cam_ax_x = torch.linspace(-1., 1., res)
    cam_ax_y = torch.linspace(-1., 1., res)
    cam_y, cam_x = torch.meshgrid(cam_ax_x, cam_ax_y, indexing='ij')
    cam_x = cam_x.flatten()
    cam_y = cam_y.flatten()

    # Orthornormal camera frame
    up_dir = up_dir - torch.dot(look_dir, up_dir) * look_dir
    up_dir /= torch.norm(up_dir)
    left_dir = torch.cross(look_dir, up_dir)

    ray_dirs = vmap(partial(camera_ray, look_dir, up_dir, left_dir, fov_deg, fov_deg))(cam_x, cam_y)
    ray_roots = torch.tile(eye_pos, (ray_dirs.shape[0], 1))
    return ray_roots, ray_dirs


def outward_normal(funcs_tuple, params_tuple, hit_pos, hit_id, eps, method='finite_differences'):
    grad_out = torch.zeros(3)
    i_func = 1
    for func, params in zip(funcs_tuple, params_tuple):
        if isinstance(func, CrownImplicitFunction):
            # f = partial(func.call_implicit_func, params)
            f = func.torch_forward
        else:
            f = partial(func, params)

        if method == 'autodiff':
            grad_f = functorch.jacfwd(f)
            grad = grad_f(hit_pos)

        elif method == 'finite_differences':
            # 'tetrahedron' central differences approximation
            # see e.g. https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm

            x_pts = hit_pos[None, :] + FD_OFFSET
            samples = vmap(f)(x_pts).squeeze(1).detach()
            grad = torch.sum(FD_OFFSET * samples[:, None], dim=0)

        else:
            raise ValueError("unrecognized method")

        grad = geometry.normalize(grad)
        grad_out = torch.where(hit_id == i_func, grad, grad_out)
        i_func += 1

    return grad_out

def outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, eps, method='finite_differences'):
    this_normal_one = lambda p, id: outward_normal(funcs_tuple, params_tuple, p, id, eps, method=method)
    if method == 'autodiff':
        total_samples = hit_pos.shape[0]
        out_normal = torch.empty_like(hit_pos)
        batch_size_per_iteration = 256
        for start_idx in range(0, total_samples, batch_size_per_iteration):
            end_idx = min(start_idx + batch_size_per_iteration, total_samples)
            out_normal[start_idx:end_idx] \
                = vmap(this_normal_one)(hit_pos[start_idx:end_idx], hit_ids[start_idx:end_idx])

        return out_normal
    elif method == 'finite_differences':
        total_samples = hit_pos.shape[0]
        out_normal = torch.empty_like(hit_pos)
        batch_size_per_iteration = 2**17
        # batch_size_per_iteration = 2**12
        for start_idx in range(0, total_samples, batch_size_per_iteration):
            end_idx = min(start_idx + batch_size_per_iteration, total_samples)
            out_normal[start_idx:end_idx] \
                = vmap(this_normal_one)(hit_pos[start_idx:end_idx], hit_ids[start_idx:end_idx])

        return out_normal
    return vmap(this_normal_one)(hit_pos, hit_ids)

def render_image(funcs_tuple, params_tuple, eye_pos, look_dir, up_dir, left_dir, res, fov_deg, frustum, branching_method, opts,
                 shading="normal", shading_color_tuple=((0.157, 0.613, 1.000)), matcaps=None, tonemap=False,
                 shading_color_func=None, tree_based=False, load_from=None, save_to=None):
    # make sure inputs are tuples not lists (can't has lists)
    if isinstance(funcs_tuple, list): funcs_tuple = tuple(funcs_tuple)
    if isinstance(params_tuple, list): params_tuple = tuple(params_tuple)
    if isinstance(shading_color_tuple, list): shading_color_tuple = tuple(shading_color_tuple)

    # wrap in tuples if single was passed
    if not isinstance(funcs_tuple, tuple):
        funcs_tuple = (funcs_tuple,)
    if not isinstance(params_tuple, tuple):
        params_tuple = (params_tuple,)
    if not isinstance(shading_color_tuple[0], tuple):
        shading_color_tuple = (shading_color_tuple,)

    L = len(funcs_tuple)
    if (len(params_tuple) != L) or (len(shading_color_tuple) != L):
        raise ValueError("render_image tuple arguments should all be same length")

    ray_roots, ray_dirs = generate_camera_rays(eye_pos, look_dir, up_dir, res=res, fov_deg=fov_deg)
    if frustum:
        # == Frustum raycasting

        cam_params = eye_pos, look_dir, up_dir, left_dir, fov_deg, fov_deg, res, res

        with Timer("frustum raycast"):
            t_raycast, hit_ids, counts, n_eval = queries.cast_rays_frustum(funcs_tuple, params_tuple, cam_params, opts)
            # t_raycast.block_until_ready()
            torch.cuda.synchronize()

        # TODO transposes here due to image layout conventions. can we get rid of them?
        t_raycast = t_raycast.transpose().flatten()
        hit_ids = hit_ids.transpose().flatten()
        counts = counts.transpose().flatten()

    else:
        # == Standard raycasting
        with Timer("raycast"):
            t_raycast, hit_ids, counts, n_eval = queries.cast_rays(funcs_tuple, params_tuple, ray_roots, ray_dirs, opts)
            # t_raycast.block_until_ready()
            # print("t_raycast", t_raycast)
            torch.cuda.synchronize()

    hit_pos = ray_roots + t_raycast[:, None] * ray_dirs

    torch.cuda.empty_cache()

    hit_normals = outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, opts['hit_eps'])
    hit_color = shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple,
                            shading_color_func=shading_color_func)
    # print(hit_pos, hit_normals, hit_color)
    img = torch.where(hit_ids[:, None].bool(), hit_color, torch.ones((res * res, 3)))

    if tonemap:
        # We intentionally tonemap before compositing in the shadow. Otherwise the white level clips the shadow and gives it a hard edge.
        img = tonemap_image(img)

    img = img.reshape(res, res, 3)
    depth = t_raycast.reshape(res, res)
    counts = counts.reshape(res, res)
    hit_ids = hit_ids.reshape(res, res)

    return img, depth, counts, hit_ids, n_eval, -1


def render_image_naive(funcs_tuple, params_tuple, eye_pos, look_dir, up_dir, left_dir, res, fov_deg, frustum, opts,
                       shading="normal", shading_color_tuple=((0.157, 0.613, 1.000)), matcaps=None, tonemap=False,
                       shading_color_func=None, tree_based=False, shell_based=False, batch_size=None, enable_clipping=False, load_from=None, save_to=None):
    # make sure inputs are tuples not lists (can't has lists)
    if isinstance(funcs_tuple, list): funcs_tuple = tuple(funcs_tuple)
    if isinstance(params_tuple, list): params_tuple = tuple(params_tuple)
    if isinstance(shading_color_tuple, list): shading_color_tuple = tuple(shading_color_tuple)

    # wrap in tuples if single was passed
    if not isinstance(funcs_tuple, tuple):
        funcs_tuple = (funcs_tuple,)
    if not isinstance(params_tuple, tuple):
        params_tuple = (params_tuple,)
    if not isinstance(shading_color_tuple[0], tuple):
        shading_color_tuple = (shading_color_tuple,)

    L = len(funcs_tuple)
    if (len(params_tuple) != L) or (len(shading_color_tuple) != L):
        raise ValueError("render_image tuple arguments should all be same length")

    ray_roots, ray_dirs = generate_camera_rays(eye_pos, look_dir, up_dir, res=res, fov_deg=fov_deg)
    if frustum:
        # == Frustum raycasting

        cam_params = eye_pos, look_dir, up_dir, left_dir, fov_deg, fov_deg, res, res

        t_raycast, hit_ids, counts, n_eval = queries.cast_rays_frustum(funcs_tuple, params_tuple, cam_params, opts)
        # t_raycast.block_until_ready()
        torch.cuda.synchronize()

        # TODO transposes here due to image layout conventions. can we get rid of them?
        t_raycast = t_raycast.transpose().flatten()
        hit_ids = hit_ids.transpose().flatten()
        counts = counts.transpose().flatten()

    elif shell_based:
        t_raycast, hit_ids, counts, n_eval = queries.cast_rays_shell_based(funcs_tuple, params_tuple, ray_roots,
                                                                          ray_dirs, batch_size=batch_size,
                                                                          load_from=load_from)
        torch.cuda.synchronize()
    else:
        # == Standard raycasting
        t_raycast, hit_ids, counts, n_eval = queries.cast_rays(funcs_tuple, params_tuple, ray_roots, ray_dirs, opts)
        torch.cuda.synchronize()

    hit_pos = ray_roots + t_raycast[:, None] * ray_dirs

    torch.cuda.empty_cache()

    hit_normals = outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, opts['hit_eps'])
    hit_color = shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple,
                            shading_color_func=shading_color_func)

    img = torch.where(hit_ids[:, None].bool(), hit_color, torch.ones((res * res, 3)))

    if tonemap:
        # We intentionally tonemap before compositing in the shadow. Otherwise the white level clips the shadow and gives it a hard edge.
        img = tonemap_image(img)

    img = img.reshape(res, res, 3)
    depth = t_raycast.reshape(res, res)
    counts = counts.reshape(res, res)
    hit_ids = hit_ids.reshape(res, res)

    return img, depth, counts, hit_ids, n_eval, -1


def render_image_mesh(funcs_tuple, params_tuple, faces, vertices, intersector, eye_pos, look_dir, up_dir, left_dir, res, fov_deg, opts,
                      shading="normal", shading_color_tuple=torch.tensor(((0.157, 0.613, 1.000),)), approx=False, delta=0.001, matcaps=None, tonemap=False,
                      shading_color_func=None):
    if isinstance(funcs_tuple, list): funcs_tuple = tuple(funcs_tuple)
    if isinstance(params_tuple, list): params_tuple = tuple(params_tuple)

    # wrap in tuples if single was passed
    if not isinstance(funcs_tuple, tuple):
        funcs_tuple = (funcs_tuple,)
    if not isinstance(params_tuple, tuple):
        params_tuple = (params_tuple,)

    ray_roots, ray_dirs = generate_camera_rays(eye_pos, look_dir, up_dir, res=res, fov_deg=fov_deg)

    # if approx:
    #     vertex_normals = outward_normals(funcs_tuple, params_tuple, vertices.float(), torch.ones_like(vertices), opts['hit_eps'], method='finite_differences')

    # _, _, _, _, _, _, _ = queries.cast_rays_shell_based(funcs_tuple, params_tuple, torch.empty_like(ray_roots), torch.empty_like(ray_dirs), intersector, approx, delta)

    time_render_start = time.time()
    hit_pos, hit_ids, hit, tri_idx, uv, _, _ = queries.cast_rays_shell_based(funcs_tuple, params_tuple, ray_roots, ray_dirs, intersector, approx, delta)
    # plt.imshow(hit_ids.detach().cpu().numpy().reshape(res, res))
    # plt.show()
    # if approx:
    #     hit_normals = torch.zeros_like(ray_dirs)
    #     tri_v = faces[tri_idx]
    #     tri_norm = vertex_normals[tri_v]
    #     hit_norm = uv[:, :1] * tri_norm[:, 0] + uv[:, 1:] * tri_norm[:, 1] + (1 - uv[:, :1] - uv[:, 1:]) * tri_norm[:, 2]
    #     hit_normals[hit] = hit_norm
    # else:
    #     hit_normals = outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, opts['hit_eps'], method='finite_differences')
    hit_normals = outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, opts['hit_eps'], method='finite_differences')

    hit_color = shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple,
                            shading_color_func=shading_color_func)
    img = torch.where(hit_ids[:, None].bool(), hit_color, torch.ones((res * res, 3)))

    if tonemap:
        # We intentionally tonemap before compositing in the shadow. Otherwise the white level clips the shadow and gives it a hard edge.
        img = tonemap_image(img)

    img = img.reshape(res, res, 3)
    time_render_end = time.time()
    print("Time rendering:", time_render_end - time_render_start)

    return img, time_render_end - time_render_start

# @torch.jit.script
def linspace_with_directional_delta(start_tensor, end_tensor, delta, directions):
    """
    Generates a concatenated linspace between start and end points using a directional delta.

    Args:
        start_tensor (torch.Tensor): Tensor of shape (N, 3) representing the start points.
        end_tensor (torch.Tensor): Tensor of shape (N, 3) representing the end points.
        delta (float): Scalar step size.
        directions (torch.Tensor): Tensor of shape (N, 3) representing direction multipliers.

    Returns:
        torch.Tensor: Concatenated tensor of all linspace values, shape (total_points, 3).
        torch.Tensor: Number of points per linspace (N,).
    """
    # Compute step sizes per dimension
    step_sizes = directions * delta  # Shape: (N, 3)

    # Compute number of points per linspace
    # num_points = torch.mean((end_tensor - start_tensor) / step_sizes, dim=-1).clip(min=0.).ceil().to(torch.int64) + 1  # Shape: (N,)
    num_points = ((end_tensor - start_tensor) / step_sizes)[:, 0].clip(min=0.).ceil().to(torch.int64) + 1  # Shape: (N,)

    # Compute start indices of each linspace in the final output tensor
    start_indices = torch.cat((torch.tensor([0], device=start_tensor.device), num_points.cumsum(0)[:-1]))

    # Compute total number of points needed
    total_points = num_points.sum()

    # Generate a 1D index tensor for all points
    index = torch.arange(total_points, device=start_tensor.device)

    # Expand indices to match their respective linspace
    linspace_ids = torch.searchsorted(start_indices, index, right=True) - 1  # Shape: (total_points,)

    # Compute the values directly using the corresponding start points and step sizes
    points = start_tensor[linspace_ids] + (index - start_indices[linspace_ids]).unsqueeze(-1) * step_sizes[linspace_ids]

    # Ensure each linspace segment ends exactly at end_tensor
    mask = torch.cat((start_indices[1:] - 1, torch.tensor([total_points - 1], device=start_tensor.device)))
    points[mask] = end_tensor  # Overwrite last points of each segment

    return points, num_points

# @torch.jit.script
def first_one_in_segments(binary_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Finds the index of the first '1' in each segment of a binary tensor.

    Parameters:
        binary_tensor (torch.Tensor): A 1D binary tensor (values 0 or 1).
        lengths (torch.Tensor): A 1D tensor specifying the lengths of each segment.

    Returns:
        torch.Tensor: A 1D tensor with the index of the first '1' in each segment,
                      or -1 if no '1' is found in a segment.
    """
    # Compute start indices of each segment
    start_indices = torch.cat((torch.tensor([0], device=binary_tensor.device), lengths.cumsum(0)[:-1]))

    # Find positions of all ones
    ones_positions = torch.nonzero(binary_tensor, as_tuple=True)[0]

    if ones_positions.numel() == 0:  # Edge case: no "1" in the entire tensor
        return torch.full_like(lengths, -1)

    # Identify the segment each "1" belongs to
    segment_ids = torch.searchsorted(start_indices, ones_positions, right=True) - 1

    if segment_ids.numel() == 0:
        return torch.full_like(lengths, -1)

    # Ensure first_indices has the same dtype as ones_positions
    print("one pos dtype:", ones_positions.dtype)
    first_indices = torch.full_like(lengths, fill_value=torch.iinfo(ones_positions.dtype).max, dtype=ones_positions.dtype)

    # Use scatter_reduce_ to find the first occurrence of '1' in each segment
    first_indices.scatter_reduce_(0, segment_ids, ones_positions, reduce="amin")

    # Replace large values (where no "1" was found) with -1
    first_indices[first_indices == torch.iinfo(ones_positions.dtype).max] = -1

    # Convert to relative indices
    # valid_mask = first_indices != -1
    # first_indices[valid_mask] -= start_indices[valid_mask]

    return first_indices

def render_image_de(funcs_tuple, params_tuple, faces, vertices, intersector, eye_pos, look_dir, up_dir, left_dir, res, fov_deg, opts,
                      shading="normal", shading_color_tuple=torch.tensor(((0.157, 0.613, 1.000),)), matcaps=None, tonemap=False,
                      shading_color_func=None):
    if isinstance(funcs_tuple, list): funcs_tuple = tuple(funcs_tuple)
    if isinstance(params_tuple, list): params_tuple = tuple(params_tuple)

    # wrap in tuples if single was passed
    if not isinstance(funcs_tuple, tuple):
        funcs_tuple = (funcs_tuple,)
    if not isinstance(params_tuple, tuple):
        params_tuple = (params_tuple,)
    func = funcs_tuple[0]
    params = params_tuple[0]

    ray_roots, ray_dirs = generate_camera_rays(eye_pos, look_dir, up_dir, res=res, fov_deg=fov_deg)
    time_render_start = time.perf_counter()

    hit_first, front_first, tri_idx, location_first, uv = intersector.intersects_closest(
        ray_roots, ray_dirs, stream_compaction=False
    )
    ray_roots[hit_first] = location_first[hit_first] #+ opts['hit_eps']
    ray_roots_copy = ray_roots.detach().clone()
    # location_first[~hit_first] = ray_roots[~hit_first]
    hit_second, front_second, tri_idx, location_second, uv = intersector.intersects_closest(
        ray_roots_copy, ray_dirs, stream_compaction=False
    )
    location_second[~hit_second] = location_first[~hit_second]
    hit_both = hit_first & hit_second
    points_to_check, num_points = linspace_with_directional_delta(location_first[hit_both], location_second[hit_both], torch.tensor(opts['hit_eps']), ray_dirs[hit_both])
    print("number of points to check", len(points_to_check))
    assert len(points_to_check) == num_points.sum()
    with torch.no_grad():
        preds = torch.empty(points_to_check.shape[0])
        total_samples = points_to_check.shape[0]
        batch_size = 1024 * 1024 * 2
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            preds[start_idx:end_idx] = func.torch_forward(points_to_check[start_idx:end_idx]).flatten()
        # preds = func.torch_forward(points_to_check)
    sign_change_mask = (preds <= 0).to(torch.int64).flatten()
    sign_change_inds = first_one_in_segments(sign_change_mask, num_points)
    check_next_round = sign_change_inds < 0
    time_intersection = time.perf_counter()
    hit = hit_both.clone()
    hit[hit_both] = ~check_next_round
    hit_pos = ray_roots.clone()
    hit_pos[hit] = points_to_check[sign_change_inds[~check_next_round]]
    plt.imshow(hit.detach().cpu().numpy().reshape(1024, 1024))
    plt.show()
    hit_ids = torch.zeros(ray_roots.shape[0])
    hit_ids[hit] = 1.

    hit_normals = outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, opts['hit_eps'], method='finite_differences')

    hit_color = shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple,
                            shading_color_func=shading_color_func)
    img = torch.where(hit_ids[:, None].bool(), hit_color, torch.ones((res * res, 3)))

    if tonemap:
        # We intentionally tonemap before compositing in the shadow. Otherwise the white level clips the shadow and gives it a hard edge.
        img = tonemap_image(img)

    img = img.reshape(res, res, 3)
    time_render_end = time.perf_counter()
    print("Time intersection", time_intersection - time_render_start)
    print("Time rendering:", time_render_end - time_render_start)

    return img, time_render_end - time_render_start

def tonemap_image(img, gamma=2.2, white_level=.75, exposure=1.):
    img = img * exposure
    num = img * (1.0 + (img / (white_level * white_level)))
    den = (1.0 + img)
    img = num / den
    img = torch.pow(img, 1.0 / gamma)
    return img

@torch.jit.script
def shade_image(shading: str, ray_dirs: torch.Tensor, hit_pos: torch.Tensor, hit_normals: torch.Tensor,
                hit_ids: torch.Tensor, up_dir: torch.Tensor, matcaps: torch.Tensor,
                shading_color_tuple: torch.Tensor, shading_color_func=None) -> torch.Tensor:
    # compute matcap coordinates
    ray_up = (up_dir - (up_dir * ray_dirs).sum(dim=-1, keepdim=True) * ray_dirs)
    ray_up = ray_up / ray_up.norm(p=2, dim=-1, keepdim=True)
    ray_left = torch.cross(ray_dirs, ray_up, dim=-1)
    matcap_u = torch.einsum('ij,ij->i', -ray_left, hit_normals)
    matcap_v = torch.einsum('ij,ij->i', ray_up, hit_normals)

    matcap_u *= 0.98
    matcap_v *= 0.98

    matcap_x = (matcap_u + 1.) / 2. * matcaps[0].shape[0]
    matcap_y = (-matcap_v + 1.) / 2. * matcaps[0].shape[1]
    matcap_coords = torch.stack((matcap_x, matcap_y), dim=-1)

    x = matcap_coords[:, 0].long().clamp(0, matcaps[0].shape[0] - 1)
    y = matcap_coords[:, 1].long().clamp(0, matcaps[0].shape[1] - 1)

    mat_r = matcaps[0][x, y]
    mat_g = matcaps[1][x, y]
    mat_b = matcaps[2][x, y]
    mat_k = matcaps[3][x, y]

    shading_color = torch.ones_like(hit_pos)
    # if shading_color_func is None:
    i_func = 1
    for c in shading_color_tuple:
        mask = (hit_ids == i_func).unsqueeze(-1)
        # shading_color = torch.where(mask, torch.tensor(c, dtype=shading_color.dtype, device=shading_color.device), shading_color)
        shading_color = torch.where(mask, c, shading_color)
        i_func += 1
    # else:
    #     shading_color = shading_color_func(hit_pos)

    c_r, c_g, c_b = shading_color[:, 0], shading_color[:, 1], shading_color[:, 2]
    c_k = 1. - (c_r + c_b + c_g)

    c_r = c_r[:, None]
    c_g = c_g[:, None]
    c_b = c_b[:, None]
    c_k = c_k[:, None]

    hit_color = c_r * mat_r + c_b * mat_b + c_g * mat_g + c_k * mat_k

    return hit_color


def look_at(eye_pos, target=None, up_dir='y'):
    if target == None:
        target = torch.tensor((0., 0., 0.,))
    if up_dir == 'y':
        up_dir = torch.tensor((0., 1., 0.,))
    elif up_dir == 'z':
        up_dir = torch.tensor((0., 0., 1.,))

    look_dir = geometry.normalize(target - eye_pos)
    up_dir = geometry.orthogonal_dir(up_dir, look_dir)
    left_dir = torch.cross(look_dir, up_dir)

    return look_dir, up_dir, left_dir


def load_matcap(fname_pattern):
    imgs = []
    for c in ['r', 'g', 'b', 'k']:
        im = imageio.imread(fname_pattern.format(c))
        im = torch.tensor(im) / 256.
        imgs.append(im)

    return tuple(imgs)