# import igl # work around some env/packaging problems by loading this first

# import sys, os, time, math
# os.environ['OptiX_INSTALL_DIR'] = '/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'

import time
import argparse
import warnings

import numpy as np
import torch
import os
import scipy
# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
import matplotlib as plt
import imageio
# import jax.numpy as jnp
import trimesh
import cv2
from PIL import Image

os.environ['OptiX_INSTALL_DIR'] = '/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'
# os.environ['OptiX_INSTALL_DIR'] = '/media/gaorz/b5df3483-c11a-42f1-b414-023f33bc5312/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'

from triro.ray.ray_optix import RayMeshIntersector  # FIXME: Should be uncommented when rendering meshes


# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def generate_camera_positions_on_grid(grid_size, radius, fov_deg):
    cameras = []
    theta_vals = torch.linspace(0, torch.pi, grid_size[0])
    phi_vals = torch.linspace(0, 2 * torch.pi, grid_size[1])

    for theta in theta_vals:
        for phi in phi_vals:
            # Calculate camera root position
            x = radius * torch.sin(theta) * torch.cos(phi)
            y = radius * torch.sin(theta) * torch.sin(phi)
            z = radius * torch.cos(theta)
            root = torch.tensor([x, y, z], dtype=torch.float32)

            # Look direction: pointing toward the origin
            look = -root / torch.norm(root)

            # Arbitrary 'up' vector to avoid singularity at poles
            arbitrary_up = torch.tensor([0.0, 0.0, 1.0]) if torch.abs(look[2]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])

            # Left direction: perpendicular to look and arbitrary up
            left = torch.cross(arbitrary_up, look)
            left /= torch.norm(left)

            # Upward direction: perpendicular to look and left
            up = torch.cross(look, left)
            up /= torch.norm(up)

            cameras.append({
                'root': root,
                'look': look,
                'up': up,
                'left': left,
                'fov_deg': fov_deg
            })

    return cameras


def save_render_current_view(args, implicit_func, params, intersector, opts, matcaps):
    # root = torch.tensor([-2.5, 0., 0.]) #+ torch.ones(3)
    # look = torch.tensor([1., 0., 0.])
    # up = torch.tensor([0., 1., 0.])
    # left = torch.tensor([0., 0., 1.])
    #
    root = torch.tensor([0., -1.5, 0.])
    left = torch.tensor([1., 0., 0.])
    look = torch.tensor([0.4, 1., 0.5])
    look = torch.tensor([-0.4, 1., 0.5])
    up = torch.tensor([0., 0., 1.])

    root = torch.tensor([0.25, -1.3, 0.45])
    left = torch.tensor([1., 0., 0.])
    look = torch.tensor([0., 1., 0.])
    up = torch.tensor([0., 0., 1.])

    # root = torch.tensor([-1.5, 2., 1.5])
    # left = torch.tensor([1., 0., 0.])
    # look = torch.tensor([0.4, -1., -0.5])
    # up = torch.tensor([0., 0., 1.])
    #
    # root = torch.tensor([-2.3, 2.3, 0.])
    # left = torch.tensor([1., 0., 0.])
    # look = torch.tensor([1., -1., 0.])
    # up = torch.tensor([0., 0., 1.])

    # root = torch.tensor([2.5, 0., 2.5])
    # up = torch.tensor([0., 1., 0.])
    # look = torch.tensor([-1., 0., -1.])
    # left = torch.tensor([1., 0., 0.])

    # root = torch.tensor([0., 0., 3.5])
    # up = torch.tensor([0., 1., 0.])
    # look = torch.tensor([0., 0., -1.])
    # left = torch.tensor([1., 0., 0.])
    #
    # root = torch.tensor([2.7, 0.2, 2.7])
    # up = torch.tensor([0., 1., 0.])
    # look = torch.tensor([-1., 0., -1.])
    # left = torch.tensor([1., 0., 0.])

    fov_deg = 30.
    # res = args.res
    res = int(args.res // opts['res_scale'])
    # print("resolution:", res)
    option = args.option

    if option == 'exact':
        img, rendering_time, count = render.render_image_mesh(implicit_func, params, intersector, root, look,
                                                       up, left, res,
                                                       fov_deg, opts,
                                                       shading='matcap_color', matcaps=matcaps, approx=False)
        print("=======")
        img, rendering_time, count = render.render_image_mesh(implicit_func, params, intersector, root, look,
                                                       up, left, res,
                                                       fov_deg, opts,
                                                       shading='matcap_color', matcaps=matcaps, approx=False)
        print("=======")
        img, rendering_time, count = render.render_image_mesh(implicit_func, params, intersector, root, look,
                                                              up, left, res,
                                                              fov_deg, opts,
                                                              shading='matcap_color', matcaps=matcaps, approx=False)
    elif option == 'approx':
        img, rendering_time, count = render.render_image_mesh(implicit_func, params, intersector, root, look,
                                                       up, left, res,
                                                       fov_deg, opts,
                                                       shading='matcap_color', matcaps=matcaps, approx=True)
    else:
        raise NotImplementedError('option must be one of "exact", "approx"')

    img = torch.flip(img, [0]).cpu().detach().numpy()
    alpha_channel = (np.min(img, axis=-1) < 1.).astype(np.float32)
    img_alpha = np.concatenate((img, alpha_channel[:, :, None]), axis=-1)
    img_alpha = np.clip(img_alpha, a_min=0., a_max=1.)
    img_alpha = (img_alpha * 255.).astype(np.uint8)
    print(f"Saving image to {args.image_write_path}")
    # img_alpha = cv2.GaussianBlur(img_alpha, (3, 3), 0)
    img_alpha = gaussian_blur_preserve_alpha(img_alpha, ksize=(3, 3), sigma=0)
    img_alpha = cv2.resize(img_alpha, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    Image.fromarray(img_alpha, 'RGBA').convert('RGB').save(args.image_write_path)


def gaussian_blur_preserve_alpha(img: np.ndarray, ksize=(15, 15), sigma=5) -> np.ndarray:
    """
    Apply Gaussian blur to an image with an alpha channel, avoiding bleed from transparent areas.

    Args:
        img (np.ndarray): Input image with shape (H, W, 4) in uint8 format (RGBA).
        ksize (tuple): Kernel size for Gaussian blur.
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        np.ndarray: Blurred image with preserved alpha, same shape and type as input.
    """
    if img.shape[2] != 4:
        raise ValueError("Input image must have 4 channels (RGBA).")

    # Normalize and split
    rgb = img[..., :3].astype(np.float32) / 255.0
    alpha = img[..., 3].astype(np.float32) / 255.0
    alpha = np.clip(alpha, 1e-5, 1.0)  # Avoid divide-by-zero

    # Premultiply RGB by alpha
    rgb_premul = rgb * alpha[..., None]

    # Apply Gaussian blur
    blurred_rgb = cv2.GaussianBlur(rgb_premul, ksize, sigma)
    blurred_alpha = cv2.GaussianBlur(alpha, ksize, sigma)

    # Un-premultiply
    final_rgb = blurred_rgb / blurred_alpha[..., None]
    final_rgb = np.clip(final_rgb, 0, 1)
    final_alpha = np.clip(blurred_alpha, 0, 1)

    # Combine and convert back to uint8
    final_img = np.dstack((final_rgb, final_alpha[..., None])) * 255
    return final_img.astype(np.uint8)

def render_random_camera(args, implicit_func, params, intersector, opts, matcaps):
    res = args.res // opts['res_scale']
    option = args.option

    cameras = generate_camera_positions_on_grid((1, 1), 2.5, 45.)
    root, look, up, left, fov_deg = cameras[0]['root'], cameras[0]['look'], cameras[0]['up'], cameras[0]['left'], cameras[0]['fov_deg']
    if option == 'exact':
        img, rendering_time, count = render.render_image_mesh(implicit_func, params, intersector, root,
                                                       look,
                                                       up, left, res,
                                                       fov_deg, opts,
                                                       shading='matcap_color', matcaps=matcaps, approx=False)
    elif option == 'approx':
        img, rendering_time, count = render.render_image_mesh(implicit_func, params, intersector, root,
                                                       look,
                                                       up, left, res,
                                                       fov_deg, opts,
                                                       shading='matcap_color', matcaps=matcaps, approx=True)
    print("=====")
    imgs, rendering_times= [], []
    # cameras = generate_camera_positions(1000, (5., 6.), (30., 90.))
    cameras = generate_camera_positions_on_grid((5, 10), 2.5, 45.)
    for cam in cameras:
        root, look, up, left, fov_deg = cam['root'], cam['look'], cam['up'], cam['left'], cam['fov_deg']


        if option == 'exact':
            img, rendering_time, count = render.render_image_mesh(implicit_func, params, intersector, root,
                                                           look,
                                                           up, left, res,
                                                           fov_deg, opts,
                                                           shading='matcap_color', matcaps=matcaps, approx=False)
        elif option == 'approx':
            img, rendering_time, count = render.render_image_mesh(implicit_func, params, intersector, root,
                                                           look,
                                                           up, left, res,
                                                           fov_deg, opts,
                                                           shading='matcap_color', matcaps=matcaps, approx=True)

        else:
            raise NotImplementedError('option must be one of "exact", "single", "double"')
        imgs.append(img.detach().cpu().numpy())
        rendering_times.append(rendering_time)

    avg_rendering_time = sum(rendering_times) / len(rendering_times)
    print("Average rendering time:", avg_rendering_time)
    if args.output:
        np.savez_compressed(args.output, imgs)
    if args.log_output:
        np.savez_compressed(args.log_output, np.array(rendering_times))

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("load_from", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--log_output", type=str)
    parser.add_argument("--option", type=str, default='exact')
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--image_write_path", type=str, default="render_out.png")
    parser.add_argument("--grid_cam", action="store_true")
    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    opts['hit_eps'] = 1e-3

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode='crown', **{})

    if args.load_from.endswith('.npz'):
        mesh_npz = np.load(args.load_from)
        vertices = mesh_npz['vertices'].astype(np.float32)
        faces = mesh_npz['faces'].astype(np.int32)
        print(len(faces))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    elif args.load_from.endswith('.obj'):
        mesh = trimesh.load(args.load_from)
        vertices = mesh.vertices
        faces = mesh.faces
        print(len(vertices))
        print(len(faces))

    # mesh.show()
    vertices = torch.tensor(vertices)
    with torch.no_grad():
        verts_dist = implicit_func.torch_forward(vertices.float())
        print(verts_dist.mean())
        pos_mask = (verts_dist > 0.).squeeze()
        neg_mask = ~pos_mask
        print(pos_mask.sum(), neg_mask.sum())

    intersector = RayMeshIntersector(mesh)

    # load the matcaps
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))
    matcaps = torch.stack(matcaps)
    if args.grid_cam:
        render_random_camera(args, implicit_func, params, intersector, opts, matcaps)
    else:
        save_render_current_view(args, implicit_func, params, intersector, opts, matcaps)


if __name__ == '__main__':
    main()
