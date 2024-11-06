# import igl # work around some env/packaging problems by loading this first

# import sys, os, time, math
# os.environ['OptiX_INSTALL_DIR'] = '/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'

import time
import argparse
import warnings

import torch
import os

from shapely.creation import points

# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
from skimage import measure
import polyscope as ps
import matplotlib as plt
import imageio
import open3d as o3d

# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CROWN_MODES = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("load_from_1", type=str)
    parser.add_argument("load_from_2", type=str)
    parser.add_argument("--mode", type=str, default='crown')
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--image_write_path", type=str, default="render_out.png")
    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=args.mode, **{})

    def sample_cube(lower_corner, upper_corner, N):
        # Create linearly spaced values along each dimension
        x = torch.linspace(lower_corner[0], upper_corner[0], N)
        y = torch.linspace(lower_corner[1], upper_corner[1], N)
        z = torch.linspace(lower_corner[2], upper_corner[2], N)
        # Create a 3D grid of points
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

        # Flatten the grid to get a list of points
        points = torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        return points

    data_bound = opts['data_bound']
    samples = sample_cube((-data_bound, -data_bound, -data_bound), (data_bound, data_bound, data_bound), 500)
    mask = torch.empty((samples.shape[0],), dtype=torch.bool)

    batch_size_per_iteration = 2 ** 17
    for start_idx in range(0, samples.shape[0], batch_size_per_iteration):
        end_idx = min(start_idx + batch_size_per_iteration, samples.shape[0])
        mask[start_idx:end_idx] = implicit_func(params, samples[start_idx:end_idx]).squeeze(1).abs() <= 0.0005
    valid_samples = samples[mask, :].cpu().numpy()

    ps.init()
    ps.register_point_cloud("surface samples", valid_samples, radius=0.001)

    mesh_npz_1 = np.load(args.load_from_1)
    shell_vertices_1 = mesh_npz_1['vertices'].astype(np.float32)
    shell_faces_1 = mesh_npz_1['faces'].astype(np.int32)
    ps.register_surface_mesh('mesh_1', shell_vertices_1, shell_faces_1)

    mesh_npz_2 = np.load(args.load_from_2)
    shell_vertices_2 = mesh_npz_2['vertices'].astype(np.float32)
    shell_faces_2 = mesh_npz_2['faces'].astype(np.int32)
    ps.register_surface_mesh('mesh_2', shell_vertices_2, shell_faces_2)
    ps.show()



if __name__ == '__main__':
    main()
