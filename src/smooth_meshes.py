# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math, datetime
import time
import argparse
import warnings

import numpy as np
import torch
import imageio
import polyscope.imgui as psim
from prettytable import PrettyTable
# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
from scipy.spatial import Delaunay
import polyscope as ps
from skimage import measure
from mesh_utils import *
import trimesh
import open3d as o3d
# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CROWN_MODES = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward']

USE_CUSTOM_LOSS_OPTION = True  # if false, uses old hard-coded loss func, if true, uses custom loss func API

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
set_t = {
    "dtype": torch.float32,
    "device": device,
}
print(device)

cache_dir = "cache_bounds/compute_bounds_cache.npz"

to_numpy = lambda x : x.detach().cpu().numpy()  # converts tensor to numpy array

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("load_from", type=str)
    parser.add_argument("save_to", type=str)
    parser.add_argument("--mode", type=str, default='crown')
    parser.add_argument("--batch_size", type=int, default=256)

    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    mode = args.mode
    batch_size = args.batch_size
    modes = ['sdf', 'interval', 'affine_fixed', 'affine_truncate', 'affine_append', 'affine_all', 'slope_interval',
             'crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward', 'affine+backward', 'affine_quad']
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 8
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    affine_opts['affine+backward'] = 1.
    affine_opts['affine_quad'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = 'absolute'
    surf_color = (0.157, 0.613, 1.000)

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    # load the matcaps
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))
    if mode == 'affine_truncate':
        # truncate options
        implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode == 'affine_append':
        # truncate options
        implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode == 'sdf':

        changed, affine_opts['sdf_lipschitz'] = psim.InputFloat("SDF Lipschitz", affine_opts['sdf_lipschitz'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode in CROWN_MODES:

        implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode == 'affine+backward':
        implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)

    elif mode == 'affine_quad':
        implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)


    mesh_1_npz = np.load(args.load_from)
    vertices_1 = mesh_1_npz['vertices'].astype(np.float32)
    faces_1 = mesh_1_npz['faces'].astype(np.float32)

    mesh_1 = trimesh.Trimesh(vertices=vertices_1, faces=faces_1)

    vertices_1 = torch.from_numpy(vertices_1).cuda()
    sdfs_1 = implicit_func.torch_forward(vertices_1)
    sdf_1_pos = sdfs_1 > 0

    mesh_2 = mesh_1.copy()
    trimesh.smoothing.filter_taubin(mesh_2, lamb=0.5, nu=0.51, iterations=10)

    np.savez(args.save_to, **{'vertices': mesh_2.vertices.astype(np.float32), 'faces': mesh_2.faces.astype(np.float32)})
    vertices_2 = mesh_2.vertices.astype(np.float32)

    vertices_2 = torch.from_numpy(vertices_2).cuda()
    sdfs_2 = implicit_func.torch_forward(vertices_2)
    sdf_2_pos = sdfs_2 > 0

    diff = sdfs_2 - sdfs_1
    diff = diff[sdf_1_pos]
    print(diff.abs().mean())
    print(diff.abs().std())
    print(diff.abs().max())
    print(diff.abs().min())
    print((sdf_1_pos != sdf_2_pos).sum())
    print(sdf_1_pos.sum())
    print(sdf_2_pos.sum())

    # mesh_1_npz = np.load(args.load_from)
    # vertices_1 = mesh_1_npz['vertices'].astype(np.float32)
    # faces_1 = mesh_1_npz['faces'].astype(np.float32)
    #
    # mesh_1 = o3d.geometry.TriangleMesh()
    # mesh_1.vertices = o3d.utility.Vector3dVector(vertices_1)
    # mesh_1.triangles = o3d.utility.Vector3iVector(faces_1)
    #
    # vertices_1 = torch.from_numpy(vertices_1).cuda()
    # sdfs_1 = implicit_func.torch_forward(vertices_1)
    # sdf_1_pos = sdfs_1 > 0
    #
    # mesh_2 = mesh_1.deform_as_rigid_as_possible(
    #     energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed,
    #     max_iter=50
    # )
    #
    # # np.savez(args.save_to, **{'vertices': mesh_2.vertices.astype(np.float32), 'faces': mesh_2.faces.astype(np.float32)})
    # vertices_2 = mesh_2.vertices.astype(np.float32)
    #
    # vertices_2 = torch.from_numpy(vertices_2).cuda()
    # sdfs_2 = implicit_func.torch_forward(vertices_2)
    # sdf_2_pos = sdfs_2 > 0
    #
    # diff = sdfs_2 - sdfs_1
    # print(diff.abs().mean())
    # print(diff.abs().std())
    # print(diff.abs().max())
    # print(diff.abs().min())
    # print((sdf_1_pos != sdf_2_pos).sum())
    # print(sdf_1_pos.sum())
    # print(sdf_2_pos.sum())

if __name__ == '__main__':
    main()
