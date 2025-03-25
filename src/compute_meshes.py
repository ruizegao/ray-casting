# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
import time
import argparse
import warnings
import numpy as np
import torch
import imageio
import polyscope.imgui as psim
# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
from scipy.spatial import Delaunay
import polyscope as ps
from skimage import measure
from mesh_utils import *
import trimesh
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import pycork

# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CROWN_MODES = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("load_from", type=str)
    parser.add_argument("save_to", type=str)
    parser.add_argument("--mode", type=str, default='crown')
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--smooth", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    mode = args.mode

    def register_plane_and_cube_with_polyscope(
            As: np.ndarray,
            bs: np.ndarray,
            lower: np.ndarray,
            upper: np.ndarray,
            pos_lower: np.ndarray,
            pos_upper: np.ndarray,
            neg_lower: np.ndarray,
            neg_upper: np.ndarray,
    ):
        start_time = time.time()

        count = 0
        num_success, num_errors = 0, 0

        trimesh_meshes = []
        for n_l, n_u in zip(neg_lower, neg_upper):
            try:
                mesh = trimesh.creation.box(bounds=np.vstack((n_l, n_u)))
                v = np.array(mesh.vertices)
                f = np.array(mesh.faces)
                if len(v) > 0 and len(f) > 0:
                    if mesh.is_volume:
                        trimesh_meshes.append(mesh)
                    count += len(v)
                num_success += 1
            except Exception as e:
                num_errors += 1
                print(f"Encountered error (count {num_errors}): \n{e}")

        for A, b, l, u in zip(As, bs, lower, upper):
            try:
                cube = trimesh.creation.box(bounds=np.stack((l, u)))
                o = np.array([0., 0., - b / A[2]])
                mesh = cube.slice_plane(o, -A, cap=True)
                v = np.array(mesh.vertices)
                f = np.array(mesh.faces)
                if len(v) > 0 and len(f) > 0:
                    if mesh.is_volume:
                        trimesh_meshes.append(mesh)
                    count += len(v)
                num_success += 1
            except Exception as e:
                num_errors += 1
                print(f"Encountered error (count {num_errors}): \n{e}")

        end_time = time.time()
        print(f"Num success: {num_success}, Num errors: {num_errors}")
        print("total time cost: ", end_time - start_time)

        trimesh_mesh = trimesh.boolean.union(trimesh_meshes)
        # unsmoothed_mesh = trimesh_mesh.copy()
        # trimesh.smoothing.filter_taubin(trimesh_mesh, lamb=0.5, nu=-0.55, iterations=10)
        # trimesh_mesh.show()
        return trimesh_mesh

        # displacements = np.linalg.norm(trimesh_mesh.vertices - unsmoothed_mesh.vertices, axis=1)
        # print(len(unsmoothed_mesh.vertices))
        print(len(trimesh_mesh.vertices))
        print(len(trimesh_mesh.faces))
        # Get the maximum shrinkage
        # print(f"Max Shrinkage: {np.max(displacements)}")
        # print(f"Mean Shrinkage: {np.mean(displacements)}")
        # print(f"Min Shrinkage: {np.min(displacements)}")

        # mesh = {'vertices': np.array(trimesh_mesh.vertices), 'faces':np.array(trimesh_mesh.faces)}
        # np.savez(args.save_to, **mesh)

    ret_val = [val for val in np.load(args.load_from).values()]
    [node_lower, node_upper, mAs, mbs, lAs, lbs, uAs, ubs, pos_lower, pos_upper, neg_lower, nge_upper, plane_constraints_lower, plane_constraints_upper] = ret_val

    num_constraints = plane_constraints_lower.shape[1]
    # num_constraints = 0
    print(f"Found {num_constraints} constraint plane(s) to add to the mesh")
    if num_constraints == 0:
        outer_shell = register_plane_and_cube_with_polyscope(lAs, lbs, node_lower, node_upper, pos_lower, pos_upper, neg_lower, nge_upper)
        outer_shell.export(args.save_to[:-4]+'_outer.obj')
        mid_shell = register_plane_and_cube_with_polyscope(mAs, mbs, node_lower, node_upper, pos_lower, pos_upper, neg_lower, nge_upper)
        mid_shell.export(args.save_to[:-4]+'_mid.obj')
        inner_shell = register_plane_and_cube_with_polyscope(uAs, ubs, node_lower, node_upper, pos_lower, pos_upper, neg_lower, nge_upper)
        both_shell_verts = np.concatenate((outer_shell.vertices, inner_shell.vertices), axis=0)
        both_shell_faces = np.concatenate((outer_shell.faces, inner_shell.faces + len(outer_shell.vertices)), axis=0)
        both_shells = trimesh.Trimesh(both_shell_verts, both_shell_faces)
        mid_shell.export(args.save_to[:-4]+'_both.obj')
    else:
        raise ValueError("More than 2 planes is not supported as of yet")
