# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
import time
import argparse
import warnings
import numpy as np
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

    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    mode = args.mode


    def register_plane_and_cube_with_polyscope(
            As: torch.Tensor,
            bs: torch.Tensor,
            lower: torch.Tensor,
            upper: torch.Tensor):

        start_time = time.time()

        tri_faces = []
        tri_vertices = []

        count = 0
        num_success, num_errors = 0, 0
        for A, b, l, u in zip(As, bs, lower, upper):
            try:
                cube = trimesh.creation.box(bounds=np.stack((l, u)))
                o = np.array([0., 0., - b / A[2]])
                # print(o, -A)
                mesh = cube.slice_plane(o, -A, cap=True)
                v = np.array(mesh.vertices)
                f = np.array(mesh.faces)
                # print(v.shape, f.shape)
                if len(v) > 0 and len(f) > 0:
                    tri_faces.append(f+count)
                    tri_vertices.append(v)
                    count += len(v)
                num_success += 1
            except Exception as e:
                num_errors += 1
                print(f"Encountered error (count {num_errors}): \n{e}")

        tri_faces = np.concatenate(tri_faces, axis=0)
        tri_vertices = np.concatenate(tri_vertices, axis=0)
        end_time = time.time()
        print(f"Num success: {num_success}, Num errors: {num_errors}")
        print("total time cost: ", end_time - start_time)

        mesh = {}
        mesh['vertices'] = tri_vertices
        mesh['faces'] = tri_faces

        np.savez(args.save_to, **mesh)
        # trimesh_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
        # trimesh_mesh.show()

    def register_planes_and_cube_with_polyscope(
            A1s: torch.Tensor,
            b1s: torch.Tensor,
            A2s: torch.Tensor,
            b2s: torch.Tensor,
            lower: torch.Tensor,
            upper: torch.Tensor):

        start_time = time.time()

        tri_faces = []
        tri_vertices = []

        count = 0
        num_success, num_errors = 0, 0
        for A1, b1, A2, b2, l, u in zip(A1s, b1s, A2s, b2s, lower, upper):
            try:
                cube = trimesh.creation.box(bounds=np.stack((l, u)))
                o1 = np.array([0., 0., - b1 / A1[2]])
                # print(o, -A)
                mesh = cube.slice_plane(o1, -A1, cap=True)
                o2 = np.array([0., 0., - b2 / A2[2]])
                mesh = mesh.slice_plane(o2, -A2, cap=True)
                v = np.array(mesh.vertices)
                f = np.array(mesh.faces)
                # print(v.shape, f.shape)
                if len(v) > 0 and len(f) > 0:
                    tri_faces.append(f+count)
                    tri_vertices.append(v)
                    count += len(v)
                num_success += 1
            except Exception as e:
                num_errors += 1

        tri_faces = np.concatenate(tri_faces, axis=0)
        tri_vertices = np.concatenate(tri_vertices, axis=0)
        end_time = time.time()
        print(f"Num success: {num_success}, Num errors: {num_errors}")
        print("total time cost: ", end_time - start_time)

        mesh = {}
        mesh['vertices'] = tri_vertices
        mesh['faces'] = tri_faces

        np.savez(args.save_to, **mesh)
        # trimesh_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
        # trimesh_mesh.show()

    ret_val = [val for val in np.load(args.load_from).values()]
    [node_lower, node_upper, mAs, mbs, lAs, lbs, uAs, ubs, plane_constraints_lower, plane_constraints_upper] = ret_val

    num_constraints = plane_constraints_lower.shape[1]
    # num_constraints = 0
    print(f"Found {num_constraints} constraint plane(s) to add to the mesh")
    if num_constraints == 1:
        cAs = plane_constraints_lower[:, :, :3].squeeze(1)
        cbs = plane_constraints_lower[:, :, 3].squeeze(1)
        register_planes_and_cube_with_polyscope(lAs, lbs, cAs, cbs, node_lower, node_upper)
    elif num_constraints == 0:
        register_plane_and_cube_with_polyscope(lAs, lbs, node_lower, node_upper)
    else:
        raise ValueError("More than 2 planes is not supported as of yet")
