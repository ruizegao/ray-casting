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

# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CROWN_MODES = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(device)


def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("load_from", type=str)
    parser.add_argument("save_to", type=str, default='meshes/mesh_0.npz')
    parser.add_argument("--mode", type=str, default='crown')
    parser.add_argument("--res", type=int, default=1024)

    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    mode = args.mode


    def register_plane_and_cube_with_polyscope(As, bs, lower, upper):
        tri_faces = []
        tri_vertices = []

        count = 0
        base_mesh = trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
        start_time = time.time()
        for A, b, l, u in zip(As, bs, lower, upper):
            cube = trimesh.creation.box(bounds=np.stack((l, u)))
            o = np.array([0., 0., - b / A[2]])
            # print(o, -A)
            mesh = cube.slice_plane(o, -A, cap=True)
            # base_mesh = base_mesh + mesh
            v = np.array(mesh.vertices)
            f = np.array(mesh.faces)
            # print(v.shape, f.shape)
            if len(v) > 0 and len(f) > 0:
                tri_faces.append(f+count)
                tri_vertices.append(v)
                count += len(v)
        end_time = time.time()
        print(end_time - start_time)
        tri_faces = np.concatenate(tri_faces, axis=0)
        tri_vertices = np.concatenate(tri_vertices, axis=0)

        trimesh_mesh = trimesh.Trimesh(vertices=tri_vertices, faces=tri_faces)
        trimesh_mesh.update_faces(trimesh_mesh.unique_faces())
        trimesh_mesh.remove_unreferenced_vertices()

        mesh = {}
        mesh['vertices'] = np.array(trimesh_mesh.vertices)
        mesh['faces'] = np.array(trimesh_mesh.faces)

        np.savez(args.save_to, **mesh)
        # trimesh_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
        trimesh_mesh.show()

    def register_shell(lAs, lbs, uAs, ubs, lower, upper):
        tri_faces = []
        tri_vertices = []

        count = 0
        base_mesh = trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
        start_time = time.time()
        for lA, lb, uA, ub, l, u in zip(lAs, lbs, uAs, ubs, lower, upper):
            cube = trimesh.creation.box(bounds=np.stack((l, u)))
            o = np.array([0., 0., - lb / lA[2]])
            # print(o, -A)
            mesh = cube.slice_plane(o, -lA, cap=True)
            o = np.array([0., 0., - ub / uA[2]])
            mesh = mesh.slice_plane(o, uA, cap=True)
            # base_mesh = base_mesh + mesh
            v = np.array(mesh.vertices)
            f = np.array(mesh.faces)
            # print(v.shape, f.shape)
            if len(v) > 0 and len(f) > 0:
                tri_faces.append(f+count)
                tri_vertices.append(v)
                count += len(v)
        end_time = time.time()
        print(end_time - start_time)
        tri_faces = np.concatenate(tri_faces, axis=0)
        tri_vertices = np.concatenate(tri_vertices, axis=0)

        trimesh_mesh = trimesh.Trimesh(vertices=tri_vertices, faces=tri_faces)
        trimesh_mesh.update_faces(trimesh_mesh.unique_faces())
        trimesh_mesh.remove_unreferenced_vertices()

        mesh = {}
        mesh['vertices'] = np.array(trimesh_mesh.vertices)
        mesh['faces'] = np.array(trimesh_mesh.faces)

        np.savez(args.save_to, **mesh)
        # trimesh_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
        trimesh_mesh.show()

    node_lower, node_upper, mAs, mbs, lAs, lbs, uAs, ubs = [val for val in np.load(args.load_from).values()]

    register_plane_and_cube_with_polyscope(lAs, lbs, node_lower, node_upper)
    # register_shell(lAs, lbs, uAs, ubs, node_lower, node_upper)

if __name__ == '__main__':
    main()
