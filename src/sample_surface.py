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
# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CROWN_MODES = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward']

import warp as wp

# Initialize Warp
wp.init()


# Define GPU Kernel for Uniform Sampling
@wp.kernel
def sample_points_kernel(vertices: wp.array(dtype=wp.vec3),
                         faces: wp.array(dtype=int),
                         areas: wp.array(dtype=float),
                         sampled_points: wp.array(dtype=wp.vec3),
                         random_u: wp.array(dtype=float),
                         random_v: wp.array(dtype=float),
                         selected_faces: wp.array(dtype=int)):
    tid = wp.tid()  # Thread index
    f_id = selected_faces[tid]  # Selected face index

    # Get face vertices
    v0 = vertices[faces[f_id * 3]]
    v1 = vertices[faces[f_id * 3 + 1]]
    v2 = vertices[faces[f_id * 3 + 2]]

    # Convert random (u, v) to a barycentric coordinate
    sqrt_u = wp.sqrt(random_u[tid])
    bary_u = 1.0 - sqrt_u
    bary_v = sqrt_u * (1.0 - random_v[tid])
    bary_w = sqrt_u * random_v[tid]

    # Compute sampled point
    sampled_points[tid] = bary_u * v0 + bary_v * v1 + bary_w * v2


# Function to sample points
def sample_points_from_mesh(vertices, faces, num_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to Warp arrays
    vertices_wp = wp.array(vertices.cpu().numpy(), dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces.cpu().numpy().flatten(), dtype=int, device=device)

    # Compute face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0), dim=1)

    # Compute probability distribution for face selection
    face_probs = face_areas / face_areas.sum()
    selected_faces = torch.multinomial(face_probs, num_samples, replacement=True).to(device)

    # Generate random barycentric coordinates
    random_u = torch.rand(num_samples, device=device)
    random_v = torch.rand(num_samples, device=device)

    # Allocate Warp arrays
    sampled_points = wp.empty(num_samples, dtype=wp.vec3, device=device)
    random_u_wp = wp.array(random_u.cpu().numpy(), dtype=float, device=device)
    random_v_wp = wp.array(random_v.cpu().numpy(), dtype=float, device=device)
    selected_faces_wp = wp.array(selected_faces.cpu().numpy(), dtype=int, device=device)
    time_start = time.time()
    # Launch GPU kernel
    wp.launch(sample_points_kernel, dim=num_samples, inputs=[
        vertices_wp, faces_wp, None, sampled_points, random_u_wp, random_v_wp, selected_faces_wp
    ], device=device)
    time_elapsed = time.time() - time_start
    print("Total sampling time: ", time_elapsed)
    print("Average sampling time: ", time_elapsed / num_samples)
    return torch.tensor(sampled_points.numpy(), device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("load_from", type=str)

    # Parse arguments
    args = parser.parse_args()
    mesh_dict = np.load(args.load_from)
    tri_vertices = mesh_dict['vertices']
    tri_faces = mesh_dict['faces']
    trimesh_mesh = trimesh.Trimesh(tri_vertices, tri_faces)
    samples = sample_points_from_mesh(torch.from_numpy(tri_vertices), torch.from_numpy(tri_faces), 60000)
    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode='crown')
    implicit_func.torch_model.cuda()
    sdfs = implicit_func.torch_model(samples.float().cuda())
    print("Avg distance from surface", sdfs.abs().mean().item())
