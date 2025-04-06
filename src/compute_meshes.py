# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
import time
import argparse
import warnings
import numpy as np
import torch
import imageio
import polyscope.imgui as psim
from trimesh.graph import is_watertight

# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
from scipy.spatial import Delaunay
import polyscope as ps
from skimage import measure
from mesh_utils import *
import trimesh

# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CROWN_MODES = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward']


def slice_box(plane_normal, plane_d, aabb_min, aabb_max, keep_pos=True):
    """
    Find the intersection points between a plane and an axis-aligned bounding box (AABB).

    Parameters:
    - plane_normal: (3,) array-like, normal vector (a, b, c) of the plane ax + by + cz + d = 0
    - plane_d: float, the d parameter in the plane equation
    - aabb_min: (3,) array-like, minimum corner (x_min, y_min, z_min) of the AABB
    - aabb_max: (3,) array-like, maximum corner (x_max, y_max, z_max) of the AABB

    Returns:
    - intersections: List of intersection points (each a 3D NumPy array)
    """
    # Generate the 8 vertices of the AABB
    # aabb_min = np.asarray(aabb_min)
    # aabb_max = np.asarray(aabb_max)

    vertices = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
    ])

    # Compute signed distances of the vertices to the plane
    signed_distances = np.dot(vertices, plane_normal) + plane_d

    # If all distances have the same sign, there's no intersection
    if np.all(signed_distances >= 0):
        if keep_pos:
            return trimesh.creation.box(bounds=np.vstack((aabb_min, aabb_max)))
        return None
    if np.all(signed_distances <= 0):
        if keep_pos:
            return None
        return trimesh.creation.box(bounds=np.vstack((aabb_min, aabb_max)))

    # Define the 12 edges of the AABB
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # z-direction
        (0, 2), (1, 3), (4, 6), (5, 7),  # y-direction
        (0, 4), (1, 5), (2, 6), (3, 7),  # x-direction
    ]

    verts_out = []
    for i1, i2 in edges:
        d1, d2 = signed_distances[i1], signed_distances[i2]
        if d1 * d2 <= 0:  # One point is inside, the other is outside
            t = -d1 / (d2 - d1)  # Interpolation factor
            p = vertices[i1] + t * (vertices[i2] - vertices[i1])
            verts_out.append(p)

    # Append points with negative signed distances
    if keep_pos:
        for i, d in enumerate(signed_distances):
            if d > 0:
                verts_out.append(vertices[i])
    else:
        for i, d in enumerate(signed_distances):
            if d < 0:
                verts_out.append(vertices[i])

    unique_verts_out = []
    for p in verts_out:
        if not any(np.linalg.norm(p - q) < 1e-6 for q in unique_verts_out):
            unique_verts_out.append(p)
    if len(unique_verts_out) < 4:
        return None

    mesh = trimesh.Trimesh(vertices=unique_verts_out)
    convex_polytope = mesh.convex_hull

    return convex_polytope

def build_shell(
        As: np.ndarray,
        bs: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        pos_lower: np.ndarray,
        pos_upper: np.ndarray,
        neg_lower: np.ndarray,
        neg_upper: np.ndarray,
        inflate=False
):
    start_time = time.time()

    num_success, num_errors = 0, 0
    trimesh_meshes = []
    for A, b, l, u in zip(As, bs, lower, upper):
        mesh = slice_box(A, b, l, u, keep_pos=False)
        if mesh:
            if inflate:
                inflation_amount = 5e-2 * (mesh.volume ** (1 / 3))
                # inflation_amount = 5e-4
                normals = mesh.vertex_normals
                mesh.vertices += normals * inflation_amount
            trimesh_meshes.append(mesh)
            num_success += 1
            print(num_success)

    for n_l, n_u in zip(neg_lower, neg_upper):
        mesh = trimesh.creation.box(bounds=np.vstack((n_l, n_u)))
        if inflate:
            inflation_amount = 5e-2 * (mesh.volume ** (1 / 3))
            # inflation_amount = 5e-4
            normals = mesh.vertex_normals
            mesh.vertices += normals * inflation_amount
        trimesh_meshes.append(mesh)
        num_success += 1
        print(num_success)

    end_time = time.time()
    print(f"Num success: {num_success}, Num errors: {num_errors}")
    print("total time cost: ", end_time - start_time)

    trimesh_mesh = trimesh.boolean.union(trimesh_meshes)
    print(len(trimesh_mesh.vertices))
    print(len(trimesh_mesh.faces))
    return trimesh_mesh

def carve_shell(
        As: np.ndarray,
        bs: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        pos_lower: np.ndarray,
        pos_upper: np.ndarray,
        neg_lower: np.ndarray,
        neg_upper: np.ndarray,
        inflate=False
):
    start_time = time.time()

    num_success, num_errors = 0, 0
    trimesh_meshes = []
    bbox = trimesh.creation.box(bounds=np.array([[-1., -1., -1.], [1., 1., 1.]]))
    trimesh_meshes.append(bbox)
    for A, b, l, u in zip(As, bs, lower, upper):
        mesh = slice_box(A, b, l, u, keep_pos=True)
        if mesh:
            if inflate:
                inflation_amount = 5e-2 * (mesh.volume ** (1 / 3))
                # inflation_amount = 5e-4
                normals = mesh.vertex_normals
                mesh.vertices += normals * inflation_amount
            trimesh_meshes.append(mesh)
            num_success += 1
            print(num_success)

    for p_l, p_u in zip(pos_lower, pos_upper):
        mesh = trimesh.creation.box(bounds=np.vstack((p_l, p_u)))
        if inflate:
            inflation_amount = 5e-2 * (mesh.volume ** (1 / 3))
            # inflation_amount = 5e-4
            normals = mesh.vertex_normals
            mesh.vertices += normals * inflation_amount
        trimesh_meshes.append(mesh)
        num_success += 1
        print(num_success)

    end_time = time.time()
    print(f"Num success: {num_success}, Num errors: {num_errors}")
    print("total time cost: ", end_time - start_time)

    trimesh_mesh = trimesh.boolean.difference(trimesh_meshes, check_volume=False)
    print(len(trimesh_mesh.vertices))
    print(len(trimesh_mesh.faces))
    return trimesh_mesh



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

    ret_val = [val for val in np.load(args.load_from).values()]
    [node_lower, node_upper, mAs, mbs, lAs, lbs, uAs, ubs, pos_lower, pos_upper, neg_lower, nge_upper, plane_constraints_lower, plane_constraints_upper] = ret_val

    num_constraints = plane_constraints_lower.shape[1]
    # num_constraints = 0
    print(f"Found {num_constraints} constraint plane(s) to add to the mesh")
    if num_constraints == 0:
        outer_shell = build_shell(lAs, lbs, node_lower, node_upper, pos_lower, pos_upper, neg_lower, nge_upper, inflate=True)
        outer_shell.fill_holes()
        outer_shell.export(args.save_to[:-4]+'_outer.obj')
        # outer_shell.show()
        # mid_shell = register_plane_and_cube_with_polyscope(mAs, mbs, node_lower, node_upper, pos_lower, pos_upper, neg_lower, nge_upper)
        # mid_shell.export(args.save_to[:-4]+'_mid.obj')
        inner_shell = carve_shell(uAs, ubs, node_lower, node_upper, pos_lower, pos_upper, neg_lower, nge_upper, inflate=True)
        inner_shell.fill_holes()
        inner_shell.export(args.save_to[:-4]+'_inner.obj')

        both_shell_verts = np.concatenate((np.array(outer_shell.vertices), np.array(inner_shell.vertices)), axis=0)
        both_shell_faces = np.concatenate((np.array(outer_shell.faces), np.array(inner_shell.faces) + len(outer_shell.vertices)), axis=0)
        both_shells = trimesh.Trimesh(both_shell_verts, both_shell_faces)
        both_shells.export(args.save_to[:-4]+'_both.obj')
    else:
        raise ValueError("More than 2 planes is not supported as of yet")
