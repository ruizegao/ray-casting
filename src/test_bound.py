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
    parser.add_argument("input", type=str)
    parser.add_argument("--load_from", type=str)
    parser.add_argument("--save_to", type=str)
    parser.add_argument("--mode", type=str, default='crown')
    parser.add_argument("--cast_frustum", action='store_true')
    parser.add_argument("--cast_tree_based", action='store_true')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--split_depth", type=int, default=21)

    parser.add_argument("--image_write_path", type=str, default="render_out.png")

    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')
    parser.add_argument("--enable_clipping", action='store_true')
    parser.add_argument("--heuristic", type=str, default='naive')
    parser.add_argument("--disable_ps", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    cast_frustum = args.cast_frustum
    cast_tree_based = args.cast_tree_based
    mode = args.mode
    split_depth = args.split_depth
    batch_size = args.batch_size
    enable_clipping = args.enable_clipping
    disable_ps = args.disable_ps
    modes = ['sdf', 'interval', 'affine_fixed', 'affine_truncate', 'affine_append', 'affine_all', 'slope_interval',
             'crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward', 'affine+backward', 'affine_quad']
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 8
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    affine_opts['affine+backward'] = 1.
    affine_opts['affine_quad'] = 1.
    affine_opts['enable_clipping'] = enable_clipping
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = 'absolute'
    surf_color = (0.157, 0.613, 1.000)

    if mode not in CROWN_MODES and enable_clipping:
        warnings.warn("'enable_clipping' was set to True but will be ignored since it is currently only supported for CROWN modes")

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

    if not disable_ps:
        ps.init()

    def generate_vertex_planes(normal_vectors: torch.Tensor,
                               offsets: torch.Tensor,
                               lowers: torch.Tensor,
                               uppers: torch.Tensor):
        # normal_vectors: Tensor of shape (n, 3)
        # offsets: Tensor of shape (n,)
        # lowers: Tensor of shape (n, 3)
        # uppers: Tensor of shape (n, 3)

        n = normal_vectors.shape[0]  # Number of planes/cubes

        # Determine which component (x, y, or z) has the largest magnitude for each normal vector
        abs_normals = torch.abs(normal_vectors)
        max_indices = torch.argmax(abs_normals, dim=1)  # Shape: (n,)

        # Prepare a tensor to store the vertices of each plane, shape: (n, 4, 3)
        vertices_plane = torch.zeros((n, 4, 3), dtype=normal_vectors.dtype, device=normal_vectors.device)

        # Process planes where z-component has the largest magnitude
        mask_z = max_indices == 2
        if mask_z.any():
            # Extract the relevant data for these planes
            normals_z = normal_vectors[mask_z]
            offsets_z = offsets[mask_z]
            lowers_z = lowers[mask_z]
            uppers_z = uppers[mask_z]

            # Compute the z values for the vertices
            z1 = -(normals_z[:, 0] * lowers_z[:, 0] + normals_z[:, 1] * lowers_z[:, 1] + offsets_z) / normals_z[:, 2]
            z2 = -(normals_z[:, 0] * lowers_z[:, 0] + normals_z[:, 1] * uppers_z[:, 1] + offsets_z) / normals_z[:, 2]
            z3 = -(normals_z[:, 0] * uppers_z[:, 0] + normals_z[:, 1] * uppers_z[:, 1] + offsets_z) / normals_z[:, 2]
            z4 = -(normals_z[:, 0] * uppers_z[:, 0] + normals_z[:, 1] * lowers_z[:, 1] + offsets_z) / normals_z[:, 2]

            # Store the vertices for these planes
            vertices_plane[mask_z] = torch.stack([
                torch.stack((lowers_z[:, 0], lowers_z[:, 1], z1), dim=-1),
                torch.stack((lowers_z[:, 0], uppers_z[:, 1], z2), dim=-1),
                torch.stack((uppers_z[:, 0], uppers_z[:, 1], z3), dim=-1),
                torch.stack((uppers_z[:, 0], lowers_z[:, 1], z4), dim=-1)
            ], dim=1)

        # Process planes where y-component has the largest magnitude
        mask_y = max_indices == 1
        if mask_y.any():
            normals_y = normal_vectors[mask_y]
            offsets_y = offsets[mask_y]
            lowers_y = lowers[mask_y]
            uppers_y = uppers[mask_y]

            y1 = -(normals_y[:, 0] * lowers_y[:, 0] + normals_y[:, 2] * lowers_y[:, 2] + offsets_y) / normals_y[:, 1]
            y2 = -(normals_y[:, 0] * lowers_y[:, 0] + normals_y[:, 2] * uppers_y[:, 2] + offsets_y) / normals_y[:, 1]
            y3 = -(normals_y[:, 0] * uppers_y[:, 0] + normals_y[:, 2] * uppers_y[:, 2] + offsets_y) / normals_y[:, 1]
            y4 = -(normals_y[:, 0] * uppers_y[:, 0] + normals_y[:, 2] * lowers_y[:, 2] + offsets_y) / normals_y[:, 1]

            vertices_plane[mask_y] = torch.stack([
                torch.stack((lowers_y[:, 0], y1, lowers_y[:, 2]), dim=-1),
                torch.stack((lowers_y[:, 0], y2, uppers_y[:, 2]), dim=-1),
                torch.stack((uppers_y[:, 0], y3, uppers_y[:, 2]), dim=-1),
                torch.stack((uppers_y[:, 0], y4, lowers_y[:, 2]), dim=-1)
            ], dim=1)

        # Process planes where x-component has the largest magnitude
        mask_x = max_indices == 0
        if mask_x.any():
            normals_x = normal_vectors[mask_x]
            offsets_x = offsets[mask_x]
            lowers_x = lowers[mask_x]
            uppers_x = uppers[mask_x]

            x1 = -(normals_x[:, 1] * lowers_x[:, 1] + normals_x[:, 2] * lowers_x[:, 2] + offsets_x) / normals_x[:, 0]
            x2 = -(normals_x[:, 1] * lowers_x[:, 1] + normals_x[:, 2] * uppers_x[:, 2] + offsets_x) / normals_x[:, 0]
            x3 = -(normals_x[:, 1] * uppers_x[:, 1] + normals_x[:, 2] * uppers_x[:, 2] + offsets_x) / normals_x[:, 0]
            x4 = -(normals_x[:, 1] * uppers_x[:, 1] + normals_x[:, 2] * lowers_x[:, 2] + offsets_x) / normals_x[:, 0]

            vertices_plane[mask_x] = torch.stack([
                torch.stack((x1, lowers_x[:, 1], lowers_x[:, 2]), dim=-1),
                torch.stack((x2, lowers_x[:, 1], uppers_x[:, 2]), dim=-1),
                torch.stack((x3, uppers_x[:, 1], uppers_x[:, 2]), dim=-1),
                torch.stack((x4, uppers_x[:, 1], lowers_x[:, 2]), dim=-1)
            ], dim=1)

        return vertices_plane.reshape(-1, 3)

    def register_plane_and_cube_with_polyscope(
            lAs: torch.Tensor,
            lbs: torch.Tensor,
            uAs: torch.Tensor,
            ubs: torch.Tensor,
            lower: torch.Tensor,
            upper: torch.Tensor):
        """
        Register a 3D plane and a cube in Polyscope using a normal vector, offset, and cube bounds.

        Args:
        - normal_vector: A torch.Tensor of shape (3,) representing the normal vector [a, b, c].
        - offset: A torch.Tensor representing the offset (d) in the plane equation ax + by + cz + d = 0.
        - lower: A torch.Tensor of shape (3,) representing the lower corner of the cube.
        - upper: A torch.Tensor of shape (3,) representing the upper corner of the cube.
        - grid_size: The size of the grid to visualize the plane.
        - grid_density: The number of points per axis to sample in the grid.
        """

        # Initialize Polyscope
        # ps.init()

        # ---------------- Visualize Plane ---------------- #
        # vertices_plane_l = torch.cat(vp_list_l, dim=0)#[np.repeat(mask, 4)]
        vertices_plane_l = generate_vertex_planes(lAs, lbs, lower, upper)

        # Create faces for the plane mesh
        faces_plane_l = np.arange(len(vertices_plane_l)).reshape(-1, 4)


        tri_faces = []
        tri_vertices = []
        intersect_count = 0
        for lA, lb, l, u in zip(lAs, lbs, lower, upper):
            cube = trimesh.creation.box(bounds=[l.cpu().numpy(), u.cpu().numpy()])
            origin = torch.tensor([0, 0, -lb/lA[2]])
            slice_mesh = cube.section(plane_origin=origin.cpu().numpy(), plane_normal=lA.cpu().numpy())
            if slice_mesh:
                vertices = slice_mesh.vertices
                # print(vertices.shape)
                vertices = sort_vertices(
                    points_on_cube_edges(lower_corner=l, upper_corner=u, points=vertices))
                tri = triangulate(vertices)
                tri_faces.append(tri + intersect_count)
                tri_vertices.append(vertices)
                intersect_count += len(vertices)

        tri_faces = torch.cat(tri_faces, dim=0)
        tri_vertices = torch.cat(tri_vertices, dim=0)
        vertices_plane_l = tri_vertices
        faces_plane_l = tri_faces.cpu().numpy()

        mesh = {}
        mesh['vertices'] = vertices_plane_l.cpu().numpy()
        mesh['faces'] = faces_plane_l

        np.savez('meshes/mesh_0.npz', **mesh)
        trimesh_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
        trimesh_mesh.show()
        if uAs is not None and ubs is not None:
            # vertices_plane_u = generate_vertex_planes(uAs, ubs, lower, upper)

            # Create faces for the plane mesh (triangulation)
            # faces_plane_u = np.arange(len(vertices_plane_u)).reshape(-1, 4)
            # Register the surface mesh for the plane with Polyscope

            tri_faces = []
            tri_vertices = []
            intersect_count = 0
            for lA, lb, l, u in zip(uAs, ubs, lower, upper):
                cube = trimesh.creation.box(bounds=[l.cpu().numpy(), u.cpu().numpy()])
                origin = torch.tensor([0, 0, -lb / lA[2]])
                slice_mesh = cube.section(plane_origin=origin.cpu().numpy(), plane_normal=lA.cpu().numpy())
                if slice_mesh:
                    vertices = slice_mesh.vertices
                    vertices = sort_vertices(
                        points_on_cube_edges(lower_corner=l, upper_corner=u, points=vertices))
                    tri = triangulate(vertices)
                    tri_faces.append(tri + intersect_count)
                    tri_vertices.append(vertices)
                    intersect_count += len(vertices)


            tri_faces = torch.cat(tri_faces, dim=0)
            tri_vertices = torch.cat(tri_vertices, dim=0)
            vertices_plane_u = tri_vertices
            faces_plane_u = tri_faces.cpu().numpy()

        # ---------------- Visualize Cube ---------------- #
        # Register the cube mesh with Polyscope
        verts, inds = generate_tree_viz_nodes_simple(lower, upper)
        if not disable_ps:
            ps.register_surface_mesh('planes_l', vertices_plane_l.cpu().numpy(), faces_plane_l)
            if uAs is not None and ubs is not None:
                ps.register_surface_mesh('planes_u', vertices_plane_u.cpu().numpy(), faces_plane_u)
            ps.register_volume_mesh("unknown tree nodes", verts.cpu().numpy(), hexes=inds.cpu().numpy())

    def planes_intersect_cubes(normals, offsets, lowers, uppers):
        # normals: Tensor of shape (n, 3) where n is the number of planes
        # offsets: Tensor of shape (n,)
        # lowers: Tensor of shape (n, 3) where n is the number of cubes
        # uppers: Tensor of shape (n, 3)

        # Create all 8 vertices for each cube
        x_min, y_min, z_min = lowers.T
        x_max, y_max, z_max = uppers.T

        vertices = torch.stack([
            torch.stack((x_min, y_min, z_min), dim=-1),
            torch.stack((x_min, y_min, z_max), dim=-1),
            torch.stack((x_min, y_max, z_min), dim=-1),
            torch.stack((x_min, y_max, z_max), dim=-1),
            torch.stack((x_max, y_min, z_min), dim=-1),
            torch.stack((x_max, y_min, z_max), dim=-1),
            torch.stack((x_max, y_max, z_min), dim=-1),
            torch.stack((x_max, y_max, z_max), dim=-1)
        ], dim=1)  # Shape: (n, 8, 3)

        # Compute the dot products for each vertex with the corresponding plane normal
        # Resulting shape: (n, 8)
        dots = torch.matmul(vertices, normals.unsqueeze(-1)).squeeze(-1) + offsets.unsqueeze(-1)

        # Check if the vertices for each cube have different signs when plugged into plane equations
        signs = (dots > 0).int()

        # Check if each cube has both positive and negative signs for its plane, indicating intersection
        intersects = (signs.min(dim=-1).values < 1) & (signs.max(dim=-1).values > 0)

        return intersects  # Shape: (n,)

    grid_res = 128
    ax_coords = torch.linspace(-1., 1., grid_res)
    grid_x, grid_y, grid_z = torch.meshgrid(ax_coords, ax_coords, ax_coords, indexing='ij')
    grid = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=-1)
    delta = (grid[1,2] - grid[0,2]).item()
    sdf_vals = vmap(partial(implicit_func, params))(grid)
    sdf_vals = sdf_vals.reshape(grid_res, grid_res, grid_res)
    bbox_min = grid[0,:]
    verts, faces, normals, values = measure.marching_cubes(sdf_vals.cpu().numpy(), level=0., spacing=(delta, delta, delta))
    verts = torch.from_numpy(verts).to(device)
    verts = verts + bbox_min[None,:]
    if not disable_ps:
        ps.register_surface_mesh("coarse shape preview", verts.cpu().numpy(), faces)
    if args.load_from:
        node_lower, node_upper, node_type, split_dim, split_val, lAs, lbs, uAs, ubs, node_guaranteed = [torch.from_numpy(val).to(device) for val in np.load(args.load_from).values()]
        node_lower_last_layer = node_lower[2 ** split_depth - 1: 2 ** (split_depth + 1) - 1]
        node_upper_last_layer = node_upper[2 ** split_depth - 1: 2 ** (split_depth + 1) - 1]


    else:
        node_lower, node_upper, node_type, split_dim, split_val = construct_full_uniform_unknown_levelset_tree(implicit_func, params, -torch.ones((1,3)), torch.ones((1,3)), split_depth=split_depth, batch_size=args.batch_size)

        node_lower_last_layer = node_lower[2 ** split_depth - 1: 2 ** (split_depth + 1) - 1]
        node_upper_last_layer = node_upper[2 ** split_depth - 1: 2 ** (split_depth + 1) - 1]
        node_valid = torch.logical_not(torch.isnan(node_lower_last_layer[:, 0]))

        lower = node_lower_last_layer[node_valid,:]
        upper = node_upper_last_layer[node_valid,:]
        # print(sum(node_valid.tolist()))

        total_samples = len(lower)

        lAs_valid = torch.empty((total_samples, 3))
        lbs_valid = torch.empty((total_samples,))
        uAs_valid = torch.empty((total_samples, 3))
        ubs_valid = torch.empty((total_samples,))

        lAs = torch.empty((2 ** split_depth, 3))
        lbs = torch.empty((2 ** split_depth,))
        uAs = torch.empty((2 ** split_depth, 3))
        ubs = torch.empty((2 ** split_depth,))

        batch_size_per_iteration = args.batch_size
        t0 = time.time()
        implicit_func.change_mode('crown')
        print(implicit_func.crown_mode)
        for start_idx in range(0, total_samples, batch_size_per_iteration):
            end_idx = min(start_idx + batch_size_per_iteration, total_samples)
            out_type, crown_ret = implicit_func.classify_box(params, lower[start_idx:end_idx], upper[start_idx:end_idx])
            lAs_valid[start_idx:end_idx] = crown_ret['lA'].squeeze(1)
            lbs_valid[start_idx:end_idx] = crown_ret['lbias'].squeeze(1)
            uAs_valid[start_idx:end_idx] = crown_ret['uA'].squeeze(1)
            ubs_valid[start_idx:end_idx] = crown_ret['ubias'].squeeze(1)
        t1 = time.time()
        print(t1 - t0)
        lAs[node_valid] = lAs_valid
        lbs[node_valid] = lbs_valid
        uAs[node_valid] = uAs_valid
        ubs[node_valid] = ubs_valid

        # mask_l = torch.tensor([plane_intersects_cube(nv, os, l, u) for nv, os, l, u in zip(lAs_valid, lbs_valid, lower, upper)])

        # mask_u = torch.tensor([plane_intersects_cube(nv, os, l, u) for nv, os, l, u in zip(uAs_valid, ubs_valid, lower, upper)])
        mask_l = planes_intersect_cubes(lAs_valid, lbs_valid, lower, upper)
        mask_u = planes_intersect_cubes(uAs_valid, ubs_valid, lower, upper)
        t2 = time.time()
        print(t2 - t1)
        # mask = torch.full_like(mask_l, True)
        mask = mask_l | mask_u

        mAs_valid = (lAs_valid + uAs_valid) / 2.
        mbs_valid = (lbs_valid + ubs_valid) / 2.
        # mask = planes_intersect_cubes(mAs_valid, mbs_valid, lower, upper)
        node_guaranteed = node_valid.clone()
        node_guaranteed[node_valid] = mask
        node_guaranteed = torch.cat((torch.full((2 ** split_depth - 1,), False), node_guaranteed))
        lAs = torch.cat((torch.full((2 ** split_depth - 1, 3), torch.nan), lAs), dim=0)
        lbs = torch.cat((torch.full((2 ** split_depth - 1,), torch.nan), lbs))
        uAs = torch.cat((torch.full((2 ** split_depth - 1, 3), torch.nan), uAs), dim=0)
        ubs = torch.cat((torch.full((2 ** split_depth - 1,), torch.nan), ubs))

        if args.save_to:
            tree = {}
            tree['node_lower'] = node_lower.cpu().numpy()
            tree['node_upper'] = node_upper.cpu().numpy()
            tree['node_type'] = node_type.cpu().numpy()
            tree['split_dim'] = split_dim.cpu().numpy()
            tree['split_val'] = split_val.cpu().numpy()
            tree['lAs'] = lAs.cpu().numpy()
            tree['lbs'] = lbs.cpu().numpy()
            tree['uAs'] = uAs.cpu().numpy()
            tree['ubs'] = ubs.cpu().numpy()
            tree['node_guaranteed'] = node_guaranteed.cpu().numpy()

            np.savez(args.save_to, **tree)

    mAs = (lAs + uAs) / 2
    mbs = (lbs + ubs) / 2
    # register_plane_and_cube_with_polyscope(lAs[node_guaranteed], lbs[node_guaranteed], uAs[node_guaranteed], ubs[node_guaranteed], node_lower_last_layer[node_guaranteed[2 ** split_depth - 1:]], node_upper_last_layer[node_guaranteed[2 ** split_depth - 1:]])
    register_plane_and_cube_with_polyscope(mAs[node_guaranteed], mbs[node_guaranteed], None, None, node_lower_last_layer[node_guaranteed[2 ** split_depth - 1:]], node_upper_last_layer[node_guaranteed[2 ** split_depth - 1:]])
    # register_plane_and_cube_with_polyscope(mAs_valid[mask], mbs_valid[mask], None, None, node_lower_last_layer[node_guaranteed[2 ** split_depth - 1:]], node_upper_last_layer[node_guaranteed[2 ** split_depth - 1:]])

    if not disable_ps:
        ps.show()


if __name__ == '__main__':
    main()
