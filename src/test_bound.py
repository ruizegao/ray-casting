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

# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CROWN_MODES = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)



def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("--load_from", type=str)
    parser.add_argument("--save_to", type=str)
    parser.add_argument("--mode", type=str, default='affine_fixed')
    parser.add_argument("--cast_frustum", action='store_true')
    parser.add_argument("--cast_tree_based", action='store_true')
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--res", type=int, default=1024)

    parser.add_argument("--image_write_path", type=str, default="render_out.png")

    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')
    parser.add_argument("--enable_clipping", action='store_true')
    parser.add_argument("--heuristic", type=str, default='naive')

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
    batch_size = args.batch_size
    enable_clipping = args.enable_clipping
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

    lower = torch.tensor(
        [[-1., -1., -1.],
         [0., -1., 0.],
         [ 0.5000, -1.0000, -1.0000],
         [ 0.5000, -0.7500, -0.2500],
         [ 0.5000,  0.7500, -0.2500],
         [-0.5000, -0.1250,  0.0000],
         [-0.5000, -0.7500,  0.3125],
         [ 0.1250, -0.4375, -0.1875],
         [-0.0625, -0.8125,  0.1875],
         [-0.1875, -0.3750, -0.1875],
         [-0.0625, 0.2812, -0.3438],
         [-0.1250, -0.1875, 0.0938],
         [-0.1875, -0.7812, 0.3125],
         [-0.0625, 0.2188, -0.0312]]
    )

    upper = torch.tensor(
        [[1., 1., 1.],
         [1., 0., 1.],
         [1., 0., 0.],
         [ 0.7500, -0.5000,  0.0000],
         [0.7500, 1.0000, 0.0000],
         [-0.3750,  0.0000,  0.1250],
         [-0.4375, -0.6875,  0.3750],
         [ 0.1875, -0.3750, -0.1250],
         [ 0.0000, -0.7500,  0.2500],
         [-0.1562, -0.3438, -0.1562],
         [-0.0312, 0.3125, -0.3125],
         [-0.0938, -0.1562, 0.1250],
         [-0.1562, -0.7500, 0.3438],
         [-0.0312, 0.2500, 0.0000]]
    )

    ps.init()
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
    ps.register_surface_mesh("coarse shape preview", verts.cpu().numpy(), faces)
    if args.load_from:
        node_valid, node_lower, node_upper = [torch.from_numpy(val).to(device) for val in np.load(args.load_from).values()]
    else:
        out_dict = construct_uniform_unknown_levelset_tree(implicit_func, params, -torch.ones(3), torch.ones(3), split_depth=24)
        node_valid = out_dict['unknown_node_valid']
        node_lower = out_dict['unknown_node_lower']
        node_upper = out_dict['unknown_node_upper']
        if args.save_to:
            out_dict['unknown_node_valid'] = node_valid.cpu().numpy()
            out_dict['unknown_node_lower'] = node_lower.cpu().numpy()
            out_dict['unknown_node_upper'] = node_upper.cpu().numpy()
            np.savez(args.save_to, **out_dict)
    lower = node_lower[node_valid,:]
    upper = node_upper[node_valid,:]
    print(node_lower.shape[0])
    print(sum(node_valid.tolist()))
    def generate_vertex_planes(normal_vector: torch.Tensor,
            offset: torch.Tensor,
            lower: torch.Tensor,
            upper: torch.Tensor,):
        # print(normal_vector)
        if np.argmax(np.abs(normal_vector)) == 2:
            z1 = -(normal_vector[0] * lower[0] + normal_vector[1] * lower[1] + offset) / normal_vector[2]  # Solve for z
            z2 = -(normal_vector[0] * lower[0] + normal_vector[1] * upper[1] + offset) / normal_vector[2]  # Solve for z
            z3 = -(normal_vector[0] * upper[0] + normal_vector[1] * upper[1] + offset) / normal_vector[2]  # Solve for z
            z4 = -(normal_vector[0] * upper[0] + normal_vector[1] * lower[1] + offset) / normal_vector[2]  # Solve for z
            # Flatten arrays for vertices
            # vertices_plane = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
            vertices_plane = np.array([
                [lower[0], lower[1], z1],
                [lower[0], upper[1], z2],
                [upper[0], upper[1], z3],
                [upper[0], lower[1], z4],
            ])
            maximum = max([z1, z2, z3, z4])
            minimum = min([z1, z2, z3, z4])
            # print("z diff: ", maximum - minimum)
        elif np.argmax(np.abs(normal_vector)) == 1:
            y1 = -(normal_vector[0] * lower[0] + normal_vector[2] * lower[2] + offset) / normal_vector[1]  # Solve for z
            y2 = -(normal_vector[0] * lower[0] + normal_vector[2] * upper[2] + offset) / normal_vector[1]  # Solve for z
            y3 = -(normal_vector[0] * upper[0] + normal_vector[2] * upper[2] + offset) / normal_vector[1]  # Solve for z
            y4 = -(normal_vector[0] * upper[0] + normal_vector[2] * lower[2] + offset) / normal_vector[1]  # Solve for z
            # Flatten arrays for vertices
            # vertices_plane = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
            vertices_plane = np.array([
                [lower[0], y1, lower[2]],
                [lower[0], y2, upper[2]],
                [upper[0], y3, upper[2]],
                [upper[0], y4, lower[2]],
            ])
            maximum = max([y1, y2, y3, y4])
            minimum = min([y1, y2, y3, y4])
            # print("y diff: ", maximum - minimum)
        else:
            x1 = -(normal_vector[1] * lower[1] + normal_vector[2] * lower[2] + offset) / normal_vector[0]  # Solve for z
            x2 = -(normal_vector[1] * lower[1] + normal_vector[2] * upper[2] + offset) / normal_vector[0]  # Solve for z
            x3 = -(normal_vector[1] * upper[1] + normal_vector[2] * upper[2] + offset) / normal_vector[0]  # Solve for z
            x4 = -(normal_vector[1] * upper[1] + normal_vector[2] * lower[2] + offset) / normal_vector[0]  # Solve for z
            # Flatten arrays for vertices
            # vertices_plane = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
            vertices_plane = np.array([
                [x1, lower[1], lower[2]],
                [x2, lower[1], upper[2]],
                [x3, upper[1], upper[2]],
                [x4, upper[1], lower[2]],
            ])
            maximum = max([x1, x2, x3, x4])
            minimum = min([x1, x2, x3, x4])
            # print("x diff: ", maximum - minimum)
        return vertices_plane

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
        # Convert torch tensors to numpy arrays
        lAs = lAs.cpu().numpy()
        lbs = lbs.cpu().numpy()
        uAs = uAs.cpu().numpy()
        ubs = ubs.cpu().numpy()
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()

        # Initialize Polyscope
        # ps.init()

        # ---------------- Visualize Plane ---------------- #
        vp_list_l = [generate_vertex_planes(nv, os, l, u) for nv, os, l, u in zip(lAs, lbs, lower, upper)]
        mask_l = np.array([plane_intersects_cube(nv, os, l, u) for nv, os, l, u in zip(lAs, lbs, lower, upper)])
        vertices_plane_l = np.concatenate(vp_list_l, axis=0)[np.repeat(mask_l, 4)]
        # Create faces for the plane mesh (triangulation)
        faces_plane_l = np.arange(len(vertices_plane_l)).reshape(-1, 4)
        # Register the surface mesh for the plane with Polyscope
        ps.register_surface_mesh('planes_l', vertices_plane_l, faces_plane_l)

        vp_list_u = [generate_vertex_planes(nv, os, l, u) for nv, os, l, u in zip(uAs, ubs, lower, upper)]
        mask_u = np.array([plane_intersects_cube(nv, os, l, u) for nv, os, l, u in zip(uAs, ubs, lower, upper)])
        # mask_u = mask_l
        vertices_plane_u = np.concatenate(vp_list_u, axis=0)[np.repeat(mask_u, 4)]
        # Create faces for the plane mesh (triangulation)
        faces_plane_u = np.arange(len(vertices_plane_u)).reshape(-1, 4)
        # Register the surface mesh for the plane with Polyscope
        ps.register_surface_mesh('planes_u', vertices_plane_u, faces_plane_u)
        mask = mask_l | mask_u
        print(np.sum(mask), mask.shape, np.sum(mask)/mask.shape)
        # ---------------- Visualize Cube ---------------- #
        # Vertices of the cube (8 corners)
        # vertices_cube = np.array([
        #     [lower[0], lower[1], lower[2]],
        #     [lower[0], lower[1], upper[2]],
        #     [lower[0], upper[1], lower[2]],
        #     [lower[0], upper[1], upper[2]],
        #     [upper[0], lower[1], lower[2]],
        #     [upper[0], lower[1], upper[2]],
        #     [upper[0], upper[1], lower[2]],
        #     [upper[0], upper[1], upper[2]]
        # ])

        # Faces of the cube (12 triangles)
        # faces_cube = np.array([
        #     [0, 1, 2], [1, 3, 2],  # Front face
        #     [4, 6, 5], [5, 6, 7],  # Back face
        #     [0, 4, 1], [1, 4, 5],  # Bottom face
        #     [2, 3, 6], [3, 7, 6],  # Top face
        #     [0, 2, 4], [2, 6, 4],  # Left face
        #     [1, 5, 3], [3, 5, 7]  # Right face
        # ])

        # Register the cube mesh with Polyscope
        # ps.register_surface_mesh('cube_'+name, vertices_cube, faces_cube)
        verts, inds = generate_tree_viz_nodes_simple(torch.from_numpy(lower[mask_l]), torch.from_numpy(upper[mask_l]))
        ps.register_volume_mesh("unknown tree nodes", verts.cpu().numpy(), hexes=inds.cpu().numpy())



    def plane_intersects_cube(normal, offset, lower, upper):
        x_min, y_min, z_min = lower
        x_max, y_max, z_max = upper

        vertices = [
            np.array((x_min, y_min, z_min)),
            np.array((x_min, y_min, z_max)),
            np.array((x_min, y_max, z_min)),
            np.array((x_min, y_max, z_max)),
            np.array((x_max, y_min, z_min)),
            np.array((x_max, y_min, z_max)),
            np.array((x_max, y_max, z_min)),
            np.array((x_max, y_max, z_max))
        ]

        signs = set()

        for vertex in vertices:
            d = np.dot(vertex, normal) + offset
            # print(d)
            signs.add(int(d > 0))  # True for positive, False for negative

        if len(signs) > 1:
            return True  # Cube intersects the plane
        else:
            return False  # Cube is entirely on one side of the plane

    # def surface_intersects_cube(func, lower, upper):
    #     x_min, y_min, z_min = lower
    #     x_max, y_max, z_max = upper
    #
    #     vertices = [
    #         torch.tensor((x_min, y_min, z_min)),
    #         torch.tensor((x_min, y_min, z_max)),
    #         torch.tensor((x_min, y_max, z_min)),
    #         torch.tensor((x_min, y_max, z_max)),
    #         torch.tensor((x_max, y_min, z_min)),
    #         torch.tensor((x_max, y_min, z_max)),
    #         torch.tensor((x_max, y_max, z_min)),
    #         torch.tensor((x_max, y_max, z_max))
    #     ]
    #
    #     signs = set()
    #
    #     for vertex in vertices:
    #         d = func(params, vertex)
    #         # print(d)
    #         signs.add(int(d > 0))  # True for positive, False for negative
    #
    #     if len(signs) > 1:
    #         return True  # Cube intersects the plane
    #     else:
    #         return False  # Cube is entirely on one side of the plane

    total_samples = len(lower)

    lAs = torch.empty((total_samples, 3))
    lbs = torch.empty((total_samples,))
    uAs = torch.empty((total_samples, 3))
    ubs = torch.empty((total_samples,))

    batch_size_per_iteration = args.batch_size
    for start_idx in range(0, total_samples, batch_size_per_iteration):
        end_idx = min(start_idx + batch_size_per_iteration, total_samples)
        out_type, crown_ret = implicit_func.classify_box(params, lower[start_idx:end_idx], upper[start_idx:end_idx])
        lAs[start_idx:end_idx] = crown_ret['lA'].squeeze(1)
        lbs[start_idx:end_idx] = crown_ret['lbias'].squeeze(1)
        uAs[start_idx:end_idx] = crown_ret['uA'].squeeze(1)
        ubs[start_idx:end_idx] = crown_ret['ubias'].squeeze(1)


    # register_plane_and_cube_with_polyscope(lAs[:25], lbs[:25], lower[:25], upper[:25], name='l')
    # register_plane_and_cube_with_polyscope(uAs[:25], ubs[:25], lower[:25], upper[:25], name='u')
    register_plane_and_cube_with_polyscope(lAs, lbs, uAs, ubs, lower, upper)
    # register_plane_and_cube_with_polyscope(uAs, ubs, lower, upper, name='u')
    # _, crown_ret = implicit_func.classify_box(params, lower, upper)
    # print(len(crown_ret))
    # _ = vmap(partial(implicit_func.classify_box, params))(lower, upper)

    ps.show()


if __name__ == '__main__':
    main()
