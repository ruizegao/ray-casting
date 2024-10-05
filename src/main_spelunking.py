# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
from functools import partial
from gc import enable

import numpy as np
import argparse
import torch
import matplotlib
import matplotlib.pyplot as plt
import imageio
from skimage import measure
from functorch import vmap

import polyscope as ps
import polyscope.imgui as psim

# Imports from this project
import render, geometry, queries
from geometry import *
from utils import *
import affine
import slope_interval
import sdf
import mlp
from kd_tree import *
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
import implicit_mlp_utils, extract_cell
import affine_layers
import slope_interval_layers

# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def save_render_current_view(args, implicit_func, params, cast_frustum, opts, matcaps, surf_color, branching_method, cast_opt_based=False):
    root = torch.tensor([2., 0., 0.])
    look = torch.tensor([-1., 0., 0.])
    up = torch.tensor([0., 1., 0.])
    left = torch.tensor([0., 0., 1.])
    # root = torch.tensor([0., -2., 0.])
    # left = torch.tensor([1., 0., 0.])
    # look = torch.tensor([0., 1., 0.])
    # up = torch.tensor([0., 0., 1.])
    fov_deg = 60.
    res = args.res // opts['res_scale']

    surf_color = tuple(surf_color)

    img, depth, count, _, eval_sum, raycast_time = render.render_image(implicit_func, params, root, look, up, left, res, fov_deg, cast_frustum, branching_method, opts, shading='matcap_color', matcaps=matcaps, shading_color_tuple=(surf_color,), tree_based=cast_opt_based)
    print(depth, count, _, eval_sum)

    # flip Y
    # img = img[::-1,:,:]
    img = torch.flip(img, [0])
    # print(img[:3][:3][:3])
    # append an alpha channel
    # alpha_channel = (torch.min(img,dim=-1) < 1.) * 1.
    alpha_channel = (torch.min(img, dim=-1).values < 1.).float()
    # print(alpha_channel[:3, :3])
    # alpha_channel = torch.ones_like(img[:,:,0])
    img_alpha = torch.concatenate((img, alpha_channel[:,:,None]), dim=-1)
    img_alpha = torch.clip(img_alpha, min=0., max=1.)
    print(f"Saving image to {args.image_write_path}")
    imageio.imwrite(args.image_write_path, img_alpha.cpu().detach().numpy())


def do_sample_surface(opts, implicit_func, params, n_samples, sample_width, n_node_thresh, do_viz_tree, do_uniform_sample):
    data_bound = opts['data_bound']
    lower = torch.tensor((-data_bound, -data_bound, -data_bound))
    upper = torch.tensor((data_bound, data_bound, data_bound))

    rngkey = torch.Generator(device=device).manual_seed(42)

    print(f"do_sample_surface n_node_thresh {n_node_thresh}")

    with Timer("sample points"):
        sample_points = sample_surface(implicit_func, params, lower, upper, n_samples, sample_width, rngkey, n_node_thresh=n_node_thresh)
        # sample_points.block_until_ready()
        torch.cuda.synchronize()

    ps.register_point_cloud("sampled points", sample_points.cpu().numpy())


    # Build the tree all over again so we can visualize it
    if do_viz_tree:
        out_dict = construct_uniform_unknown_levelset_tree(implicit_func, params, lower, upper, n_node_thresh, offset=sample_width)
        node_valid = out_dict['unknown_node_valid']
        node_lower = out_dict['unknown_node_lower']
        node_upper = out_dict['unknown_node_upper']
        node_lower = node_lower[node_valid,:]
        node_upper = node_upper[node_valid,:]
        verts, inds = generate_tree_viz_nodes_simple(node_lower, node_upper, shrink_factor=0.05)
        ps_vol = ps.register_volume_mesh("tree nodes", verts.cpu().numpy(), hexes=inds.cpu().numpy())

    # If requested, also do uniform sampling
    if do_uniform_sample:

        with Timer("sample points uniform"):
            sample_points = sample_surface_uniform(implicit_func, params, lower, upper, n_samples, sample_width, rngkey)
            # sample_points.block_until_ready()
            torch.cuda.synchronize()

        ps.register_point_cloud("uniform sampled points", sample_points.cpu().numpy())



# def split_generator(generator, num_splits=2):
#     return [torch.Generator().manual_seed(generator.initial_seed() + i + 1) for i in range(num_splits)]


def do_hierarchical_mc(opts, implicit_func, params, n_mc_depth, do_viz_tree, compute_dense_cost, enable_clipping=False):


    data_bound = float(opts['data_bound'])
    # data_bound = 2.

    lower = torch.tensor((-data_bound, -data_bound, -data_bound))
    upper = torch.tensor((data_bound, data_bound, data_bound))

    # lower = torch.tensor((0.25, -0.6875, 0.46875), dtype=torch.float32)
    # upper = torch.tensor((0.28125, -0.65625, 0.5), dtype=torch.float32)
    # torch.set_printoptions(precision=8)
    # print("lower: ", lower)
    # lower = torch.tensor((0.25, -0.625, 0.4375), dtype=torch.float64)
    # upper = torch.tensor((0.3125, -0.6875, 0.5), dtype=torch.float64)
    print("Clipping enabled: ", enable_clipping)
    print(f"do_hierarchical_mc {n_mc_depth}")
    

    with Timer("extract mesh"):
        tri_pos = hierarchical_marching_cubes(implicit_func, params, lower, upper, n_mc_depth, enable_clipping=enable_clipping, n_subcell_depth=3)
        # tri_pos.block_until_ready()
        torch.cuda.synchronize()

    tri_inds = torch.reshape(torch.arange(3*tri_pos.shape[0]), (-1,3))
    tri_pos = torch.reshape(tri_pos, (-1,3))
    ps.register_surface_mesh("extracted mesh", tri_pos.cpu().numpy(), tri_inds.cpu().numpy())

    # Build the tree all over again so we can visualize it
    if do_viz_tree:
        n_mc_subcell=3
        if enable_clipping:
            out_dict = construct_non_uniform_unknown_levelset_tree(implicit_func, params, lower, upper,
                                                               split_depth=3 * (n_mc_depth - n_mc_subcell),
                                                               with_interior_nodes=True, with_exterior_nodes=True)
        else:
            out_dict = construct_uniform_unknown_levelset_tree(implicit_func, params, lower, upper, split_depth=3*(n_mc_depth-n_mc_subcell), with_interior_nodes=True, with_exterior_nodes=True)

        node_valid = out_dict['unknown_node_valid']
        node_lower = out_dict['unknown_node_lower']
        node_upper = out_dict['unknown_node_upper']
        node_lower = node_lower[node_valid,:]
        node_upper = node_upper[node_valid,:]
        verts, inds = generate_tree_viz_nodes_simple(node_lower, node_upper, shrink_factor=0.05)
        ps_vol = ps.register_volume_mesh("unknown tree nodes", verts.cpu().numpy(), hexes=inds.cpu().numpy())

        node_valid = out_dict['interior_node_valid']
        node_lower = out_dict['interior_node_lower']
        node_upper = out_dict['interior_node_upper']
        node_lower = node_lower[node_valid,:]
        node_upper = node_upper[node_valid,:]
        if node_lower.shape[0] > 0:
            verts, inds = generate_tree_viz_nodes_simple(node_lower, node_upper, shrink_factor=0.05)
            ps_vol = ps.register_volume_mesh("interior tree nodes", verts.cpu().numpy(), hexes=inds.cpu().numpy())
        
        node_valid = out_dict['exterior_node_valid']
        node_lower = out_dict['exterior_node_lower']
        node_upper = out_dict['exterior_node_upper']
        node_lower = node_lower[node_valid,:]
        # print("node lower:", node_lower)
        node_upper = node_upper[node_valid,:]
        if node_lower.shape[0] > 0:
            verts, inds = generate_tree_viz_nodes_simple(node_lower, node_upper, shrink_factor=0.05)
            ps_vol = ps.register_volume_mesh("exterior tree nodes", verts.cpu().numpy(), hexes=inds.cpu().numpy())

def do_closest_point(opts, func, params, n_closest_point):

    data_bound = float(opts['data_bound'])
    eps = float(opts['hit_eps'])
    lower = torch.tensor((-data_bound, -data_bound, -data_bound))
    upper = torch.tensor((data_bound,   data_bound,  data_bound))

    print(f"do_closest_point {n_closest_point}")
   
    # generate some query points
    rngkey = torch.Generator().manual_seed(n_closest_point)

    rngkey, subkey = split_generator(rngkey)
    query_points = (upper - lower) * torch.rand((n_closest_point,3), generator=subkey) + lower

    with Timer("closest point"):
        query_dist, query_min_loc = closest_point(func, params, lower, upper, query_points, eps=eps)
        # query_dist.block_until_ready()
        torch.cuda.synchronize()

    # visualize only the outside ones
    is_outside = vmap(partial(func,params))(query_points) > 0
    query_points = torch.tensor(query_points[is_outside,:])
    query_dist = query_dist[is_outside]
    query_min_loc = torch.tensor(query_min_loc[is_outside,:])

    viz_line_nodes = torch.reshape(torch.stack((query_points, query_min_loc), dim=1), (-1,3))
    viz_line_edges = torch.reshape(torch.arange(2*query_points.shape[0]), (-1,2))
    ps.register_point_cloud("closest point query", query_points.cpu().numpy())
    ps.register_point_cloud("closest point result", query_min_loc.cpu().numpy())
    ps.register_curve_network("closest point line", viz_line_nodes.cpu().numpy(), viz_line_edges.cpu().numpy())


def compute_bulk(args, implicit_func, params, opts):
    
    data_bound = float(opts['data_bound'])
    lower = torch.tensor((-data_bound, -data_bound, -data_bound))
    upper = torch.tensor((data_bound, data_bound, data_bound))
        
    rngkey = torch.Generator().manual_seed(0)

    with Timer("bulk properties"):
        mass, centroid = bulk_properties(implicit_func, params, lower, upper, rngkey)
        # mass.block_until_ready()
        torch.cuda.synchronize()

    print(f"Bulk properties:")
    print(f"  Mass: {mass}")
    print(f"  Centroid: {centroid}")

    ps.register_point_cloud("centroid", centroid.unsqueeze(0).cpu().numpy())

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    
    parser.add_argument("--res", type=int, default=1024)
    
    parser.add_argument("--image_write_path", type=str, default="render_out.png")

    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')
    parser.add_argument("--enable-clipping", action='store_true')
    # Parse arguments
    args = parser.parse_args()

    print(device)
    ## Small options
    debug_log_compiles = False
    debug_disable_jit = False
    debug_debug_nans = False
    # if args.log_compiles:
    #     jax.config.update("jax_log_compiles", 1)
    #     debug_log_compiles = True
    # if args.disable_jit:
    #     jax.config.update('jax_disable_jit', True)
    #     debug_disable_jit = True
    # if args.debug_nans:
    #     jax.config.update("jax_debug_nans", True)
    #     debug_debug_nans = True
    # if args.enable_double_precision:
    #     jax.config.update("jax_enable_x64", True)


    # GUI Parameters
    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 2
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    cast_frustum = False
    cast_opt_based = False
    mode = 'affine_fixed'
    modes = ['sdf', 'interval', 'affine_fixed', 'affine_truncate', 'affine_append', 'affine_all', 'affine_quad', 'slope_interval', 'crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward', 'dynamic_forward+backward', 'affine+backward']
    crown_modes = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
                   'dynamic_forward+backward']
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 8
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    affine_opts['affine+backward'] = 1.
    affine_opts['affine_quad'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = 'absolute'
    n_sample_pts = 100000
    sample_width = 0.01
    n_node_thresh = 4096
    do_uniform_sample = False
    do_viz_tree = False
    enable_clipping = False
    n_mc_depth = 8
    compute_dense_cost = False
    n_closest_point = 16
    shade_style = 'matcap_color'
    surf_color = (0.157,0.613,1.000)

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    # load the matcaps
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))

    def callback():

        nonlocal implicit_func, params, mode, modes, cast_frustum, cast_opt_based, debug_log_compiles, debug_disable_jit, debug_debug_nans, shade_style, surf_color, n_sample_pts, sample_width, n_node_thresh, do_uniform_sample, do_viz_tree, enable_clipping, n_mc_depth, compute_dense_cost, n_closest_point
            
    
        ## Options for general affine evaluation
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Eval options"):
            psim.PushItemWidth(100)
    
            old_mode = mode
            changed, mode = utils.combo_string_picker("Method", mode, modes)
            if mode != old_mode:
                implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

            if mode not in crown_modes:
                enable_clipping = False

            if mode == 'affine_truncate':
                # truncate options

                changed, affine_opts['affine_n_truncate'] = psim.InputInt("affine_n_truncate", affine_opts['affine_n_truncate'])
                if changed: 
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
            
                changed, affine_opts['affine_truncate_policy'] = utils.combo_string_picker("Method", affine_opts['affine_truncate_policy'], truncate_policies)
                if changed:
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
            
            if mode == 'affine_append':
                # truncate options

                changed, affine_opts['affine_n_append'] = psim.InputInt("affine_n_append", affine_opts['affine_n_append'])
                if changed: 
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
            
            if mode == 'sdf':

                changed, affine_opts['sdf_lipschitz'] = psim.InputFloat("SDF Lipschitz", affine_opts['sdf_lipschitz'])
                if changed:
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

            if mode == 'affine+backward':
                changed, affine_opts['affine+backward'] = psim.InputFloat("affine+backward", affine_opts['affine+backward'])
                if changed:
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)

            if mode == 'affine_quad':
                changed, affine_opts['affine_quad'] = psim.InputFloat("affine_quad", affine_opts['affine_quad'])
                if changed:
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)

            if mode in crown_modes:
                changed, enable_clipping = psim.Checkbox("enable_clipping", enable_clipping)
                if changed:
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **{"enable_clipping": enable_clipping})

            psim.PopItemWidth()
            psim.TreePop()


        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Raycast"):
            psim.PushItemWidth(100)
        
            if psim.Button("Save Render"):
                branching_method = None
                save_render_current_view(args, implicit_func, params, cast_frustum, opts, matcaps, surf_color, branching_method, cast_opt_based=cast_opt_based)


            _, cast_frustum = psim.Checkbox("cast frustum", cast_frustum)
            _, cast_opt_based = psim.Checkbox("cast opt based", cast_opt_based)
            _, opts['hit_eps'] = psim.InputFloat("delta", opts['hit_eps'])
            _, opts['max_dist'] = psim.InputFloat("max_dist", opts['max_dist'])

            if cast_frustum:
                _, opts['n_side_init'] = psim.InputInt("n_side_init", opts['n_side_init'])

            psim.PopItemWidth()
            psim.TreePop()


        # psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Sample Surface "):
            psim.PushItemWidth(100)

            if psim.Button("Sample"):
                do_sample_surface(opts, implicit_func, params, n_sample_pts, sample_width, n_node_thresh, do_viz_tree, do_uniform_sample)

            _, n_sample_pts = psim.InputInt("n_sample_pts", n_sample_pts)

            psim.SameLine()
            _, sample_width = psim.InputFloat("sample_width", sample_width)
            _, n_node_thresh = psim.InputInt("n_node_thresh", n_node_thresh)
            _, do_viz_tree = psim.Checkbox("viz tree", do_viz_tree)
            psim.SameLine()
            _, do_uniform_sample = psim.Checkbox("also uniform sample", do_uniform_sample)

            
            psim.PopItemWidth()
            psim.TreePop()


        if psim.TreeNode("Extract mesh"):
            psim.PushItemWidth(100)

            if psim.Button("Extract"):
                do_hierarchical_mc(opts, implicit_func, params, n_mc_depth, do_viz_tree, compute_dense_cost, enable_clipping)

            psim.SameLine()
            _, n_mc_depth = psim.InputInt("n_mc_depth", n_mc_depth)
            _, do_viz_tree = psim.Checkbox("viz tree", do_viz_tree)
            psim.SameLine()
            _, compute_dense_cost = psim.Checkbox("compute dense cost", compute_dense_cost)

            
            psim.PopItemWidth()
            psim.TreePop()
       

        if psim.TreeNode("Closest point"):
            psim.PushItemWidth(100)

            if psim.Button("Find closest pionts"):
                do_closest_point(opts, implicit_func, params, n_closest_point)

            _, n_closest_point= psim.InputInt("n_closest_point", n_closest_point)
            
            psim.PopItemWidth()
            psim.TreePop()

        ## Bulk
        if psim.TreeNode("Bulk Properties"):
            psim.PushItemWidth(100)
        
            if psim.Button("Compute bulk"):
                compute_bulk(args, implicit_func, params, opts)

            psim.PopItemWidth()
            psim.TreePop()


        if psim.TreeNode("Debug"):
            psim.PushItemWidth(100)

            changed, debug_log_compiles = psim.Checkbox("debug_log_compiles", debug_log_compiles)
            # if changed:
            #     jax.config.update("jax_log_compiles", 1 if debug_log_compiles else 0)
            #
            # changed, debug_disable_jit = psim.Checkbox("debug_disable_jit", debug_disable_jit)
            # if changed:
            #     jax.config.update('jax_disable_jit', debug_disable_jit)
            #
            # changed, debug_debug_nans = psim.Checkbox("debug_debug_nans", debug_debug_nans)
            # if changed:
            #     jax.config.update("jax_debug_nans", debug_debug_nans)

            
            psim.PopItemWidth()
            psim.TreePop()
               
    ps.set_use_prefs_file(False)
    ps.init()


    # Visualize the data via quick coarse marching cubes, so we have something to look at

    # Construct the regular grid
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

    print("REMEMBER: All routines will be slow on the first invocation due to JAX kernel compilation. Subsequent calls will be fast.")

    # Hand off control to the main callback
    ps.show(1)
    ps.set_user_callback(callback)
    ps.show()


if __name__ == '__main__':
    main()
