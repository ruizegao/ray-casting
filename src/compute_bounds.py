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
# import trimesh
from auto_LiRPA.hyperplane_volume_intersection import custom_loss_batch_estimate_volume

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
    parser.add_argument("save_to", type=str)
    parser.add_argument("--mode", type=str, default='crown')
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--split_depth", type=int, default=21)
    parser.add_argument("--max_split_depth", type=int, default=36)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alpha_pass", action='store_true')
    parser.add_argument("--use_cache", action='store_true')
    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    mode = args.mode
    use_cache = args.use_cache
    split_depth = args.split_depth
    max_split_depth = args.max_split_depth
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

    data_bound = float(opts['data_bound'])
    lower = torch.tensor((-data_bound, -data_bound, -data_bound), dtype=set_t['dtype'])
    upper = torch.tensor((data_bound, data_bound, data_bound), dtype=set_t['dtype'])
    start_time = time.time()
    print(f"Using implicit function with mode {mode} and type {type(implicit_func)}")
    if not use_cache:
        print(f"Not using cache, computing bounds and saving to (and potentially overwriting) {cache_dir}")
        # out_dict = construct_uniform_unknown_levelset_tree(implicit_func, params, lower, upper, split_depth=split_depth, with_interior_nodes=True)
        # out_dict = construct_adaptive_tree(implicit_func, params, lower, upper, split_depth=split_depth, with_interior_nodes=True)
        node_lower, node_upper, lAs, lbs, uAs, ubs = construct_hybrid_unknown_tree(implicit_func, params, lower, upper, base_depth=split_depth, max_depth=max_split_depth, delta=0.001, batch_size=256)
        lAs = lAs.cpu().numpy()
        lbs = lbs.cpu().numpy()
        uAs = uAs.cpu().numpy()
        ubs = ubs.cpu().numpy()
        node_valid = torch.full((node_lower.shape[0],), True)
        # node_valid = torch.cat((out_dict['unknown_node_valid'], out_dict['interior_node_valid']), dim=0)
        # node_lower = torch.cat((out_dict['unknown_node_lower'], out_dict['interior_node_lower']), dim=0)
        # node_upper = torch.cat((out_dict['unknown_node_upper'], out_dict['interior_node_upper']), dim=0)
        # FIXME: Tree construction should support returning the bounding planes
        # node_lA = torch.empty(1)
        # node_uA = torch.empty(1)
        # node_lbias = torch.empty(1)
        # node_ubias = torch.empty(1)
        cache_dict = {
            "node_valid": to_numpy(node_valid),
            "node_lower": to_numpy(node_lower),
            "node_upper": to_numpy(node_upper),
            # "node_lA": to_numpy(node_lA),
            # "node_uA": to_numpy(node_uA),
            # "node_lbias": to_numpy(node_lbias),
            # "node_ubias": to_numpy(node_ubias),
        }
        np.savez(cache_dir, **cache_dict)
    else:
        modification_time = os.path.getmtime(cache_dir)
        modification_datetime = datetime.datetime.fromtimestamp(modification_time)
        formatted_time = modification_datetime.strftime('%Y-%m-%d %I:%M %p')
        print(f"Using cache, reading from {cache_dir}, last modified: {formatted_time}")
        loaded_cache = np.load(cache_dir)
        node_valid = torch.from_numpy(loaded_cache["node_valid"]).to(device=device)
        node_lower = torch.from_numpy(loaded_cache["node_lower"]).to(device=device)
        node_upper = torch.from_numpy(loaded_cache["node_upper"]).to(device=device)
        # FIXME: This will not work until the planes actually get saved
        # node_lA = torch.from_numpy(loaded_cache["node_lA"]).to(device=device)
        # node_uA = torch.from_numpy(loaded_cache["node_uA"]).to(device=device)
        # node_lbias = torch.from_numpy(loaded_cache["node_lbias"]).to(device=device)
        # node_ubias = torch.from_numpy(loaded_cache["node_ubias"]).to(device=device)

    tree_time = time.time() - start_time
    print(f"Time to build (or load tree from cache): {tree_time}")

    ### first, we do a pass of CROWN to save the planes ###
    # FIXME: Eventaully this first pass should be removed and the above tree construction should give us the first
    #   set of plane constraints

    node_lower_valid = node_lower[node_valid]
    node_upper_valid = node_upper[node_valid]
    num_valid = node_valid.sum().item()
    # lAs = np.empty_like(to_numpy(node_lower_valid))
    # lbs = np.empty((num_valid,))
    # uAs = np.empty_like(to_numpy(node_lower_valid))
    # ubs = np.empty((num_valid,))
    # for start_idx in range(0, num_valid, batch_size):
    #     end_idx = min(start_idx + batch_size, num_valid)
    #     i = start_idx // batch_size
    #     print(f"i: {i} | start_idx: {start_idx}, end_idx: {end_idx}, num_valid: {num_valid}")
    #     out_type, crown_ret = implicit_func.classify_box(params, node_lower_valid[start_idx:end_idx], node_upper_valid[start_idx:end_idx], swap_loss=True)
    #     lAs[start_idx:end_idx] = to_numpy(crown_ret['lA'].squeeze(1))
    #     lbs[start_idx:end_idx] = to_numpy(crown_ret['lbias'].squeeze(1))
    #     uAs[start_idx:end_idx] = to_numpy(crown_ret['uA'].squeeze(1))
    #     ubs[start_idx:end_idx] = to_numpy(crown_ret['ubias'].squeeze(1))

    first_stage_time = time.time() - start_time
    print("First pass time: ", first_stage_time)

    # concatenate the constraints into a single tensor
    plane_constraints_lower = np.concatenate((lAs, lbs.reshape(-1, 1)), axis=-1).reshape(num_valid, 1, 4)
    plane_constraints_upper = np.concatenate((uAs, ubs.reshape(-1, 1)), axis=-1).reshape(num_valid, 1, 4)

    second_stage_time = None
    if args.alpha_pass:
        ### second, we do a pass of alpha-CROWN with one plane constraint ###

        # update the implicit function
        opt_bound_args = {
            'iteration': 20,
            'lr_alpha': 1e-1,
            'keep_best': False,
            'early_stop_patience': 1e6,
            'lr_decay': 1,
            'save_loss_graphs': True,
            'swap_loss_iter': 10
        }
        if USE_CUSTOM_LOSS_OPTION:
            opt_bound_args.update({'use_custom_loss': True, 'custom_loss_func': custom_loss_batch_estimate_volume})
        alpha_bound_params = {'optimize_bound_args': opt_bound_args}
        implicit_func.change_mode("alpha-crown", alpha_bound_params)

        for start_idx in range(0, num_valid, batch_size):
            end_idx = min(start_idx + batch_size, num_valid)
            i = start_idx // batch_size
            print(f"i: {i} | start_idx: {start_idx}, end_idx: {end_idx}, num_valid: {num_valid}")
            out_type, crown_ret = implicit_func.classify_box(params, node_lower_valid[start_idx:end_idx],
                                                             node_upper_valid[start_idx:end_idx], swap_loss=True,
                                                             use_custom_loss=USE_CUSTOM_LOSS_OPTION,
                                                             plane_constraints_lower=torch.from_numpy(
                                                                 plane_constraints_lower[start_idx:end_idx]),
                                                             plane_constraints_upper=torch.from_numpy(
                                                                 plane_constraints_upper[start_idx:end_idx]),
                                                             # plane_constraints_lower=None,
                                                             # plane_constraints_upper=None,
                                                             )
            lAs[start_idx:end_idx] = to_numpy(crown_ret['lA'].squeeze(1))
            lbs[start_idx:end_idx] = to_numpy(crown_ret['lbias'].squeeze(1))
            uAs[start_idx:end_idx] = to_numpy(crown_ret['uA'].squeeze(1))
            ubs[start_idx:end_idx] = to_numpy(crown_ret['ubias'].squeeze(1))

        new_plane_constraints_lower = np.concatenate((lAs, lbs.reshape(-1, 1)), axis=-1).reshape(num_valid, 1, 4)
        new_plane_constraints_upper = np.concatenate((uAs, ubs.reshape(-1, 1)), axis=-1).reshape(num_valid, 1, 4)

        plane_constraints_lower = np.concatenate((plane_constraints_lower, new_plane_constraints_lower), axis=1)
        plane_constraints_upper = np.concatenate((plane_constraints_upper, new_plane_constraints_upper), axis=1)

        second_stage_time = time.time() - start_time
        print("Second pass time: ", second_stage_time)

        # TODO: Potentially remove this third pass altogether. It does not seem to provide any meaningful benefit
        ### third, we do a final pass of CROWN with two plane constraints ###

        # new_bound_opt_args = {
        #         'save_loss_graphs': True,
        #         'perpendicular_multiplier': 100,
        # }
        # opt_bound_args.update(new_bound_opt_args)
        # alpha_bound_params.update({'optimize_bound_args': opt_bound_args})
        # implicit_func.change_mode("alpha-crown", alpha_bound_params)
        #
        # for start_idx in range(0, num_valid, batch_size):
        #     end_idx = min(start_idx + batch_size, num_valid)
        #     i = start_idx // batch_size
        #     print(f"i: {i} | start_idx: {start_idx}, end_idx: {end_idx}, num_valid: {num_valid}")
        #     out_type, crown_ret = implicit_func.classify_box(params, node_lower_valid[start_idx:end_idx],
        #                                                      node_upper_valid[start_idx:end_idx], swap_loss=True,
        #                                                      use_custom_loss=USE_CUSTOM_LOSS_OPTION,
        #                                                      plane_constraints_lower=torch.from_numpy(
        #                                                          plane_constraints_lower[start_idx:end_idx]),
        #                                                      plane_constraints_upper=torch.from_numpy(
        #                                                          plane_constraints_upper[start_idx:end_idx]),
        #                                                      # plane_constraints_lower=None,
        #                                                      # plane_constraints_upper=None,
        #                                                      )
        #     lAs[start_idx:end_idx] = to_numpy(crown_ret['lA'].squeeze(1))
        #     lbs[start_idx:end_idx] = to_numpy(crown_ret['lbias'].squeeze(1))
        #     uAs[start_idx:end_idx] = to_numpy(crown_ret['uA'].squeeze(1))
        #     ubs[start_idx:end_idx] = to_numpy(crown_ret['ubias'].squeeze(1))
        #
        # new_plane_constraints_lower = np.concatenate((lAs, lbs.reshape(-1, 1)), axis=-1).reshape(num_valid, 1, 4)
        # new_plane_constraints_upper = np.concatenate((uAs, ubs.reshape(-1, 1)), axis=-1).reshape(num_valid, 1, 4)
        #
        # plane_constraints_lower = np.concatenate((plane_constraints_lower, new_plane_constraints_lower), axis=1)
        # plane_constraints_upper = np.concatenate((plane_constraints_upper, new_plane_constraints_upper), axis=1)

        # third_stage_time = time.time() - start_time
        # print("Third pass time: ", third_stage_time)

    # save all bounds and node bounds to .npz to later compute the mesh of the object
    out_valid = {
        'lower': to_numpy(node_lower_valid),
        'upper': to_numpy(node_upper_valid),
        'mA': 0.5 * lAs + 0.5 * uAs,
        'mb': 0.5 * lbs + 0.5 * ubs,
        'lA': lAs,
        'lb': lbs,
        'uA': uAs,
        'ub': ubs,
        'plane_constraints_lower': plane_constraints_lower[:, 1:, :],  # the first plane constraint is lA, lb so skip
        'plane_constraints_upper': plane_constraints_upper[:, 1:, :],  # the first plane constraint is lA, lb so skip
    }
    np.savez(args.save_to, **out_valid)
    total_time = time.time() - start_time

    # Helper function to convert seconds to hours, minutes, seconds, milliseconds
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds * 1000) % 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

    # Create a table and add rows for each runtime
    table = PrettyTable()
    table.field_names = ["Stage", "Run Time (hh:mm:ss.ms)"]
    table.add_row(["Tree Building Time", format_time(first_stage_time)])
    table.add_row(["CROWN Pass", format_time(first_stage_time)])
    table.add_row(["alpha-CROWN Pass 1", format_time(second_stage_time) if second_stage_time is not None else "N/A"])
    # table.add_row(["alpha-CROWN Pass 2", format_time(third_stage_time)])
    table.add_row(["Total", format_time(total_time)])

    # Print the table
    print(table)

if __name__ == '__main__':
    main()
