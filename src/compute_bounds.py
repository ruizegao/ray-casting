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
    parser.add_argument("save_to", type=str)
    parser.add_argument("--mode", type=str, default='crown')
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--split_depth", type=int, default=21)
    parser.add_argument("--batch_size", type=int, default=256)
    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    mode = args.mode
    split_depth = args.split_depth
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
    lower = torch.tensor((-data_bound, -data_bound, -data_bound))
    upper = torch.tensor((data_bound, data_bound, data_bound))
    out_dict = construct_uniform_unknown_levelset_tree(implicit_func, params, lower, upper, split_depth=split_depth, with_interior_nodes=True)
    node_valid = torch.cat((out_dict['unknown_node_valid'], out_dict['interior_node_valid']), dim=0)
    node_lower = torch.cat((out_dict['unknown_node_lower'], out_dict['interior_node_lower']), dim=0)
    node_upper = torch.cat((out_dict['unknown_node_upper'], out_dict['interior_node_upper']), dim=0)

    node_lower_valid = node_lower[node_valid]
    node_upper_valid = node_upper[node_valid]
    num_valid = node_valid.sum().item()
    mAs = torch.empty_like(node_lower_valid)
    mbs = torch.empty((num_valid,))
    lAs = torch.empty_like(node_lower_valid)
    lbs = torch.empty((num_valid,))
    uAs = torch.empty_like(node_lower_valid)
    ubs = torch.empty((num_valid,))
    for start_idx in range(0, num_valid, batch_size):
        end_idx = min(start_idx + batch_size, num_valid)
        out_type, crown_ret = implicit_func.classify_box(params, node_lower_valid[start_idx:end_idx], node_upper_valid[start_idx:end_idx])
        mAs[start_idx:end_idx] = 0.5 * (crown_ret['lA'] + crown_ret['uA']).squeeze(1)
        mbs[start_idx:end_idx] = 0.5 * (crown_ret['lbias'] + crown_ret['ubias']).squeeze(1)
        lAs[start_idx:end_idx] = crown_ret['lA'].squeeze(1)
        lbs[start_idx:end_idx] = crown_ret['lbias'].squeeze(1)
        uAs[start_idx:end_idx] = crown_ret['uA'].squeeze(1)
        ubs[start_idx:end_idx] = crown_ret['ubias'].squeeze(1)
    out_valid = {}
    out_valid['lower'] = node_lower_valid.cpu().numpy()
    out_valid['upper'] = node_upper_valid.cpu().numpy()
    out_valid['mA'] = mAs.cpu().numpy()
    out_valid['mb'] = mbs.cpu().numpy()
    out_valid['lA'] = lAs.cpu().numpy()
    out_valid['lb'] = lbs.cpu().numpy()
    out_valid['uA'] = uAs.cpu().numpy()
    out_valid['ub'] = ubs.cpu().numpy()
    np.savez(args.save_to, **out_valid)

if __name__ == '__main__':
    main()
