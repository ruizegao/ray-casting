# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
import time
import argparse
import warnings

import torch
import imageio
import polyscope.imgui as psim

# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils

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
         [-0.2500,  0.7500,  0.2500],
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
         [-0.2500,  0.7500,  0.2500],
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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    # for l, u in zip(lower, upper):
    #     _ = implicit_func.classify_box(params, l.unsqueeze(0), u.unsqueeze(0))
    #     # _ = vmap(partial(implicit_func.classify_box, params))(l.unsqueeze(0), u.unsqueeze(0))
    #     _ = partial(implicit_func.classify_box, params)(l, u)

    _ = implicit_func.classify_box(params, lower, upper)
    # _ = vmap(partial(implicit_func.classify_box, params))(lower, upper)

    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

if __name__ == '__main__':
    main()
