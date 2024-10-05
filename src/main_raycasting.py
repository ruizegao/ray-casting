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


def save_render_current_view(args, implicit_func, params, cast_frustum, opts, matcaps, surf_color,
                             cast_tree_based=False, batch_size=None, enable_clipping=False, load_from=None, save_to=None):
    root = torch.tensor([5., 0., 0.]) #+ torch.ones(3)
    look = torch.tensor([-1., 0., 0.])
    up = torch.tensor([0., 1., 0.])
    left = torch.tensor([0., 0., 1.])
    # root = torch.tensor([0., -2., 0.])
    # left = torch.tensor([1., 0., 0.])
    # look = torch.tensor([0., 1., 0.])
    # up = torch.tensor([0., 0., 1.])
    fov_deg = 30
    res = args.res // opts['res_scale']

    surf_color = tuple(surf_color)

    img, depth, count, _, eval_sum, raycast_time = render.render_image_naive(implicit_func, params, root, look, up,
                                                                             left, res, fov_deg, cast_frustum, opts,
                                                                             shading='matcap_color', matcaps=matcaps,
                                                                             shading_color_tuple=(surf_color,),
                                                                             tree_based=cast_tree_based, batch_size=batch_size,
                                                                             enable_clipping=enable_clipping, load_from=load_from, save_to=save_to)

    # flip Y
    img = torch.flip(img, [0])
    # append an alpha channel
    alpha_channel = (torch.min(img, dim=-1).values < 1.).float()
    # print(alpha_channel[:3, :3])
    # alpha_channel = torch.ones_like(img[:,:,0])
    img_alpha = torch.concatenate((img, alpha_channel[:, :, None]), dim=-1)
    img_alpha = torch.clip(img_alpha, min=0., max=1.)
    print(f"Saving image to {args.image_write_path}")
    imageio.imwrite(args.image_write_path, img_alpha.cpu().detach().numpy())


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
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--save-to", type=str, default=None)
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
    # if mode == 'affine_truncate':
    #     # truncate options
    #     implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
    #
    # elif mode == 'affine_append':
    #     # truncate options
    #     implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
    #
    # elif mode == 'sdf':
    #
    #     changed, affine_opts['sdf_lipschitz'] = psim.InputFloat("SDF Lipschitz", affine_opts['sdf_lipschitz'])
    #     if changed:
    #         implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
    #
    # elif mode in CROWN_MODES:
    #
    #     implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
    #
    # elif mode == 'affine+backward':
    #     implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)
    #
    # elif mode == 'affine_quad':
    #     implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)

    save_render_current_view(args, implicit_func, params, cast_frustum, opts, matcaps, surf_color,
                             cast_tree_based=cast_tree_based, batch_size=batch_size, enable_clipping=enable_clipping, load_from=args.load_from, save_to=args.save_to)


if __name__ == '__main__':
    main()
