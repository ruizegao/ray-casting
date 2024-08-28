# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
from functools import partial

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


def save_render_current_view(args, implicit_func, params, cast_frustum, opts, matcaps, surf_color, cast_tree_based=False):
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

    img, depth, count, _, eval_sum, raycast_time = render.render_image_naive(implicit_func, params, root, look, up, left, res, fov_deg, cast_frustum, opts, shading='matcap_color', matcaps=matcaps, shading_color_tuple=(surf_color,), tree_based=cast_tree_based)
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

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)

    parser.add_argument("--mode", type=str, default='affine_fixed')
    parser.add_argument("--cast_frustum", action='store_true')
    parser.add_argument("--cast_tree_based", action='store_true')


    parser.add_argument("--res", type=int, default=1024)
    
    parser.add_argument("--image_write_path", type=str, default="render_out.png")

    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    print("args parsed!!!")

    # GUI Parameters
    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    cast_frustum = args.cast_frustum
    cast_tree_based = args.cast_tree_based
    mode = args.mode
    modes = ['sdf', 'interval', 'affine_fixed', 'affine_truncate', 'affine_append', 'affine_all', 'slope_interval', 'crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward', 'dynamic_forward+backward', 'affine+backward']
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 8
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    affine_opts['crown'] = 1.
    affine_opts['alpha_crown'] = 1.
    affine_opts['forward+backward'] = 1.
    affine_opts['forward'] = 1.
    affine_opts['forward-optimized'] = 1.
    affine_opts['dynamic_forward'] = 1.
    affine_opts['dynamic_forward+backward'] = 1.
    affine_opts['affine+backward'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = 'absolute'
    surf_color = (0.157,0.613,1.000)

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    # load the matcaps
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))
    print("matcaps loaded!!!")
    print("mode is: ", mode)
    if mode == 'affine_truncate':
        # truncate options

        changed, affine_opts['affine_n_truncate'] = psim.InputInt("affine_n_truncate", affine_opts['affine_n_truncate'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

        changed, affine_opts['affine_truncate_policy'] = utils.combo_string_picker("Method", affine_opts['affine_truncate_policy'], truncate_policies)
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode == 'affine_append':
        # truncate options

        changed, affine_opts['affine_n_append'] = psim.InputInt("affine_n_append", affine_opts['affine_n_append'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode == 'sdf':

        changed, affine_opts['sdf_lipschitz'] = psim.InputFloat("SDF Lipschitz", affine_opts['sdf_lipschitz'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode == 'crown':

        changed, affine_opts['crown'] = psim.InputFloat("crown", affine_opts['crown'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode == 'alpha_crown':

        changed, affine_opts['alpha_crown'] = psim.InputFloat("alpha_crown", affine_opts['alpha_crown'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode == 'forward+backward':

        changed, affine_opts['forward+backward'] = psim.InputFloat("forward+backward", affine_opts['forward+backward'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)

    elif mode == 'forward':

        changed, affine_opts['forward'] = psim.InputFloat("forward", affine_opts['forward'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)

    elif mode == 'forward-optimized':

        changed, affine_opts['forward-optimized'] = psim.InputFloat("forward-optimized", affine_opts['forward-optimized'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)

    elif mode == 'dynamic_forward':

        changed, affine_opts['dynamic_forward'] = psim.InputFloat("dynamic_forward", affine_opts['dynamic_forward'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)

    elif mode == 'dynamic_forward+backward':

        changed, affine_opts['dynamic_forward+backward'] = psim.InputFloat("dynamic_forward+backward", affine_opts['dynamic_forward+backward'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)

    elif mode == 'affine+backward':
        changed, affine_opts['affine+backward'] = psim.InputFloat("affine+backward", affine_opts['affine+backward'])
        if changed:
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode)


    save_render_current_view(args, implicit_func, params, cast_frustum, opts, matcaps, surf_color, cast_tree_based=cast_tree_based)


if __name__ == '__main__':
    main()
