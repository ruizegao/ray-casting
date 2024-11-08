# import igl # work around some env/packaging problems by loading this first

# import sys, os, time, math
# os.environ['OptiX_INSTALL_DIR'] = '/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'

import time
import argparse
import warnings

import torch
import os

# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
import matplotlib as plt
import imageio

# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CROWN_MODES = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def save_render_current_view(args, implicit_func, params, opts, matcaps, load_from):
    root = torch.tensor([5., 0., 0.]) #+ torch.ones(3)
    look = torch.tensor([-1., 0., 0.])
    up = torch.tensor([0., 1., 0.])
    left = torch.tensor([0., 0., 1.])
    # root = torch.tensor([0., -3., 0.])
    # left = torch.tensor([1., 0., 0.])
    # look = torch.tensor([0., 1., 0.])
    # up = torch.tensor([0., 0., 1.])
    fov_deg = 30
    res = args.res // opts['res_scale']


    img = render.render_image_mesh(implicit_func, params, load_from, root, look, up, left, res, fov_deg, opts,
                                   shading='matcap_color', matcaps=matcaps)
    img = torch.flip(img, [0])
    # append an alpha channel
    alpha_channel = (torch.min(img, dim=-1).values < 1.).float()
    img_alpha = torch.concatenate((img, alpha_channel[:, :, None]), dim=-1)
    img_alpha = torch.clip(img_alpha, min=0., max=1.)
    img_alpha = (img_alpha * 255.).byte()
    print(f"Saving image to {args.image_write_path}")
    imageio.imwrite(args.image_write_path, img_alpha.cpu().detach().numpy())

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("load_from", type=str)
    parser.add_argument("--mode", type=str, default='crown')
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--image_write_path", type=str, default="render_out.png")
    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=args.mode, **{})

    # load the matcaps
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))

    save_render_current_view(args, implicit_func, params, opts, matcaps, load_from=args.load_from)

if __name__ == '__main__':
    main()
