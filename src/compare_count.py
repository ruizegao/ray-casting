# import igl # work around some env/packaging problems by loading this first

# import sys, os, time, math
# os.environ['OptiX_INSTALL_DIR'] = '/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'

import time
import gc
import argparse
import warnings

import torch
import os

# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm, SymLogNorm
import imageio
# import jax.numpy as jnp
import trimesh
from PIL import Image
from matplotlib.ticker import MaxNLocator

os.environ['OptiX_INSTALL_DIR'] = '/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'
# os.environ['OptiX_INSTALL_DIR'] = '/media/gaorz/b5df3483-c11a-42f1-b414-023f33bc5312/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'

from triro.ray.ray_optix import RayMeshIntersector  # FIXME: Should be uncommented when rendering meshes


# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def get_count(args, implicit_func, params, load_from, opts, matcaps):
    # root = torch.tensor([-2.5, 0., 0.]) #+ torch.ones(3)
    # look = torch.tensor([1., 0., 0.])
    # up = torch.tensor([0., 1., 0.])
    # left = torch.tensor([0., 0., 1.])
    #
    # root = torch.tensor([0., -1.5, 0.])
    # left = torch.tensor([1., 0., 0.])
    # look = torch.tensor([0.4, 1., 0.5])
    # up = torch.tensor([0., 0., 1.])

    root = torch.tensor([0., -3.5, 0.])
    left = torch.tensor([1., 0., 0.])
    look = torch.tensor([0., 1., 0.])
    up = torch.tensor([0., 0., 1.])

    # root = torch.tensor([2.5, 0., 2.5])
    # up = torch.tensor([0., 1., 0.])
    # look = torch.tensor([-1., 0., -1.])
    # left = torch.tensor([1., 0., 0.])
    #
    root = torch.tensor([0., 0., 3.5])
    up = torch.tensor([0., 1., 0.])
    look = torch.tensor([0., 0., -1.])
    left = torch.tensor([1., 0., 0.])

    fov_deg = 30
    res = args.res // opts['res_scale']

    mesh = trimesh.load(load_from)

    intersector = RayMeshIntersector(mesh)

    img, rendering_time, counts = render.render_image_mesh(implicit_func, params, intersector, root, look,
                                                   up, left, res,
                                                   fov_deg, opts,
                                                   shading='matcap_color', matcaps=matcaps, approx=False,
                                                   shading_color_tuple=torch.tensor(((0., 0.5, 0.),)))

    del intersector
    del mesh
    gc.collect()
    torch.cuda.empty_cache()
    return rendering_time, counts.detach().cpu().numpy().reshape(res, res)

def get_count_baseline(args, implicit_func, params, opts, matcaps):
    # root = torch.tensor([0., -3.5, 0.])
    # left = torch.tensor([1., 0., 0.])
    # look = torch.tensor([0., 1., 0.])
    # up = torch.tensor([0., 0., 1.])

    root = torch.tensor([0., 0., 3.5])
    up = torch.tensor([0., 1., 0.])
    look = torch.tensor([0., 0., -1.])
    left = torch.tensor([1., 0., 0.])
    fov_deg = 30
    res = args.res // opts['res_scale']
    img, depth, count, _, eval_sum, raycast_time = render.render_image_naive(implicit_func, params, root, look, up,
                                                                             left, res, fov_deg, False, opts,
                                                                             shading='matcap_color', matcaps=matcaps)

    return raycast_time, count.detach().cpu().numpy().reshape(res, res)

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("load_from_1", type=str)
    parser.add_argument("load_from_2", type=str)
    parser.add_argument("--rendering", type=str)
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    opts['hit_eps'] = 1e-3

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode='crown', **{})

    # load the matcaps
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))
    matcaps = torch.stack(matcaps)

    t1, img1 = get_count(args, implicit_func, params, args.load_from_1, opts, matcaps)
    t2, img2 = get_count(args, implicit_func, params, args.load_from_2, opts, matcaps)

    del implicit_func, params
    gc.collect()
    torch.cuda.empty_cache()
    # norm = PowerNorm(gamma=0.05, vmin=0., vmax=max(img1.max(), img2.max()))
    # norm = LogNorm(vmin=1e-3, vmax=max(img1.max(), img2.max()))
    norm = SymLogNorm(vmin=0., vmax=max(img1.max(), img2.max()), linthresh=1)

    if args.rendering:
        raise NotImplementedError("Rendering image support has been removed as per updated requirements.")
    else:
        cmap = 'hot_r'

        # Figure for adaptive shells
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        im1 = ax1.imshow(img1, cmap=cmap, norm=norm, origin='lower')
        ax1.set_xticks([])
        ax1.set_yticks([])
        for spine in ax1.spines.values():
            spine.set_visible(False)
        fig1.tight_layout()
        # fig1.savefig(f"{args.output}_adaptive.png") if args.output else plt.show()
        fig1.savefig('/media/gaorz/b5df3483-c11a-42f1-b414-023f33bc5312/home/ruize/3d_vnn_ref/de_sample_counts.pdf', bbox_inches='tight')


        # Figure for our shells
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        im2 = ax2.imshow(img2, cmap=cmap, norm=norm, origin='lower')
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_visible(False)

        cbar = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, shrink=0.8)
        # cbar.locator = MaxNLocator(nbins=5)
        # cbar.update_ticks()
        # cbar.ax.tick_params(labelsize=10)

        # for label in cbar.ax.get_yticklabels():
        #     label.set_fontname("serif")
        # cbar.set_label('')  # remove label

        # cbar = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, orientation='vertical')
        # cbar.set_label('steps', labelpad=10, fontsize=12, fontname='serif', loc='center')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.tick_top()
        cbar.set_ticks([cbar.vmin, cbar.vmax])
        cbar.ax.set_yticklabels([f'{int(cbar.vmin)}', f'{int(cbar.vmax)}'], fontname='serif', fontsize=36)

        fig2.tight_layout()
        fig2.savefig('/media/gaorz/b5df3483-c11a-42f1-b414-023f33bc5312/home/ruize/3d_vnn_ref/gios_sample_counts.pdf', bbox_inches='tight')
        plt.show()
        # fig2.savefig(f"{args.output}_ours.png", bbox_inches='tight') if args.output else plt.show()


if __name__ == '__main__':
    main()
