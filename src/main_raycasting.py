# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
import time
import argparse
import warnings
from enum import Enum

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

class MainApplicationMethod(Enum):
    """
    1) Default manner of raycasting
    2) Runs ray-casting on the Thingi10K dataset
    3) Runs ray-casting on the Meshes Master dataset
    """
    Default = 1
    TrainThingi10K = 2
    TrainMeshesMaster = 3

    @classmethod
    def get(cls, identifier, default=None):
        # Check if the identifier is a valid name
        if isinstance(identifier, str):
            return cls.__members__.get(identifier, default)
        # Check if the identifier is a valid value
        elif isinstance(identifier, int):
            for member in cls:
                if member.value == identifier:
                    return member
            return default
        else:
            raise TypeError("Identifier must be a string (name) or an integer (value)")


def save_render_current_view(args: dict, implicit_func, params, cast_frustum, opts, matcaps, surf_color,
                             cast_tree_based=False, shell_based=False, batch_size=None, enable_clipping=False, load_from=None, save_to=None):
    # root = torch.tensor([5., 0., 0.]) #+ torch.ones(3)
    # look = torch.tensor([-1., 0., 0.])
    # up = torch.tensor([0., 1., 0.])
    # left = torch.tensor([0., 0., 1.])
    root = torch.tensor([0., -3., 0.])
    left = torch.tensor([1., 0., 0.])
    look = torch.tensor([0., 1., 0.])
    up = torch.tensor([0., 0., 1.])
    fov_deg = 30
    res = args['res'] // opts['res_scale']

    surf_color = tuple(surf_color)

    img, depth, count, _, eval_sum, raycast_time = render.render_image_naive(implicit_func, params, root, look, up,
                                                                             left, res, fov_deg, cast_frustum, opts,
                                                                             shading='matcap_color', matcaps=matcaps,
                                                                             shading_color_tuple=(surf_color,),
                                                                             tree_based=cast_tree_based, shell_based=shell_based, batch_size=batch_size,
                                                                             enable_clipping=enable_clipping, load_from=load_from, save_to=save_to)

    # flip Y
    img = torch.flip(img, [0])
    # append an alpha channel
    alpha_channel = (torch.min(img, dim=-1).values < 1.).float()
    # print(alpha_channel[:3, :3])
    # alpha_channel = torch.ones_like(img[:,:,0])
    img_alpha = torch.concatenate((img, alpha_channel[:, :, None]), dim=-1)
    img_alpha = torch.clip(img_alpha, min=0., max=1.)
    img_alpha = (img_alpha * 255.).byte()
    print(f"Saving image to {args['image_write_path']}")
    imageio.imwrite(args['image_write_path'], img_alpha.cpu().detach().numpy())

def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    # Program mode
    parser.add_argument("--program_mode", type=str, default=MainApplicationMethod.Default.name)

    # Build arguments
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--mode", type=str, default='affine_fixed')
    parser.add_argument("--cast_frustum", action='store_true')
    parser.add_argument("--cast_tree_based", action='store_true')
    parser.add_argument("--cast_shell_based", action='store_true')
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--res", type=int, default=1024)

    parser.add_argument("--image_write_path", type=str, default="render_out.png")

    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')
    parser.add_argument("--enable_clipping", action='store_true')
    parser.add_argument("--heuristic", type=str, default='naive')
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--save_to", type=str, default=None)
    # Parse arguments
    args = parser.parse_args()

    return vars(args)

def main(args: dict):

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    cast_frustum = args['cast_frustum']
    cast_tree_based = args['cast_tree_based']
    cast_shell_based = args['cast_shell_based']
    mode = args['mode']
    batch_size = args['batch_size']
    enable_clipping = args['enable_clipping']
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

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args['input'], mode=mode, **affine_opts)

    # load the matcaps
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))

    save_render_current_view(args, implicit_func, params, cast_frustum, opts, matcaps, surf_color,
                             cast_tree_based=cast_tree_based, shell_based=cast_shell_based, batch_size=batch_size, enable_clipping=enable_clipping, load_from=args['load_from'], save_to=args['save_to'])


def TrainThingi10K_main(args: dict):
    """
    Main program for training implicit surfaces on the entire Thingi10K dataset
    :param args: Default main program arguments/configurations
    :return:
    """
    input_directory = "sample_inputs/Thingi10K_inputs/"
    output_directory = "sample_outputs/Thingi10K_outputs/" + args['mode'] + '/'
    os.makedirs(output_directory, exist_ok=True)

    file_names = [f for f in os.listdir(input_directory) if f.endswith('.npz')]
    input_files = [input_directory + f for f in file_names]
    output_files = [output_directory + f.replace(".npz", ".png") for f in file_names]
    for in_file, out_file in zip(input_files, output_files):
        args.update({
            'input': in_file,
            'image_write_path': out_file,
        })
        main(args)

def TrainMeshesMaster_main(args: dict):
    """
    Main program for training implicit surfaces on the entire Meshes Master dataset
    :param args: Default main program arguments/configurations
    :return:
    """
    input_directory = "sample_inputs/meshes-master_inputs/"
    output_directory = "sample_outputs/meshes-master_outputs/" + args['mode'] + '/'
    os.makedirs(output_directory, exist_ok=True)

    file_names = [f for f in os.listdir(input_directory) if f.endswith('.npz')]
    input_files = [input_directory + f for f in file_names]
    output_files = [output_directory + f.replace(".npz", ".png") for f in file_names]
    for in_file, out_file in zip(input_files, output_files):
        args.update({
            'input': in_file,
            'image_write_path': out_file,
        })
        main(args)

if __name__ == '__main__':
    # parse user arguments
    args_dict = parse_args()
    program_mode_name = args_dict.pop('program_mode')
    program_mode = MainApplicationMethod.get(program_mode_name, None)

    # run the specified program
    if program_mode == MainApplicationMethod.Default:
        main(args_dict)
    elif program_mode == MainApplicationMethod.TrainThingi10K:
        TrainThingi10K_main(args_dict)
    elif program_mode == MainApplicationMethod.TrainMeshesMaster:
        TrainMeshesMaster_main(args_dict)
    else:
        raise ValueError(f"Invalid program_mode of {program_mode_name}")
