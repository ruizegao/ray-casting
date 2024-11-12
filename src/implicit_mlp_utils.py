from functools import partial

from numpy.core.defchararray import rindex

import utils
import mlp, sdf, affine, slope_interval, crown
import torch
from main_fit_implicit_torch import *
from siren_pytorch import SirenNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_implicit_from_file(input_path, mode, shift=None, **kwargs):

    obj_name = ''
    ## Load the file
    if input_path.endswith(".npz"):
        params = mlp.load(input_path, shift=shift)
        obj_name = input_path[input_path.rindex('/')+1:-4]
    elif input_path.endswith(".pth"):
        # params = load_net_object(input_path)
        params = SirenNet(
            dim_in=3,
            dim_hidden=256,
            dim_out=1,
            num_layers=5,
            final_activation=nn.Identity(),
        ).to(device)
        # print(params)
        state_dict = torch.load(input_path, weights_only=True, map_location=torch.device(device))
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        params.load_state_dict(new_state_dict)
        # params.eval()
    else:
        raise ValueError("unrecognized filetype")

    enable_clipping = kwargs.pop('enable_clipping', False)

    # `params` is now populated

    # Construct an `ImplicitFunction` object ready to do the appropriate kind of evaluation
    if mode == 'sdf':
        implicit_func = mlp.func_from_spec(mode='default')
        if 'sdf_lipschitz' in kwargs:
            lipschitz_bound = kwargs['sdf_lipschitz']
        else:
            lipschitz_bound = 1.
        return sdf.WeakSDFImplicitFunction(implicit_func, lipschitz_bound=lipschitz_bound), params
    
    elif mode == 'interval':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('interval')
        torch_model = mlp.func_as_torch(params)
        return affine.AffineImplicitFunction(implicit_func, affine_ctx, torch_model), params
    
    elif mode == 'affine_fixed':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine_fixed')
        torch_model = mlp.func_as_torch(params)
        return affine.AffineImplicitFunction(implicit_func, affine_ctx, torch_model, obj_name=obj_name), params
   
    elif mode == 'affine_truncate':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine_truncate', 
                truncate_count=kwargs['affine_n_truncate'], truncate_policy=kwargs['affine_truncate_policy'])
        torch_model = mlp.func_as_torch(params)
        return affine.AffineImplicitFunction(implicit_func, affine_ctx, torch_model=torch_model, obj_name=obj_name), params
    
    elif mode == 'affine_append':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine_append', 
                n_append=kwargs['affine_n_append'])
        torch_model = mlp.func_as_torch(params)
        return affine.AffineImplicitFunction(implicit_func, affine_ctx, torch_model=torch_model, obj_name=obj_name), params
    
    elif mode == 'affine_all':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine_all')
        torch_model = mlp.func_as_torch(params)
        return affine.AffineImplicitFunction(implicit_func, affine_ctx, torch_model=torch_model, obj_name=obj_name), params

    elif mode == 'affine_quad':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine_quad')
        torch_model = mlp.func_as_torch(params)
        # data_bound = 100.
        # lower = torch.tensor((0.25, -0.6875, 0.46875), dtype=torch.float32)
        # upper = torch.tensor((0.28125, -0.6875, 0.46875))
        # upper = torch.tensor((0.28125, -0.65625, 0.5), dtype=torch.float32)
        # samples = (torch.rand((10000, 3))) * 0.03125 + lower
        # outputs = torch_model(samples)
        # print("min and max of sampled values: ")
        # print(outputs.min(), outputs.max())
        func = affine.AffineImplicitFunction(implicit_func, affine_ctx, torch_model)
        # print(func.classify_box(params, lower, upper))
        return affine.AffineImplicitFunction(implicit_func, affine_ctx, torch_model=torch_model, obj_name=obj_name), params

    elif mode == 'slope_interval':
        implicit_func = mlp.func_from_spec(mode='slope_interval')
        return slope_interval.SlopeIntervalImplicitFunction(implicit_func), params

    elif mode == 'crown':
        if input_path.endswith('pth'):
            crown_func = params
        else:
            crown_func = mlp.func_as_torch(params)
        # data_bound = 1.ren
        # lower = torch.tensor((0.25, -0.6875, 0.46875), dtype=torch.float32)
        # upper = torch.tensor((0.28125, -0.6875, 0.46875))
        # upper = torch.tensor((0.28125, -0.65625, 0.5), dtype=torch.float32)
        # samples = (torch.rand((10000, 3))) * 0.03125 + lower
        # outputs = crown_func(samples)
        # print("min and max of sampled values: ")
        # print(outputs.min(), outputs.max())
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='crown', enable_clipping=enable_clipping, obj_name=obj_name), params

    elif mode == 'alpha_crown':
        crown_func = mlp.func_as_torch(params)
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='alpha-CROWN', enable_clipping=enable_clipping, obj_name=obj_name), params

    elif mode == 'forward+backward':
        crown_func = mlp.func_as_torch(params)
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='Forward+Backward', enable_clipping=enable_clipping, obj_name=obj_name), params

    elif mode == 'forward':
        crown_func = mlp.func_as_torch(params)
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='forward', enable_clipping=enable_clipping, obj_name=obj_name), params

    elif mode == 'forward-optimized':
            crown_func = mlp.func_as_torch(params)
            implicit_func = mlp.func_from_spec(mode='default')
            return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='forward-optimized', enable_clipping=enable_clipping, obj_name=obj_name), params

    elif mode == 'dynamic_forward':
        crown_func = mlp.func_as_torch(params)
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='dynamic-forward', enable_clipping=enable_clipping, obj_name=obj_name), params

    elif mode == 'dynamic_forward+backward':
            crown_func = mlp.func_as_torch(params)
            implicit_func = mlp.func_from_spec(mode='default')
            return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='dynamic-forward+backward', enable_clipping=enable_clipping, obj_name=obj_name), params

    elif mode == 'affine+backward':
        implicit_func = mlp.func_from_spec(mode='affine')
        bounded_func = mlp.bounded_func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine+backward')
        return affine.AffineImplicitFunction(implicit_func, affine_ctx, bounded_func, obj_name=obj_name), params

    else:
        raise RuntimeError("unrecognized mode")
