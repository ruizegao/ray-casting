from functools import partial


import utils
import mlp, sdf, affine, slope_interval, crown


def generate_implicit_from_file(input_path, mode, **kwargs):
    
    ## Load the file
    if input_path.endswith(".npz"):
        params = mlp.load(input_path)
    else:
        raise ValueError("unrecognized filetype")

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
        return affine.AffineImplicitFunction(implicit_func, affine_ctx), params
    
    elif mode == 'affine_fixed':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine_fixed')
        return affine.AffineImplicitFunction(implicit_func, affine_ctx), params
   
    elif mode == 'affine_truncate':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine_truncate', 
                truncate_count=kwargs['affine_n_truncate'], truncate_policy=kwargs['affine_truncate_policy'])
        return affine.AffineImplicitFunction(implicit_func, affine_ctx), params
    
    elif mode == 'affine_append':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine_append', 
                n_append=kwargs['affine_n_append'])
        return affine.AffineImplicitFunction(implicit_func, affine_ctx), params
    
    elif mode == 'affine_all':
        implicit_func = mlp.func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine_all')
        return affine.AffineImplicitFunction(implicit_func, affine_ctx), params

    elif mode == 'slope_interval':
        implicit_func = mlp.func_from_spec(mode='slope_interval')
        return slope_interval.SlopeIntervalImplicitFunction(implicit_func), params

    elif mode == 'crown':
        crown_func = mlp.func_as_torch(params)
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='crown'), params

    elif mode == 'alpha_crown':
        crown_func = mlp.func_as_torch(params)
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='alpha-CROWN'), params

    elif mode == 'forward+backward':
        crown_func = mlp.func_as_torch(params)
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='Forward+Backward'), params

    elif mode == 'forward':
        crown_func = mlp.func_as_torch(params)
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='forward'), params

    elif mode == 'forward-optimized':
            crown_func = mlp.func_as_torch(params)
            implicit_func = mlp.func_from_spec(mode='default')
            return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='forward-optimized'), params

    elif mode == 'dynamic_forward':
        crown_func = mlp.func_as_torch(params)
        implicit_func = mlp.func_from_spec(mode='default')
        return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='dynamic-forward'), params

    elif mode == 'dynamic_forward+backward':
            crown_func = mlp.func_as_torch(params)
            implicit_func = mlp.func_from_spec(mode='default')
            return crown.CrownImplicitFunction(implicit_func, crown_func, crown_mode='dynamic-forward+backward'), params

    elif mode == 'affine+backward':
        implicit_func = mlp.func_from_spec(mode='affine')
        bounded_func = mlp.bounded_func_from_spec(mode='affine')
        affine_ctx = affine.AffineContext('affine+backward')
        return affine.AffineImplicitFunction(implicit_func, affine_ctx, bounded_func), params

    else:
        raise RuntimeError("unrecognized mode")
