from functools import partial

import torch
from functorch import vmap
import affine
import mlp
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dense(input, A, b, ctx):
    A_tensor = torch.tensor(A, dtype=input[0].dtype, device=input[0].device)
    if(affine.is_const(input)):
        out = torch.matmul(input[0], A_tensor)
        if b is not None:
            b = torch.tensor(b, dtype=input[0].dtype, device=input[0].device)
            out += b
        return out, None, None


    base, aff, err = input

    def dot(x, with_abs=False):
        if with_abs:
            return torch.matmul(x, torch.abs(A_tensor))
        return torch.matmul(x, A_tensor)


    base = dot(base)
    aff = vmap(dot)(aff)
    err = dot(err, with_abs=True)

    if b is not None:
        b = torch.tensor(b, dtype=input[0].dtype, device=input[0].device)
        base += b
    return base, aff, err
mlp.apply_func['affine']['dense'] = dense

def relu(input, ctx):
    # Chebyshev bound
    base, aff, err = input

    if affine.is_const(input):
        return torch.nn.functional.relu(base), aff, err

    lower, upper = affine.may_contain_bounds(ctx, input)

    # Compute the linearized approximation
    alpha = (torch.nn.functional.relu(upper) - torch.nn.functional.relu(lower)) / (upper - lower)
    alpha = torch.where(lower >= 0, 1., alpha)
    alpha = torch.where(upper < 0, 0., alpha)
    # handle numerical badness in the denominator above
    alpha = torch.nan_to_num(alpha, nan=0.0) # necessary?
    alpha = torch.clip(alpha, min=0., max=1.)

    # here, alpha/beta are necessarily positive, which makes this simpler
    beta = (torch.nn.functional.relu(lower) - alpha * lower) / 2
    delta = beta
    output = affine.apply_linear_approx(ctx, input, alpha, beta, delta)

    return output
mlp.apply_func['affine']['relu'] = relu

def elu(input, ctx, bounded=False):
    # Chebyshev bound
    # Confusingly, elu has a parameter typically called 'alpha', and we also use 'alpha' for our linearizaiton notation. Here we simply ignore and do not support elu's alpha.
    base, aff, err = input

    if affine.is_const(input):
        return torch.nn.functional.elu(base), aff, err

    lower, upper = affine.may_contain_bounds(ctx, input)

    # Compute the linearized approximation
    lowerF = torch.nn.functional.elu(lower)
    upperF = torch.nn.functional.elu(upper)
    # lowerS = torch.where(lower < 0, lowerF + 1., 1.)
    # upperS = torch.where(upper < 0, upperF + 1., 1.)
    lowerS = torch.minimum(torch.exp(lower), torch.as_tensor(1.)) # more numerically stable than ^^^, but costs exp()
    upperS = torch.minimum(torch.exp(upper), torch.as_tensor(1.))

    alpha = (upperF - lowerF) / (upper - lower)
    alpha = torch.where(lower >= 0, 1., alpha)
    # handle numerical badness in the denominator above
    alpha = torch.nan_to_num(alpha, nan=0.0) # necessary?
    alpha = torch.clip(alpha, min=lowerS, max=upperS)

    # the alpha tangent point necessarily occurs in the <= 0. part of the function
    r_upper = (lowerF - alpha * lower)
    x_lower = torch.clip(torch.log(alpha), min=lower, max=upper)
    r_lower = (alpha-1.) - alpha * x_lower
    beta = 0.5 * (r_upper + r_lower)
    # delta = r_upper - beta
    delta = 0.5 * torch.abs(r_upper - r_lower) # this is very defensive, to ensure delta>=0

    # in strictly > 0 case, just directly set the result
    alpha = torch.where(lower >= 0, 1., alpha)
    beta = torch.where(lower >= 0, 0., beta)
    delta = torch.where(lower >= 0, 0., delta)

    output = affine.apply_linear_approx(ctx, input, alpha, beta, delta)
    return output
mlp.apply_func['affine']['elu'] = elu

def sin(input, ctx):
    # not-quite Chebyshev bound
    base, aff, err = input
    pi = torch.pi

    if affine.is_const(input):
        return torch.sin(base), aff, err

    lower, upper = affine.may_contain_bounds(ctx, input)

    slope_lower, slope_upper = utils.cos_bound(lower, upper)
    alpha = 0.5 * (slope_lower + slope_upper) # this is NOT the Chebyshev value, but seems reasonable
    alpha = torch.clip(alpha, min=-1., max=1.) # (should already be there, this is for numerics only)

    # We want to find the minima/maxima of (sin(x) - alpha*x) on [lower, upper] to compute our 
    # beta and delta. In addition to the endpoints, some calc show there can be interior 
    # extrema at +-arccos(alpha) + 2kpi for some integer k.
    # The extrema will 
    intA = torch.arccos(alpha)
    intB = -intA

    # The the first and last occurence of a value which repeats mod 2pi on the domain [lower, upper]
    # (these give the only possible locations for our extrema)
    def first(x): return 2.*pi*torch.ceil((lower + x) / (2.*pi)) - x
    def last(x): return 2.*pi*torch.floor((upper - x) / (2.*pi)) + x

    extrema_locs = [lower, upper, first(intA), last(intA), first(intB), last(intB)]
    extrema_locs = [torch.clip(x, min=lower, max=upper) for x in extrema_locs]
    extrema_vals = [torch.sin(x) - alpha * x for x in extrema_locs]

    r_lower = utils.minimum_all(extrema_vals)
    r_upper = utils.maximum_all(extrema_vals)

    beta = 0.5 * (r_upper + r_lower)
    delta = r_upper - beta

    output = affine.apply_linear_approx(ctx, input, alpha, beta, delta)
    return output
mlp.apply_func['affine']['sin'] = sin

def pow2_frequency_encode(input, ctx, coefs, shift=None):
    base, aff, err = input

    # TODO debug
    if len(base.shape) > 1:
        raise ValueError("big base")

    # expand the length-d inputs to a lenght-d*c vector
    def s(with_shift, x): 
        out = (x[:,None] * coefs[None,:])
        if with_shift and shift is not None:
            out += shift
        return out.flatten()

    base = s(True, base)
    if affine.is_const(input):
        return base, None, None

    aff = vmap(partial(s, False))(aff)
    err = s(False, err)
    
    return base, aff, err
mlp.apply_func['affine']['pow2_frequency_encode'] = pow2_frequency_encode

def squeeze_last(input, ctx):
    base, aff, err = input
    s = lambda x : torch.squeeze(x, dim=0)
    base = s(base)
    if affine.is_const(input):
        return base, None, None
    aff = vmap(s)(aff)
    err = s(err)
    return base, aff, err
mlp.apply_func['affine']['squeeze_last'] = squeeze_last

def spatial_transformation(input, R, t, ctx):
    # if the shape transforms by R,t, input points need the opposite transform
    R_inv = torch.linalg.inv(torch.from_numpy(R))
    t_inv = torch.matmul(R_inv, -torch.from_numpy(t))
    return dense(input, A=R_inv, b=t_inv, ctx=ctx)
mlp.apply_func['affine']['spatial_transformation'] = spatial_transformation
