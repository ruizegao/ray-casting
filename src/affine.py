from functools import partial
import dataclasses 
from dataclasses import dataclass
import torch.nn as nn
from functorch import grad
import torch
import utils

import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE

# === Function wrappers
torch.set_default_tensor_type(torch.cuda.FloatTensor)

class AffineImplicitFunction(implicit_function.ImplicitFunction):

    def __init__(self, affine_func, ctx, bounded_func=None):
        super().__init__("classify-only")
        self.affine_func = affine_func
        self.ctx = ctx
        self.bounded_func = bounded_func
        self.mode_dict = {'ctx' : self.ctx}


    def __call__(self, params, x):
        f = lambda x : self.affine_func(params, x, self.mode_dict)
        return wrap_scalar(f)(x)

    # the parent class automatically delegates to this
    # def classify_box(self, params, box_lower, box_upper):
        # pass
        
    def classify_general_box(self, params, box_center, box_vecs, offset=0.):
        d = box_center.shape[-1]
        v = box_vecs.shape[-2]
        # assert box_center.shape == (d,), "bad box_vecs shape"
        # assert box_vecs.shape == (v,d), "bad box_vecs shape"
        keep_ctx = dataclasses.replace(self.ctx, affine_domain_terms=v)
        # evaluate the function
        input = coordinates_in_general_box(keep_ctx, box_center, box_vecs)
        if keep_ctx.mode == 'affine+backward':
            bound_dict = self.bounded_func(params, input, {'ctx' : keep_ctx})
            box_diff = torch.diag(box_vecs)
            may_lower, may_upper = pseudo_crown(bound_dict, box_center, box_diff)
            # print(may_upper, may_lower)
        else:
            output = self.affine_func(params, input, {'ctx' : keep_ctx})
            # compute relevant bounds
            may_lower, may_upper = may_contain_bounds(keep_ctx, output)
        output_type = torch.full_like(may_lower, SIGN_UNKNOWN)
        output_type = output_type.where(may_lower <= offset, torch.full_like(may_lower, SIGN_POSITIVE))
        output_type = output_type.where(may_upper >= -offset, torch.full_like(may_lower, SIGN_NEGATIVE))
        return output_type


# === Affine utilities

# We represent affine data as a tuple input=(base,aff,err). Base is a normal shape (d,) primal vector value, affine is a (v,d) array of affine coefficients (may be v=0), err is a centered interval error shape (d,), which must be nonnegative.
# For constant values, aff == err == None. If is_const(input) == False, then it is guaranteed that aff and err are non-None.

@dataclass(frozen=True)
class AffineContext():
    mode: str = 'affine_fixed'
    truncate_count: int = -777
    truncate_policy: str = 'absolute'
    affine_domain_terms: int = 0
    n_append: int = 0

    def __post_init__(self):
        if self.mode not in ['interval', 'affine_fixed', 'affine_truncate', 'affine_append', 'affine_all', 'affine+backward', 'affine_quad']:
            raise ValueError("invalid mode")

        if self.mode == 'affine_truncate':
            if self.truncate_count is None:
                raise ValueError("must specify truncate count")

def is_const(input):
    base, aff, err = input
    if err is not None: return False
    return aff is None or aff.shape[0] == 0


# Compute the 'radius' (width of the approximation)
def radius(input):
    if is_const(input): return 0.
    base, aff, err = input
    rad = torch.sum(torch.abs(aff), dim=0)
    if err is not None:
        rad += err
    return rad

# Constuct affine inputs for the coordinates in k-dimensional box
# lower,upper should be vectors of length-k
def coordinates_in_box(ctx, lower, upper):
    center = 0.5 * (lower+upper)
    vec = upper - center
    axis_vecs = torch.diag(vec)
    return coordinates_in_general_box(ctx, center, axis_vecs)

# Constuct affine inputs for the coordinates in k-dimensional box,
# which is not necessarily axis-aligned
#  - center is the center of the box
#  - vecs is a (V,D) array of vectors which point from the center of the box to its
#    edges. These will correspond to each of the affine symbols, with the direction
#    of the vector becoming the positive orientaiton for the symbol.
# (this function is nearly a no-op, but giving it this name makes it easier to
#  reason about)
def coordinates_in_general_box(ctx, center, vecs):
    base = center
    if ctx.mode == 'interval':
        aff = torch.zeros((0,center.shape[-1]))
        err = torch.sum(torch.abs(vecs), dim=0)
    else:
        aff = vecs
        err = torch.zeros_like(center)
    return base, aff, err

def may_contain_bounds(ctx, input):
    '''
    An interval range of values that `input` _may_ take along the domain
    '''
    base, aff, err = input
    rad = radius(input)
    return base-rad, base+rad


# def compute_bounds(bounds, center, diff, optimizable_parameters=None):
#     N_ops = len(bounds)
#     A_l = bounds[N_ops - 1][
#         'A_l']  # Initialize the linear bound coefficient with the parameters of the last linear layer
#     A_u = bounds[N_ops - 1]['A_u']
#     d_l = bounds[N_ops - 1]['b_l']
#     d_u = bounds[N_ops - 1]['b_u']
#     k = 0
#     for i in range(len(bounds) - 2, -1,
#                    -1):  # Perform propagation in the backward direction to update the coefficients
#         # print("layer: ", i)
#         W_l = bounds[i]['A_l']
#         if optimizable_parameters is None:
#             W_u = bounds[i]['A_u']
#         else:
#             W_u = optimizable_parameters[k]
#             k = k + 1
#         b_l = bounds[i]['b_l']
#         b_u = bounds[i]['b_u']
#         if bounds[i][
#             'name'] == 'relu':  # Update approximated linear bounds of relu layers. The details of approximation is in relu() in affine_layers.py
#             d_l = d_l + torch.where(A_l > 0, A_l, 0.) @ b_l + torch.where(A_l <= 0, A_l, 0.) @ b_u
#             d_u = d_u + torch.where(A_u > 0, A_u, 0.) @ b_u + torch.where(A_u <= 0, A_u, 0.) @ b_l
#             A_l = torch.where(A_l > 0, A_l, 0.) @ W_l + torch.where(A_l <= 0, A_l, 0.) @ W_u
#             A_u = torch.where(A_u > 0, A_u, 0.) @ W_u + torch.where(A_u <= 0, A_u, 0.) @ W_l
#         elif bounds[i]['name'] == 'dense':  # Update linear bounds from linear layers
#             d_l = d_l + A_l @ b_l
#             d_u = d_u + A_u @ b_u
#             A_l = A_l @ W_l
#             A_u = A_u @ W_u
#
#     lower_bound = center @ A_l.T - diff @ A_l.abs().T + d_l  # Concretize the linear bound with input bounds
#     upper_bound = center @ A_u.T + diff @ A_u.abs().T + d_u
#
#     return lower_bound.sum(), upper_bound.sum()  # .sum() is used as a replacement for .item(), which is not supported by vmap()

def pseudo_crown(bounds, center, diff, optimize=False, num_iter=15, lr=0.0001):
    def compute_bounds():
        N_ops = len(bounds)
        A_l = bounds[N_ops - 1][
            'A_l']  # Initialize the linear bound coefficient with the parameters of the last linear layer
        A_u = bounds[N_ops - 1]['A_u']
        d_l = bounds[N_ops - 1]['b_l']
        d_u = bounds[N_ops - 1]['b_u']

        for i in range(len(bounds) - 2, -1,
                       -1):  # Perform propagation in the backward direction to update the coefficients
            # print("layer: ", i)
            W_l = bounds[i]['A_l']
            W_u = bounds[i]['A_u']
            b_l = bounds[i]['b_l']
            b_u = bounds[i]['b_u']
            if bounds[i][
                'name'] == 'relu':  # Update approximated linear bounds of relu layers. The details of approximation is in relu() in affine_layers.py
                d_l = d_l + torch.where(A_l > 0, A_l, 0.) @ b_l + torch.where(A_l <= 0, A_l, 0.) @ b_u
                d_u = d_u + torch.where(A_u > 0, A_u, 0.) @ b_u + torch.where(A_u <= 0, A_u, 0.) @ b_l
                A_l = torch.where(A_l > 0, A_l, 0.) @ W_l + torch.where(A_l <= 0, A_l, 0.) @ W_u
                A_u = torch.where(A_u > 0, A_u, 0.) @ W_u + torch.where(A_u <= 0, A_u, 0.) @ W_l
            elif bounds[i]['name'] == 'dense':  # Update linear bounds from linear layers
                d_l = d_l + A_l @ b_l
                d_u = d_u + A_u @ b_u
                A_l = A_l @ W_l
                A_u = A_u @ W_u

        lower_bound = center @ A_l.T - diff @ A_l.abs().T + d_l  # Concretize the linear bound with input bounds
        upper_bound = center @ A_u.T + diff @ A_u.abs().T + d_u

        return lower_bound.sum(), upper_bound.sum()  # .sum() is used as a replacement for .item(), which is not supported by vmap()

    def compute_loss(optimizable_parameters=None):
        N_ops = len(bounds)
        A_l = bounds[N_ops - 1][
            'A_l']  # Initialize the linear bound coefficient with the parameters of the last linear layer
        A_u = bounds[N_ops - 1]['A_u']
        d_l = bounds[N_ops - 1]['b_l']
        d_u = bounds[N_ops - 1]['b_u']
        k = 0
        if optimizable_parameters is not None:
            k = len(optimizable_parameters) - 1
        for i in range(len(bounds) - 2, -1,
                       -1):  # Perform propagation in the backward direction to update the coefficients
            if bounds[i]['name'] == 'relu' and optimizable_parameters is not None:
                W_l = optimizable_parameters[k]
                k = k - 1
            else:
                W_l = bounds[i]['A_l']
            # W_l = bounds[i]['A_l']
            W_u = bounds[i]['A_u']
            b_l = bounds[i]['b_l']
            b_u = bounds[i]['b_u']
            if bounds[i][
                'name'] == 'relu':  # Update approximated linear bounds of relu layers. The details of approximation is in relu() in affine_layers.py
                d_l = d_l + torch.where(A_l > 0, A_l, 0.) @ b_l + torch.where(A_l <= 0, A_l, 0.) @ b_u
                d_u = d_u + torch.where(A_u > 0, A_u, 0.) @ b_u + torch.where(A_u <= 0, A_u, 0.) @ b_l
                A_l = torch.where(A_l > 0, A_l, 0.) @ W_l + torch.where(A_l <= 0, A_l, 0.) @ W_u
                A_u = torch.where(A_u > 0, A_u, 0.) @ W_u + torch.where(A_u <= 0, A_u, 0.) @ W_l
            elif bounds[i]['name'] == 'dense':  # Update linear bounds from linear layers
                d_l = d_l + A_l @ b_l
                d_u = d_u + A_u @ b_u
                A_l = A_l @ W_l
                A_u = A_u @ W_u

        lower_bound = center @ A_l.T - diff @ A_l.abs().T + d_l  # Concretize the linear bound with input bounds
        upper_bound = center @ A_u.T + diff @ A_u.abs().T + d_u
        # loss = upper_bound.sum() - lower_bound.sum()
        loss = - lower_bound.sum()
        return loss  # .sum() is used as a replacement for .item(), which is not supported by vmap()

    if not optimize:
        return compute_bounds()
    optim_params = []
    for n_op in bounds:
        if bounds[n_op]['name'] == 'relu':
            # bounds[n_op]['A_l'] = nn.Parameter(bounds[n_op]['A_l'])
            # optim_params.append(bounds[n_op]['A_l'])
            optim_params.append(nn.Parameter(bounds[n_op]['A_l']))

    # optim_params = tuple(optim_params)
    optimizer = torch.optim.SGD(optim_params, lr=lr)
    old_loss = torch.inf
    for i in range(num_iter):
        # loss = compute_loss()
        # optimizer.zero_grad()
        for param in optim_params:
            if param.grad is not None:
                param.grad.zero_()

        gradients = grad(compute_loss)(optim_params)
        for optim_param, gradient in zip(optim_params, gradients):
            fro_norm = torch.norm(gradient, p='fro')
            optim_param.grad = gradient / fro_norm
        optimizer.step()
        for param in optim_params:
            param = torch.clamp(param, 0., 1.)
        loss = compute_loss()
        # print("Loss: ", compute_loss())
        # print("mean of alpha: ", optim_params[0].mean())
    return compute_bounds()

def truncate_affine(ctx, input):
    # do nothing if the input is a constant or we are not in truncate mode
    if is_const(input): return input
    if ctx.mode != 'affine_truncate':
        return input

    # gather values
    base, aff, err = input
    n_keep = ctx.truncate_count

    # if the affine list is shorter than the truncation length, nothing to do
    if aff.shape[0] <= n_keep:
        return input

    # compute the magnitudes of each affine value
    # TODO fanicier policies?
    if ctx.truncate_policy == 'absolute':
        affine_mags = torch.sum(torch.abs(aff), dim=-1)
    elif ctx.truncate_policy == 'relative':
        affine_mags = torch.sum(torch.abs(aff), dim=-1) / torch.abs(base)
    else:
        raise RuntimeError("bad policy")


    # sort the affine terms by by magnitude
    sort_inds = torch.argsort(-affine_mags, dim=-1) # sort to decreasing order
    aff = aff[sort_inds,:]

    # keep the n_keep highest-magnitude entries
    aff_keep = aff[:n_keep,:]
    aff_drop = aff[n_keep:,:]

    # for all the entries we aren't keeping, add their contribution to the interval error
    err = err + torch.sum(torch.abs(aff_drop), dim=0)

    return base, aff_keep, err

def apply_linear_approx(ctx, input, alpha, beta, delta, kappa=None):
    base, aff, err = input
    if ctx.mode != 'affine_quad':
        base = alpha * base + beta
        if aff is not None:
            aff = alpha * aff
        # This _should_ always be positive by definition. Always be sure your
        # approximation routines are generating positive delta.
        # At most, we defending against floating point error here.
        delta = torch.abs(delta)

    if ctx.mode in ['interval', 'affine_fixed']:
        err = alpha * err + delta
    elif ctx.mode in ['affine_truncate', 'affine_all']:
        err = alpha * err
        new_aff = torch.diag(delta)
        aff = torch.cat((aff, new_aff), dim=0)
        base, aff, err = truncate_affine(ctx, (base, aff, err))
    elif ctx.mode in ['affine_append']:
        err = alpha * err
        
        keep_vals, keep_inds = torch.topk(delta, ctx.n_append)
        row_inds = torch.arange(ctx.n_append)
        new_aff = torch.zeros((ctx.n_append, aff.shape[-1]))
        new_aff[row_inds, keep_inds] = keep_vals
        aff = torch.cat((aff, new_aff), dim=0)

        err = err + (torch.sum(delta) - torch.sum(keep_vals)) # add in the error for the affs we didn't keep
    elif ctx.mode in ['affine+backward']:
        err = alpha * err
        new_aff = torch.diag(delta)
        aff = torch.cat((aff, new_aff), dim=0)
        return (base, aff, err), alpha, beta - delta, beta + delta
    elif ctx.mode in ['affine_quad']:
        delta = torch.abs(delta)
        c_1 = (2 * kappa * base + alpha)
        quad_mask = kappa != 0.
        base = kappa * base ** 2 + alpha * base + beta #+ kappa * aff.abs().sum(dim=0) ** 2
        if aff is not None:
            aff = c_1 * aff
        err = c_1 * err
        new_aff = torch.diag(delta)
        aff = torch.cat((aff, new_aff), dim=0)

    return base, aff, err

# Convert to/from the affine representation from an ordinary value representing a scalar
def from_scalar(x):
    return x, None, None
def to_scalar(input):
    if not is_const(input):
        raise ValueError("non const input")
    return input[0]
def wrap_scalar(func):
    return lambda x : to_scalar(func(from_scalar(x)))
