import os.path

import itertools
from functools import partial
import math
import random

import numpy as np
from functorch import vmap
from triton.language import dtype

import utils
from bucketing import *
import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
import extract_cell
import geometry
import torch
from torch import Tensor
from typing import Tuple, Union

from crown import CrownImplicitFunction
from crown import deconstruct_lbias
from heuristics import input_split_branching
from clip_utils import clip_domains
from split import kd_split

INVALID_IND = 2**30
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_printoptions(threshold=6)
DEBUG_NONUNIFORM_KDTREE = False


def planes_intersect_cubes(normals, offsets, lowers, uppers):
    # normals: Tensor of shape (n, 3) where n is the number of planes
    # offsets: Tensor of shape (n,)
    # lowers: Tensor of shape (n, 3) where n is the number of cubes
    # uppers: Tensor of shape (n, 3)

    # Create all 8 vertices for each cube
    x_min, y_min, z_min = lowers.T
    x_max, y_max, z_max = uppers.T

    vertices = torch.stack([
        torch.stack((x_min, y_min, z_min), dim=-1),
        torch.stack((x_min, y_min, z_max), dim=-1),
        torch.stack((x_min, y_max, z_min), dim=-1),
        torch.stack((x_min, y_max, z_max), dim=-1),
        torch.stack((x_max, y_min, z_min), dim=-1),
        torch.stack((x_max, y_min, z_max), dim=-1),
        torch.stack((x_max, y_max, z_min), dim=-1),
        torch.stack((x_max, y_max, z_max), dim=-1)
    ], dim=1)  # Shape: (n, 8, 3)

    # Compute the dot products for each vertex with the corresponding plane normal
    # Resulting shape: (n, 8)
    dots = torch.matmul(vertices, normals.unsqueeze(-1)).squeeze(-1) + offsets.unsqueeze(-1)

    # Check if the vertices for each cube have different signs when plugged into plane equations
    signs = (dots > 0).int()

    # Check if each cube has both positive and negative signs for its plane, indicating intersection
    intersects = (signs.min(dim=-1).values < 1) & (signs.max(dim=-1).values > 0)

    return intersects  # Shape: (n,)

def get_distance(x_L, x_U, n_lower, d_lower, n_upper, d_upper):
    r"""
    x_L: (batch_size, input_size)
    x_U: (batch_size, input_size)

    n_lower (n_upper): (batch_size, 3, 1)
    d_lower (d_upper): (batch_size, 1)

    plane equation: n[0]*x + n[1]*y + n[2]*z + d = 0
    """
    device = x_L.device
    ndim = x_L.shape[-1]

    # Create indices
    # indices:
    # [[1, 1],
    #  [1, 2],
    #  [2, 1],
    #  [2, 2]]

    # all_indices (batch_size, 2^(input_size-1)*input_size, input_size):
    # [[0, 1, 1],
    #  [0, 1, 2],
    #  [0, 2, 1],
    #  [0, 2, 2],
    #  [1, 0, 1],
    #  [1, 0, 2],
    #  ...
    #  [2, 2, 0]] (repeat batch_size times)
    # 2^(input_size-1)*input_size is the number of edges
    binary_numbers = [list(map(int, bits)) for bits in itertools.product('12', repeat=ndim - 1)]
    indices = torch.tensor(binary_numbers, device=device)
    indices_with_zeros = []
    for i in range(ndim):
        zeros_column = torch.zeros((2 ** (ndim - 1), 1), dtype=int, device=device)
        new_matrix = torch.cat((indices[:, :i], zeros_column, indices[:, i:]), dim=1)
        indices_with_zeros.append(new_matrix)
    all_indices = torch.cat(indices_with_zeros, dim=0)
    all_indices = all_indices.unsqueeze(0).repeat(x_L.shape[0], 1, 1)

    # input_domain (batch_size, 3, input_size):
    # [[0,   0,   0  ],
    #  [x_l, y_l, z_l],
    #  [x_u, y_u, z_u]]
    input_domain = torch.stack((torch.zeros_like(x_L), x_L, x_U), dim=1)

    # Two end points for each edge (batch_size, 2^(input_size-1)*input_size, 2)
    bound_to_check_in_box = torch.zeros(*all_indices.shape[:2], 2, device=device)
    index_to_check_in_box = (all_indices == 0).nonzero()[:, 2].reshape(bound_to_check_in_box.shape[0], -1)
    bound_to_check_in_box[:, :, 0] = torch.gather(x_L, dim=1, index=index_to_check_in_box)
    bound_to_check_in_box[:, :, 1] = torch.gather(x_U, dim=1, index=index_to_check_in_box)

    # All vertices of the box (batch_size, 2^input_size, input_size)
    binary_numbers = [list(map(int, bits)) for bits in itertools.product('12', repeat=ndim)]
    vertices_indices = torch.tensor(binary_numbers, device=device).unsqueeze(0).repeat(x_L.shape[0], 1, 1)
    all_vertices = torch.gather(input_domain, dim=1, index=vertices_indices)

    def _get_hook(n, d):
        # temp_cofficients[b][i][j] = input_domain[b][all_indices[b][i][j]][j]
        temp_edge_intersections = torch.gather(input_domain, dim=1, index=all_indices)
        temp_edge = torch.zeros_like(temp_edge_intersections, device=temp_edge_intersections.device)

        denominators = n.repeat(1, 1, 2 ** (ndim - 1)).flatten(1)
        intersections = -(torch.bmm(temp_edge_intersections, n).squeeze(-1) + d) / denominators
        temp_edge[all_indices == 0] = intersections.flatten()
        edge_intersections = temp_edge_intersections + temp_edge

        valid_intersections = torch.logical_and(intersections >= bound_to_check_in_box[:, :, 0],
                                                intersections <= bound_to_check_in_box[:, :, 1])

        average_intersections = torch.einsum('bij, bi -> bij', edge_intersections, valid_intersections).mean(dim=1)

        # Now compute the distances from vertices to planes
        # distance = (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2) (signed)
        all_distances = (torch.bmm(all_vertices, n).squeeze(-1) + d) / torch.norm(n, dim=1)

        completely_outside = torch.logical_or(torch.all(all_distances >= 0, dim=1),
                                              torch.all(all_distances <= 0, dim=1))

        shortest_distance, shortest_index = torch.min(torch.abs(all_distances), dim=1)

        # x_h = x - (ax + by + cz + d)/(a^2 + b^2 + c^2) * a
        feet_perpendicular = all_vertices - (all_distances / torch.norm(n, dim=1)).unsqueeze(-1) * n.unsqueeze(
            1).squeeze(-1)
        shortest_feet = feet_perpendicular[torch.arange(feet_perpendicular.shape[0]), shortest_index]

        chosen_feet = torch.einsum('bj, b -> bj', shortest_feet, completely_outside)

        hook = average_intersections + chosen_feet
        return hook

    hook_lower = _get_hook(n_lower, d_lower)
    hook_upper = _get_hook(n_upper, d_upper)
    domain_loss = torch.norm(hook_lower - hook_upper, dim=1)
    return domain_loss


def split_generator(generator, num_splits=2):
    return [torch.Generator(device=device).manual_seed(generator.initial_seed() + i + 1) for i in range(num_splits)]


# @partial(jax.jit, static_argnames=("func","continue_splitting"), donate_argnums=(7,8,9,10))
def construct_uniform_unknown_levelset_tree_iter(
        func, params, continue_splitting,
        node_valid, node_lower, node_upper,
        ib, out_valid, out_lower, out_upper, out_n_valid,
        finished_interior_lower, finished_interior_upper, N_finished_interior,
        finished_exterior_lower, finished_exterior_upper, N_finished_exterior,
        offset=0.
        ):
    N_in = node_lower.shape[0]
    d = node_lower.shape[-1]

    def eval_one_node(lower, upper):
        # perform an affine evaluation
        if isinstance(func, CrownImplicitFunction):
            node_type = func.classify_box(params, lower, upper, offset=offset)[0]
        else:
            node_type = func.classify_box(params, lower, upper, offset=offset)

        # use the largest length along any dimension as the split policy
        worst_dim = torch.argmax(upper - lower, dim=-1)
        return node_type, worst_dim

    def eval_batch_of_nodes(lower, upper):
        if isinstance(func, CrownImplicitFunction):
            node_type = func.classify_box(params, lower, upper, offset=offset)[0].squeeze(-1)
        else:
            node_type = func.classify_box(params, lower, upper, offset=offset).squeeze(-1)

        worst_dim = torch.argmax(upper - lower, dim=-1)
        return node_type, worst_dim

    if isinstance(func, CrownImplicitFunction):
        batch_size_per_iteration = 256
        total_samples = node_lower.shape[0]
        node_types = torch.empty((total_samples,))
        node_split_dim = torch.empty((total_samples,))
        for start_idx in range(0, total_samples, batch_size_per_iteration):
            end_idx = min(start_idx + batch_size_per_iteration, total_samples)
            node_types[start_idx:end_idx], node_split_dim[start_idx:end_idx] \
                = eval_batch_of_nodes(node_lower[start_idx:end_idx], node_upper[start_idx:end_idx])
    else:
        # evaluate the function inside nodes
        node_types, node_split_dim = vmap(eval_one_node)(node_lower, node_upper)


    # if requested, write out interior nodes
    if finished_interior_lower is not None:
        out_mask = torch.logical_and(node_valid, node_types == SIGN_NEGATIVE)
        out_inds = utils.enumerate_mask(out_mask) + N_finished_interior
        mask = (- 1 < out_inds) & (out_inds < finished_interior_lower.shape[0])
        out_inds = out_inds[mask]
        node_interior_lower = node_lower[mask].float()
        node_interior_upper = node_upper[mask].float()
        # rand_ind = random.randint(0, len(node_interior_lower) - 1)
        # print("interior: ", node_interior_lower[rand_ind], node_interior_lower[rand_ind])
        # finished_interior_lower = finished_interior_lower.at[out_inds,:].set(node_lower, mode='drop')
        # finished_interior_upper = finished_interior_upper.at[out_inds,:].set(node_upper, mode='drop')
        finished_interior_lower[out_inds, :] = node_interior_lower
        finished_interior_upper[out_inds, :] = node_interior_upper
        N_finished_interior += torch.sum(out_mask)

    # if requested, write out exterior nodes
    if finished_exterior_lower is not None:
        out_mask = torch.logical_and(node_valid, node_types == SIGN_POSITIVE)
        out_inds = utils.enumerate_mask(out_mask) + N_finished_exterior
        mask = (- 1 < out_inds) & (out_inds < finished_exterior_lower.shape[0])
        out_inds = out_inds[mask]
        node_exterior_lower = node_lower[mask].float()
        node_exterior_upper = node_upper[mask].float()
        # rand_ind = random.randint(0, len(node_exterior_lower)-1)
        # print("exterior: ", node_exterior_lower[rand_ind], node_exterior_upper[rand_ind])
        # finished_exterior_lower = finished_exterior_lower.at[out_inds,:].set(node_lower, mode='drop')
        # finished_exterior_upper = finished_exterior_upper.at[out_inds,:].set(node_upper, mode='drop')
        finished_exterior_lower[out_inds, :] = node_exterior_lower
        finished_exterior_upper[out_inds, :] = node_exterior_upper
        N_finished_exterior += torch.sum(out_mask)

    # split the unknown nodes to children
    # (if split_children is False this will just not create any children at all)
    split_mask = utils.logical_and_all([node_valid, node_types == SIGN_UNKNOWN])
    N_new = torch.sum(split_mask)  # each split leads to two children (for a total of 2*N_new)
    # rand_ind = random.randint(0, N_new - 1)
    # print("unknown: ", node_lower[split_mask][rand_ind], node_upper[split_mask][rand_ind])
    ## now actually build the child nodes
    if continue_splitting:

        # extents of the new child nodes along each split dimension
        new_lower = node_lower
        new_upper = node_upper
        new_mid = 0.5 * (new_lower + new_upper)
        new_coord_mask = torch.arange(3)[None, :] == node_split_dim[:, None]
        newA_lower = new_lower
        newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
        newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
        newB_upper = new_upper

        # concatenate the new children to form output arrays
        node_valid = torch.cat((split_mask, split_mask))
        node_lower = torch.cat((newA_lower, newB_lower))
        node_upper = torch.cat((newA_upper, newB_upper))
        new_N_valid = 2 * N_new
        outL = out_valid.shape[1]
        # print(node_valid.sum(), node_valid.shape)

    else:
        node_valid = torch.logical_and(node_valid, node_types == SIGN_UNKNOWN)
        new_N_valid = torch.sum(node_valid)
        outL = node_valid.shape[0]

    # write the result in to arrays
    # utils.printarr(out_valid, node_valid, out_lower, node_lower, out_upper, node_upper)
    out_valid[ib, :outL] = node_valid
    out_lower[ib, :outL, :] = node_lower
    out_upper[ib, :outL, :] = node_upper
    out_n_valid = out_n_valid + new_N_valid

    return out_valid, out_lower, out_upper, out_n_valid, \
        finished_interior_lower, finished_interior_upper, N_finished_interior, \
        finished_exterior_lower, finished_exterior_upper, N_finished_exterior


def construct_uniform_unknown_levelset_tree(func, params, lower, upper, node_terminate_thresh=None, split_depth=None,
                                            compress_after=False, with_childern=False, with_interior_nodes=False,
                                            with_exterior_nodes=False, offset=0., batch_process_size=2048):
    # Validate input
    # ASSUMPTION: all of our bucket sizes larger than batch_process_size must be divisible by batch_process_size
    for b in bucket_sizes:
        if b > batch_process_size and (b // batch_process_size) * batch_process_size != b:
            raise ValueError(
                f"batch_process_size must be a factor of our bucket sizes, is not a factor of {b} (try a power of 2)")
    if node_terminate_thresh is None and split_depth is None:
        raise ValueError("must specify at least one of node_terminate_thresh or split_depth as a terminating condition")
    if node_terminate_thresh is None:
        node_terminate_thresh = 9999999999

    d = lower.shape[-1]
    B = batch_process_size

    print(f"\n == CONSTRUCTING LEVELSET TREE")

    # Initialize data
    node_lower = lower[None, :]
    node_upper = upper[None, :]
    node_valid = torch.ones((1,), dtype=torch.bool)
    N_curr_nodes = 1
    finished_interior_lower = torch.zeros((batch_process_size, 3)) if with_interior_nodes else None
    finished_interior_upper = torch.zeros((batch_process_size, 3)) if with_interior_nodes else None
    N_finished_interior = 0
    finished_exterior_lower = torch.zeros((batch_process_size, 3)) if with_exterior_nodes else None
    finished_exterior_upper = torch.zeros((batch_process_size, 3)) if with_exterior_nodes else None
    N_finished_exterior = 0
    N_func_evals = 0

    ## Recursively build the tree
    i_split = 0
    n_splits = 99999999 if split_depth is None else split_depth + 1  # 1 extra because last round doesn't split
    for i_split in range(n_splits):
        # Reshape in to batches of size <= B
        init_bucket_size = node_lower.shape[0]
        this_b = min(B, init_bucket_size)
        N_func_evals += node_lower.shape[0]
        # utils.printarr(node_valid, node_lower, node_upper)
        node_valid = torch.reshape(node_valid, (-1, this_b))
        node_lower = torch.reshape(node_lower, (-1, this_b, d))
        node_upper = torch.reshape(node_upper, (-1, this_b, d))
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(
            N_curr_nodes / this_b))  # only the batches which are occupied (since valid nodes are densely packed at the start)

        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = (N_curr_nodes >= node_terminate_thresh) or i_split + 1 == n_splits
        do_continue_splitting = not quit_next

        print(
            f"Uniform levelset tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  bucket size: {init_bucket_size}  batch size: {this_b}  number of batches: {nb}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")

        # enlarge the finished nodes if needed
        if with_interior_nodes:
            while finished_interior_lower.shape[0] - N_finished_interior < N_curr_nodes:
                finished_interior_lower = utils.resize_array_axis(finished_interior_lower,
                                                                  2 * finished_interior_lower.shape[0])
                finished_interior_upper = utils.resize_array_axis(finished_interior_upper,
                                                                  2 * finished_interior_upper.shape[0])
        if with_exterior_nodes:
            while finished_exterior_lower.shape[0] - N_finished_exterior < N_curr_nodes:
                finished_exterior_lower = utils.resize_array_axis(finished_exterior_lower,
                                                                  2 * finished_exterior_lower.shape[0])
                finished_exterior_upper = utils.resize_array_axis(finished_exterior_upper,
                                                                  2 * finished_exterior_upper.shape[0])

        # map over the batches
        out_valid = torch.zeros((nb, 2 * this_b), dtype=torch.bool)
        out_lower = torch.zeros((nb, 2 * this_b, 3))
        out_upper = torch.zeros((nb, 2 * this_b, 3))
        total_n_valid = 0
        for ib in range(n_occ):
            out_valid, out_lower, out_upper, total_n_valid, \
                finished_interior_lower, finished_interior_upper, N_finished_interior, \
                finished_exterior_lower, finished_exterior_upper, N_finished_exterior, \
                = \
                construct_uniform_unknown_levelset_tree_iter(func, params, do_continue_splitting, \
                                                             node_valid[ib, ...], node_lower[ib, ...],
                                                             node_upper[ib, ...], \
                                                             ib, out_valid, out_lower, out_upper, total_n_valid, \
                                                             finished_interior_lower, finished_interior_upper,
                                                             N_finished_interior, \
                                                             finished_exterior_lower, finished_exterior_upper,
                                                             N_finished_exterior, \
                                                             offset=offset)

        node_valid = out_valid
        node_lower = out_lower
        node_upper = out_upper
        N_curr_nodes = total_n_valid

        # flatten back out
        node_valid = torch.reshape(node_valid, (-1,))
        node_lower = torch.reshape(node_lower, (-1, d))


        node_upper = torch.reshape(node_upper, (-1, d))

        # Compactify and rebucket arrays
        target_bucket_size = get_next_bucket_size(total_n_valid)
        node_valid, N_curr_nodes, node_lower, node_upper = compactify_and_rebucket_arrays(node_valid,
                                                                                          target_bucket_size,
                                                                                          node_lower, node_upper)

        if quit_next:
            break


    # pack the output in to a dict to support optional outputs
    out_dict = {
        'unknown_node_valid': node_valid,
        'unknown_node_lower': node_lower,
        'unknown_node_upper': node_upper,
    }

    if with_interior_nodes:
        out_dict['interior_node_valid'] = torch.arange(finished_interior_lower.shape[0]) < N_finished_interior
        out_dict['interior_node_lower'] = finished_interior_lower
        out_dict['interior_node_upper'] = finished_interior_upper

    if with_exterior_nodes:
        out_dict['exterior_node_valid'] = torch.arange(finished_exterior_lower.shape[0]) < N_finished_exterior
        out_dict['exterior_node_lower'] = finished_exterior_lower
        out_dict['exterior_node_upper'] = finished_exterior_upper


    return out_dict

def construct_adaptive_tree_iter(
        func, params, continue_splitting,
        node_valid, node_lower, node_upper,
        ib, out_valid, out_lower, out_upper, out_n_valid,
        finished_interior_lower, finished_interior_upper, N_finished_interior,
        finished_exterior_lower, finished_exterior_upper, N_finished_exterior,
        offset=0.
        ):
    N_in = node_lower.shape[0]
    d = node_lower.shape[-1]

    assert isinstance(func, CrownImplicitFunction)

    def eval_batch_of_nodes(lower, upper):
        node_type, crown_ret = func.classify_box(params, lower, upper, offset=offset)
        node_type = node_type.squeeze(-1)
        worst_dim = torch.argmax(upper - lower, dim=-1)
        # to_check = ((~planes_intersect_cubes(crown_ret['lA'].squeeze(1), crown_ret['lbias'].squeeze(1), lower, upper)) |
        #             (~planes_intersect_cubes(crown_ret['uA'].squeeze(1), crown_ret['ubias'].squeeze(1), lower, upper)))
        to_check = get_distance(lower, upper, crown_ret['lA'].transpose(1, 2), crown_ret['lbias'], crown_ret['uA'].transpose(1, 2), crown_ret['ubias']) > 0.001
        to_check = to_check | (~planes_intersect_cubes(crown_ret['lA'].squeeze(1), crown_ret['lbias'].squeeze(1), lower, upper))
        to_check = to_check | (~planes_intersect_cubes(crown_ret['uA'].squeeze(1), crown_ret['ubias'].squeeze(1), lower, upper))
        return node_type, worst_dim, to_check

    batch_size_per_iteration = 256
    total_samples = node_lower.shape[0]
    node_types = torch.empty((total_samples,))
    node_split_dim = torch.empty((total_samples,))
    node_to_check = torch.empty((total_samples,), dtype=torch.bool)
    for start_idx in range(0, total_samples, batch_size_per_iteration):
        end_idx = min(start_idx + batch_size_per_iteration, total_samples)
        node_types[start_idx:end_idx], node_split_dim[start_idx:end_idx], node_to_check[start_idx:end_idx] \
            = eval_batch_of_nodes(node_lower[start_idx:end_idx], node_upper[start_idx:end_idx])

    # if requested, write out interior nodes
    if finished_interior_lower is not None:
        out_mask = torch.logical_and(node_valid, node_types == SIGN_NEGATIVE) | utils.logical_and_all([node_valid, node_types == SIGN_UNKNOWN, ~node_to_check])
        out_inds = utils.enumerate_mask(out_mask) + N_finished_interior
        mask = (- 1 < out_inds) & (out_inds < finished_interior_lower.shape[0])
        out_inds = out_inds[mask]
        node_interior_lower = node_lower[mask].float()
        node_interior_upper = node_upper[mask].float()
        finished_interior_lower[out_inds, :] = node_interior_lower
        finished_interior_upper[out_inds, :] = node_interior_upper
        N_finished_interior += torch.sum(out_mask)

    # if requested, write out exterior nodes
    if finished_exterior_lower is not None:
        out_mask = torch.logical_and(node_valid, node_types == SIGN_POSITIVE)
        out_inds = utils.enumerate_mask(out_mask) + N_finished_exterior
        mask = (- 1 < out_inds) & (out_inds < finished_exterior_lower.shape[0])
        out_inds = out_inds[mask]
        node_exterior_lower = node_lower[mask].float()
        node_exterior_upper = node_upper[mask].float()
        finished_exterior_lower[out_inds, :] = node_exterior_lower
        finished_exterior_upper[out_inds, :] = node_exterior_upper
        N_finished_exterior += torch.sum(out_mask)


    # split the unknown nodes to children
    # (if split_children is False this will just not create any children at all)
    split_mask = utils.logical_and_all([node_valid, node_types == SIGN_UNKNOWN, node_to_check])
    N_new = torch.sum(split_mask)  # each split leads to two children (for a total of 2*N_new)
    ## now actually build the child nodes
    continue_splitting = split_mask.any()
    if split_mask.any():
        # extents of the new child nodes along each split dimension
        new_lower = node_lower
        new_upper = node_upper
        new_mid = 0.5 * (new_lower + new_upper)
        new_coord_mask = torch.arange(3)[None, :] == node_split_dim[:, None]
        newA_lower = new_lower
        newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
        newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
        newB_upper = new_upper

        # concatenate the new children to form output arrays
        node_valid = torch.cat((split_mask, split_mask))
        node_lower = torch.cat((newA_lower, newB_lower))
        node_upper = torch.cat((newA_upper, newB_upper))
        new_N_valid = 2 * N_new
        outL = out_valid.shape[1]
        # print(node_valid.sum(), node_valid.shape)

    else:
        # node_valid = torch.logical_and(node_valid, node_types == SIGN_UNKNOWN)
        node_valid = utils.logical_and_all([node_valid, node_types == SIGN_UNKNOWN, node_to_check])
        new_N_valid = torch.sum(node_valid)
        outL = node_valid.shape[0]

    # write the result in to arrays
    # utils.printarr(out_valid, node_valid, out_lower, node_lower, out_upper, node_upper)
    out_valid[ib, :outL] = node_valid
    out_lower[ib, :outL, :] = node_lower
    out_upper[ib, :outL, :] = node_upper
    out_n_valid = out_n_valid + new_N_valid

    return (out_valid, out_lower, out_upper, out_n_valid, \
        finished_interior_lower, finished_interior_upper, N_finished_interior, \
        finished_exterior_lower, finished_exterior_upper, N_finished_exterior, \
            continue_splitting)

def construct_adaptive_tree(func, params, lower, upper, node_terminate_thresh=None, split_depth=None,
                                            compress_after=False, with_childern=False, with_interior_nodes=False,
                                            with_exterior_nodes=False, offset=0., batch_process_size=2048):
    # Validate input
    # ASSUMPTION: all of our bucket sizes larger than batch_process_size must be divisible by batch_process_size
    for b in bucket_sizes:
        if b > batch_process_size and (b // batch_process_size) * batch_process_size != b:
            raise ValueError(
                f"batch_process_size must be a factor of our bucket sizes, is not a factor of {b} (try a power of 2)")
    if node_terminate_thresh is None and split_depth is None:
        raise ValueError("must specify at least one of node_terminate_thresh or split_depth as a terminating condition")
    if node_terminate_thresh is None:
        node_terminate_thresh = 9999999999

    d = lower.shape[-1]
    B = batch_process_size

    print(f"\n == CONSTRUCTING LEVELSET TREE")

    # Initialize data
    node_lower = lower[None, :]
    node_upper = upper[None, :]
    node_valid = torch.ones((1,), dtype=torch.bool)
    N_curr_nodes = 1
    finished_interior_lower = torch.zeros((batch_process_size, 3)) if with_interior_nodes else None
    finished_interior_upper = torch.zeros((batch_process_size, 3)) if with_interior_nodes else None
    N_finished_interior = 0
    finished_exterior_lower = torch.zeros((batch_process_size, 3)) if with_exterior_nodes else None
    finished_exterior_upper = torch.zeros((batch_process_size, 3)) if with_exterior_nodes else None
    N_finished_exterior = 0
    N_func_evals = 0

    ## Recursively build the tree
    i_split = 0
    n_splits = 99999999 if split_depth is None else split_depth + 1  # 1 extra because last round doesn't split
    do_continue_splitting = True
    while do_continue_splitting:
        if i_split > 36:
            break
        # Reshape in to batches of size <= B
        init_bucket_size = node_lower.shape[0]
        this_b = min(B, init_bucket_size)
        N_func_evals += node_lower.shape[0]
        # utils.printarr(node_valid, node_lower, node_upper)
        node_valid = torch.reshape(node_valid, (-1, this_b))
        node_lower = torch.reshape(node_lower, (-1, this_b, d))
        node_upper = torch.reshape(node_upper, (-1, this_b, d))
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(
            N_curr_nodes / this_b))  # only the batches which are occupied (since valid nodes are densely packed at the start)

        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = not do_continue_splitting

        print(
            f"Adaptive levelset tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  bucket size: {init_bucket_size}  batch size: {this_b}  number of batches: {nb}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")
        # enlarge the finished nodes if needed
        if with_interior_nodes:
            while finished_interior_lower.shape[0] - N_finished_interior < N_curr_nodes:
                finished_interior_lower = utils.resize_array_axis(finished_interior_lower,
                                                                  2 * finished_interior_lower.shape[0])
                finished_interior_upper = utils.resize_array_axis(finished_interior_upper,
                                                                  2 * finished_interior_upper.shape[0])
        if with_exterior_nodes:
            while finished_exterior_lower.shape[0] - N_finished_exterior < N_curr_nodes:
                finished_exterior_lower = utils.resize_array_axis(finished_exterior_lower,
                                                                  2 * finished_exterior_lower.shape[0])
                finished_exterior_upper = utils.resize_array_axis(finished_exterior_upper,
                                                                  2 * finished_exterior_upper.shape[0])

        # map over the batches
        out_valid = torch.zeros((nb, 2 * this_b), dtype=torch.bool)
        out_lower = torch.zeros((nb, 2 * this_b, 3))
        out_upper = torch.zeros((nb, 2 * this_b, 3))
        total_n_valid = 0
        for ib in range(n_occ):
            if i_split <= split_depth:
                (out_valid, out_lower, out_upper, total_n_valid,
                    finished_interior_lower, finished_interior_upper, N_finished_interior,
                    finished_exterior_lower, finished_exterior_upper, N_finished_exterior) \
                    = \
                    construct_uniform_unknown_levelset_tree_iter(func, params, do_continue_splitting,
                                                 node_valid[ib, ...], node_lower[ib, ...], node_upper[ib, ...],
                                                 ib, out_valid, out_lower, out_upper, total_n_valid,
                                                 finished_interior_lower, finished_interior_upper, N_finished_interior,
                                                 finished_exterior_lower, finished_exterior_upper, N_finished_exterior,
                                                 offset=offset)
            else:
                (out_valid, out_lower, out_upper, total_n_valid,
                 finished_interior_lower, finished_interior_upper, N_finished_interior,
                 finished_exterior_lower, finished_exterior_upper, N_finished_exterior,
                 do_continue_splitting) \
                    = \
                    construct_adaptive_tree_iter(func, params, do_continue_splitting,
                                                 node_valid[ib, ...], node_lower[ib, ...], node_upper[ib, ...],
                                                 ib, out_valid, out_lower, out_upper, total_n_valid,
                                                 finished_interior_lower, finished_interior_upper, N_finished_interior,
                                                 finished_exterior_lower, finished_exterior_upper, N_finished_exterior,
                                                 offset=offset)

        node_valid = out_valid
        node_lower = out_lower
        node_upper = out_upper
        N_curr_nodes = total_n_valid

        # flatten back out
        node_valid = torch.reshape(node_valid, (-1,))
        node_lower = torch.reshape(node_lower, (-1, d))


        node_upper = torch.reshape(node_upper, (-1, d))

        # Compactify and rebucket arrays
        target_bucket_size = get_next_bucket_size(total_n_valid)
        node_valid, N_curr_nodes, node_lower, node_upper = compactify_and_rebucket_arrays(node_valid,
                                                                                          target_bucket_size,
                                                                                          node_lower, node_upper)
        i_split += 1


    # pack the output in to a dict to support optional outputs
    out_dict = {
        'unknown_node_valid': node_valid,
        'unknown_node_lower': node_lower,
        'unknown_node_upper': node_upper,
    }

    if with_interior_nodes:
        out_dict['interior_node_valid'] = torch.arange(finished_interior_lower.shape[0]) < N_finished_interior
        out_dict['interior_node_lower'] = finished_interior_lower
        out_dict['interior_node_upper'] = finished_interior_upper

    if with_exterior_nodes:
        out_dict['exterior_node_valid'] = torch.arange(finished_exterior_lower.shape[0]) < N_finished_exterior
        out_dict['exterior_node_lower'] = finished_exterior_lower
        out_dict['exterior_node_upper'] = finished_exterior_upper


    return out_dict

def construct_static_unknown_tree_iter(func, params, node_lower, node_upper, continue_splitting, batch_size=256):
    def eval_batch_of_nodes(lower, upper):
        types, crown_ret = func.classify_box(params, lower, upper)
        types = types.squeeze(-1)
        lAs = crown_ret['lA'].squeeze(1)
        lbs = crown_ret['lbias'].squeeze(1)
        uAs = crown_ret['uA'].squeeze(1)
        ubs = crown_ret['ubias'].squeeze(1)
        return types, lAs, lbs, uAs, ubs

    total_samples = node_lower.shape[0]
    node_type = torch.empty((total_samples,))
    node_lA = torch.empty((total_samples, 3))
    node_lb = torch.empty((total_samples,))
    node_uA = torch.empty((total_samples, 3))
    node_ub = torch.empty((total_samples,))

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        (node_type[start_idx:end_idx],
         node_lA[start_idx:end_idx], node_lb[start_idx:end_idx],
         node_uA[start_idx:end_idx], node_ub[start_idx:end_idx]) = (
            eval_batch_of_nodes(node_lower[start_idx:end_idx], node_upper[start_idx:end_idx]))


    neg_mask = node_type == SIGN_NEGATIVE
    unk_mask = node_type == SIGN_UNKNOWN

    if continue_splitting:
        split_mask = unk_mask
        split_node_lower = node_lower[split_mask]
        split_node_upper = node_upper[split_mask]
        node_split_dim = torch.argmax(split_node_upper - split_node_lower, dim=-1)
        new_lower = split_node_lower
        new_upper = split_node_upper
        new_mid = 0.5 * (new_lower + new_upper)
        new_coord_mask = torch.arange(3)[None, :] == node_split_dim[:, None]
        newA_lower = new_lower
        newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
        newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
        newB_upper = new_upper

        # concatenate the new children to form output arrays
        split_node_lower = torch.cat((newA_lower, newB_lower))
        split_node_upper = torch.cat((newA_upper, newB_upper))

        finished_mask = neg_mask
    else:
        split_node_lower, split_node_upper = None, None
        finished_mask = neg_mask | unk_mask

    finished_node_lower = node_lower[finished_mask]
    finished_node_upper = node_upper[finished_mask]
    finished_node_lA = node_lA[finished_mask]
    finished_node_lb = node_lb[finished_mask]
    finished_node_uA = node_uA[finished_mask]
    finished_node_ub = node_ub[finished_mask]

    return (
    finished_node_lower, finished_node_upper, finished_node_lA, finished_node_lb, finished_node_uA, finished_node_ub,
    split_node_lower, split_node_upper)


def construct_dynamic_unknown_tree_iter(func, params, node_lower, node_upper, continue_splitting, batch_size=256):
    def eval_batch_of_nodes(lower, upper):
        types, crown_ret = func.classify_box(params, lower, upper)
        types = types.squeeze(-1)
        lAs = crown_ret['lA'].squeeze(1)
        lbs = crown_ret['lbias'].squeeze(1)
        uAs = crown_ret['uA'].squeeze(1)
        ubs = crown_ret['ubias'].squeeze(1)
        return types, lAs, lbs, uAs, ubs

    total_samples = node_lower.shape[0]
    node_type = torch.empty((total_samples,))
    node_lA = torch.empty((total_samples, 3))
    node_lb = torch.empty((total_samples,))
    node_uA = torch.empty((total_samples, 3))
    node_ub = torch.empty((total_samples,))

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        (node_type[start_idx:end_idx],
         node_lA[start_idx:end_idx], node_lb[start_idx:end_idx],
         node_uA[start_idx:end_idx], node_ub[start_idx:end_idx]) = (
            eval_batch_of_nodes(node_lower[start_idx:end_idx], node_upper[start_idx:end_idx]))



    neg_mask = node_type == SIGN_NEGATIVE
    unk_mask = node_type == SIGN_UNKNOWN
    large_dist_mask = get_distance(node_lower, node_upper, node_lA.unsqueeze(-1), node_lb.unsqueeze(-1),
                            node_uA.unsqueeze(-1), node_ub.unsqueeze(-1)) > 0.001
    bad_plane_mask = ((~planes_intersect_cubes(node_lA, node_lb, node_lower, node_upper)) |
                      (~planes_intersect_cubes(node_uA, node_ub, node_lower, node_upper)))
    if continue_splitting:
        split_mask = unk_mask & (large_dist_mask | bad_plane_mask)
        split_node_lower = node_lower[split_mask]
        split_node_upper = node_upper[split_mask]
        node_split_dim = torch.argmax(split_node_upper - split_node_lower, dim=-1)
        new_lower = split_node_lower
        new_upper = split_node_upper
        new_mid = 0.5 * (new_lower + new_upper)
        new_coord_mask = torch.arange(3)[None, :] == node_split_dim[:, None]
        newA_lower = new_lower
        newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
        newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
        newB_upper = new_upper

        # concatenate the new children to form output arrays
        split_node_lower = torch.cat((newA_lower, newB_lower))
        split_node_upper = torch.cat((newA_upper, newB_upper))

        finished_mask = neg_mask | (unk_mask & (~large_dist_mask & ~bad_plane_mask))
    else:
        split_node_lower, split_node_upper = None, None
        finished_mask = neg_mask | unk_mask

    finished_node_lower = node_lower[finished_mask]
    finished_node_upper = node_upper[finished_mask]
    finished_node_lA = node_lA[finished_mask]
    finished_node_lb = node_lb[finished_mask]
    finished_node_uA = node_uA[finished_mask]
    finished_node_ub = node_ub[finished_mask]

    return (finished_node_lower, finished_node_upper, finished_node_lA, finished_node_lb, finished_node_uA, finished_node_ub,
            split_node_lower, split_node_upper)

def construct_hybrid_unknown_tree(func, params, lower, upper, base_depth=21, max_depth=36, delta=0.001, batch_size=256):
    out_lower = []
    out_upper = []
    out_lA = []
    out_lb = []
    out_uA = []
    out_ub = []
    i_depth = 0
    to_split_lower = lower.unsqueeze(0)
    to_split_upper = upper.unsqueeze(0)
    continue_splitting = True
    while i_depth < base_depth:
        ret = construct_static_unknown_tree_iter(func, params, to_split_lower, to_split_upper, continue_splitting, batch_size)
        out_lower.append(ret[0])
        out_upper.append(ret[1])
        out_lA.append(ret[2])
        out_lb.append(ret[3])
        out_uA.append(ret[4])
        out_ub.append(ret[5])
        to_split_lower = ret[6]
        to_split_upper = ret[7]
        i_depth += 1

    while i_depth < max_depth:
        if i_depth + 1 == max_depth:
            continue_splitting = False
        ret = construct_dynamic_unknown_tree_iter(func, params, to_split_lower, to_split_upper, continue_splitting,
                                                 batch_size)
        out_lower.append(ret[0])
        out_upper.append(ret[1])
        out_lA.append(ret[2])
        out_lb.append(ret[3])
        out_uA.append(ret[4])
        out_ub.append(ret[5])
        to_split_lower = ret[6]
        to_split_upper = ret[7]
        i_depth += 1

    out_lower = torch.cat(out_lower)
    out_upper = torch.cat(out_upper)
    out_lA = torch.cat(out_lA)
    out_lb = torch.cat(out_lb)
    out_uA = torch.cat(out_uA)
    out_ub = torch.cat(out_ub)
    return out_lower, out_upper, out_lA, out_lb, out_uA, out_ub


def construct_full_uniform_unknown_levelset_tree_iter(
        func, params, continue_splitting,
        node_lower, node_upper,
        split_level,
        offset=0., batch_size=None
):
    """

    A single iteration which takes batches of nodes, evaluates their status, and splits UNKNOWN nodes uniformly.

    :param func:                The implicit function to evaluate nodes. Should be a CROWN mode
    :param params:
    :param continue_splitting:  Denotes whether we should split after bounding the nodes
    :param node_lower:          Lower input domain for batches
    :param node_upper:          Upper input domain for batches
    :param split_level:         The current split depth in the tree
    :param offset:              Spec to verify against. Typically, 0.
    :param batch_size:          If not None, nodes are processed in batches
    :return:
    """
    N_in = node_lower.shape[0]
    N_out = 2 * N_in
    d = node_lower.shape[-1]
    internal_node_mask = torch.logical_not(torch.isnan(node_lower[:, 0]))
    node_types = torch.full((N_in,), torch.nan)
    node_split_dim = torch.full((N_in,), torch.nan)
    node_split_val = torch.full((N_in,), torch.nan)
    node_lower_out = torch.full((N_out, 3), torch.nan)
    node_upper_out = torch.full((N_out, 3), torch.nan)

    def eval_one_node(lower, upper):

        # perform an affine evaluation
        node_type = func.classify_box(params, lower, upper, offset=offset)
        # use the largest length along any dimension as the split policy
        worst_dim = torch.argmax(upper - lower, dim=-1)
        return node_type.float(), worst_dim.float()

    def eval_batch_of_nodes(lower, upper):
        node_type, _ = func.classify_box(params, lower, upper, offset=offset)
        node_type = node_type.squeeze(-1)
        worst_dim = torch.argmax(upper - lower, dim=-1)
        return node_type.float(), worst_dim.float()

    node_types_temp = node_types[internal_node_mask]
    node_split_dim_temp = node_split_dim[internal_node_mask]

    if isinstance(func, CrownImplicitFunction):
        eval_func = eval_batch_of_nodes
    else:
        # raise NotImplementedError("Nonuniform kd tree with clipping only supported for CROWN methods")
        eval_func = vmap(eval_one_node)

    if batch_size is None:
        node_types[internal_node_mask], node_split_dim[internal_node_mask] = eval_func(
            node_lower[internal_node_mask], node_upper[internal_node_mask])
    else:
        batch_size_per_iteration = batch_size
        total_samples = node_lower[internal_node_mask].shape[0]
        for start_idx in range(0, total_samples, batch_size_per_iteration):
            end_idx = min(start_idx + batch_size_per_iteration, total_samples)
            node_types_temp[start_idx:end_idx], node_split_dim_temp[start_idx:end_idx] \
                = eval_func(node_lower[internal_node_mask][start_idx:end_idx],
                            node_upper[internal_node_mask][start_idx:end_idx])

        node_types[internal_node_mask] = node_types_temp
        node_split_dim[internal_node_mask] = node_split_dim_temp

    # split the unknown nodes to children
    # (if split_children is False this will just not create any children at all)
    split_mask = torch.logical_and(internal_node_mask, node_types == SIGN_UNKNOWN)
    ## now actually build the child nodes
    if continue_splitting:
        # extents of the new child nodes along each split dimension
        new_lower = node_lower
        new_upper = node_upper
        new_mid = 0.5 * (new_lower + new_upper)
        new_coord_mask = torch.arange(3)[None, :] == node_split_dim[:, None]
        newA_lower = new_lower
        newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
        newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
        newB_upper = new_upper

        # concatenate the new children to form output arrays
        inter_split_mask = split_mask.repeat_interleave(2)
        node_lower_out[inter_split_mask] = torch.hstack(
            (newA_lower[split_mask], newB_lower[split_mask])).view(inter_split_mask.sum(), d)
        node_upper_out[inter_split_mask] = torch.hstack(
            (newA_upper[split_mask], newB_upper[split_mask])).view(inter_split_mask.sum(), d)
        node_split_val[split_mask] = new_mid[
            torch.arange(len(node_types))[split_mask], node_split_dim[split_mask].long()]

    return node_lower_out, node_upper_out, node_types, node_split_dim, node_split_val


def construct_full_uniform_unknown_levelset_tree(
        func,
        params,
        lower,
        upper,
        split_depth=None,
        offset=0.,
        batch_size=None,
        load_from=None,
        save_to=None
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """

    Constructs a tree that is uniform. For now, only uses the naive branching heuristic.

    :param func:                The implicit function to evaluate nodes. Should be a CROWN mode
    :param params:
    :param lower:               Lower input domain for batches
    :param upper:               Upper input domain for batches
    :param split_depth:         Desired split depth granularity of the kD tree
    :param offset:              Spec to verify against. Typically, 0.
    :param batch_size:          If not None, nodes are processed in batches
    :return:
    """
    # if load_from and os.path.exists(load_from):
    #     tree = np.load(load_from).values()
    #     for _ in tree:
    #         _ = torch.from_numpy(_)
    #     return tree

    d = lower.shape[-1]

    print(f"\n == CONSTRUCTING LEVELSET TREE")

    # Initialize datas
    node_lower = [lower]
    node_upper = [upper]
    node_type = []
    split_dim = []
    split_val = []
    N_curr_nodes = 1
    N_func_evals = 0
    N_total_nodes = 1
    ## Recursively build the tree
    i_split = 0
    n_splits = split_depth + 1  # 1 extra because last round doesn't split
    for i_split in range(n_splits):
        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = i_split + 1 == n_splits
        do_continue_splitting = not quit_next

        print(
            f"Uniform levelset tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")

        total_n_valid = 0
        lower, upper, out_node_type, out_split_dim, out_split_val = construct_full_uniform_unknown_levelset_tree_iter(
            func, params, do_continue_splitting,
            lower, upper, i_split, offset=offset, batch_size=batch_size)
        node_lower.append(lower)
        node_upper.append(upper)
        node_type.append(out_node_type)
        split_dim.append(out_split_dim)
        split_val.append(out_split_val)
        N_curr_nodes = torch.logical_not(lower[:, 0].isnan()).sum()

        N_total_nodes += N_curr_nodes
        if quit_next:
            break

    node_lower = torch.cat(node_lower)
    node_upper = torch.cat(node_upper)
    node_type = torch.cat(node_type)
    split_dim = torch.cat(split_dim)
    split_val = torch.cat(split_val)

    # for key in out_dict:
    #     print(key, out_dict[key][:10])
    print("Total number of nodes evaluated: ", N_total_nodes)
    return node_lower, node_upper, node_type, split_dim, split_val


def construct_full_non_uniform_unknown_levelset_tree_iter(
        func,
        params,
        continue_splitting: bool,
        node_lower: Tensor,
        node_upper: Tensor,
        split_level: int,
        branching_method: str,
        offset: float = 0.,
        batch_size: Union[int, None] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """

    A single iteration that progresses the kD tree by one split depth. Clipping is introduced which creates an imbalance in the nodes. The procedure is now:

    * Bound nodes
    * Decides split dimensions via split heuristic
    * Split nodes and collect *lA* and *lbias*
    * Clip new nodes if possible to reduce input volume

    :param func:                The implicit function to evaluate nodes. Should be a CROWN mode
    :param params:
    :param continue_splitting:  Denotes whether we should split after bounding the nodes
    :param node_lower:          Lower input domain for batches
    :param node_upper:          Upper input domain for batches
    :param split_level:         The current split depth in the tree
    :param branching_method:    Specified branching heuristic to decide which dimension to split upon
    :param offset:              Spec to verify against. Typically, 0.
    :param batch_size:          If not None, nodes are processed in batches
    :return:                    Returns the newly split modes along with their status, split dimension, and split value
    """
    N_in = node_lower.shape[0]  # number of original nodes
    N_out = 2 * N_in  # number of nodes after splitting
    d = node_lower.shape[-1]  # input dimension
    internal_node_mask = torch.logical_not(torch.isnan(node_lower[:, 0]))  # mask for nodes that have not been verified
    internal_node_mask_len = internal_node_mask.to(int).sum().item()
    node_types = torch.full((N_in,), torch.nan)  # verification status
    node_split_dim = torch.full((N_in,), torch.nan)  # dimension to split upon
    node_split_val = torch.full((N_in,), torch.nan)  # midpoint split value
    node_lower_out = torch.full((N_out, 3), torch.nan)  # Lower input domain for new subdomains
    node_upper_out = torch.full((N_out, 3), torch.nan)  # Upper input domain for new subdomains

    def eval_batch_of_nodes(
            lower,
            upper,
            continue_splitting=False,
            clip_dimension=None
    ):
        """

        :param lower:               The input lower bounds of the nodes
        :param upper:               The input upper bounds of the nodes
        :param continue_splitting:  True if this evaluation should split the nodes after bounding them
        :return:                    Returns the (verification status, split dimension, split value, new_lb, new_ub)
        """

        # bounds the nodes
        ret = func.classify_box(params, lower, upper, offset)
        node_type, crown_dict = ret
        node_type = node_type.squeeze(-1)

        unverified_mask = node_type == SIGN_UNKNOWN  # only interested in domains that are unverified from this point

        new_split_values = torch.full_like(node_type, torch.nan, device=node_type.device)
        new_lower = torch.full((node_type.shape[0] * 2, 3), torch.nan, device=node_type.device)
        new_upper = torch.full((node_type.shape[0] * 2, 3), torch.nan, device=node_type.device)

        # call a branching heuristic to select dimensions to split upon
        dm_lb = crown_dict.get('dm_lb')
        lA = crown_dict.get('lA')
        lbias = crown_dict.get('lbias')
        new_split_idx = input_split_branching(dm_lb, lower, upper, lA, torch.full_like(lbias, offset),
                                              branching_method=branching_method).squeeze(1)
        # dm_lb dimension: (batches, 1)
        # lA dimension: (batches, 1, 3)
        # lbias dimension: (batches, 1)
        # split_idx dimension: (batches, 1)

        if not continue_splitting:
            # nodes only need to be bounded, not split
            return node_type, new_split_idx.float(), new_split_values.float(), lower, upper

        if not unverified_mask.any():
            return node_type.float(), new_split_idx.float(), new_split_values.float(), new_lower.float(), new_upper.float()

        # filter out domains that are already verified
        dm_lb = dm_lb[unverified_mask]
        lA = lA[unverified_mask]
        lbias = lbias[unverified_mask]
        split_idx = new_split_idx[unverified_mask]

        # FIXME: This should be removed but is here for debugging purposes
        check_lbias = deconstruct_lbias(lower[unverified_mask], upper[unverified_mask], lA, dm_lb)
        assert torch.allclose(lbias, check_lbias), "Deconstructed lbias does not match CROWN lbias"

        # produce the splits on the domains
        split_result = kd_split(lower[unverified_mask], upper[unverified_mask], split_idx)
        _newA_lower, _newA_upper, _newB_lower, _newB_upper, split_values = split_result

        # stack lower and upper bounds to work with clip_domains
        total_new_lower = torch.vstack((_newA_lower, _newB_lower)).float()
        total_new_upper = torch.vstack((_newA_upper, _newB_upper)).float()

        new_split_values[unverified_mask] = split_values

        # repeat along the batch dimension for newly split nodes
        lA = lA.repeat(2, 1, 1)
        lbias = lbias.repeat(2, 1)

        # apply clipping to reduce volume of input domain
        clip_results = clip_domains(
            total_new_lower, total_new_upper, offset, lA, None, lbias, True, clip_dimension)
        clipped_total_new_lower, clipped_total_new_upper = clip_results
        # clipped_total_new_lower/upper dimension: (N_out, n)

        if DEBUG_NONUNIFORM_KDTREE:
            clipped_total_new_lower = total_new_lower
            clipped_total_new_upper = total_new_upper

        # Clipping may cause x_L > x_U which means that the domain is verified,
        # and domains that are verified have NaN bounds
        clip_verified = (clipped_total_new_lower > clipped_total_new_upper).any(1)
        clipped_total_new_lower[clip_verified] = torch.nan
        clipped_total_new_upper[clip_verified] = torch.nan

        # put clipped domains back into our new domain list
        new_lower[unverified_mask.repeat(2)] = clipped_total_new_lower.float()
        new_upper[unverified_mask.repeat(2)] = clipped_total_new_upper.float() # Not interleave

        return node_type.float(), new_split_idx.float(), new_split_values.float(), new_lower.float(), new_upper.float()


    newA_lower, newB_lower, newA_upper, newB_upper = [torch.full_like(node_lower, torch.nan) for _ in range(4)]
    node_types_temp = node_types[internal_node_mask]
    node_split_dim_temp = node_split_dim[internal_node_mask]
    node_split_val_temp = node_split_val[internal_node_mask]
    newA_lower_temp = newA_lower[internal_node_mask]
    newB_lower_temp = newB_lower[internal_node_mask]
    newA_upper_temp = newA_upper[internal_node_mask]
    newB_upper_temp = newB_upper[internal_node_mask]

    if isinstance(func, CrownImplicitFunction):
        eval_func = eval_batch_of_nodes
    else:
        raise TypeError("Evaluation function must be a CROWN implicit function")

    if batch_size is None:
        eval_result = eval_func(
            node_lower[internal_node_mask], node_upper[internal_node_mask], continue_splitting, clip_dimension=1)

        node_types[internal_node_mask], node_split_dim[internal_node_mask], node_split_val[
            internal_node_mask] = eval_result[:3]
        if continue_splitting:
            total_new_lower, total_new_upper = eval_result[3:]
            newA_lower[internal_node_mask] = total_new_lower[:internal_node_mask_len]
            newB_lower[internal_node_mask] = total_new_lower[internal_node_mask_len:]
            newA_upper[internal_node_mask] = total_new_upper[:internal_node_mask_len]
            newB_upper[internal_node_mask] = total_new_upper[internal_node_mask_len:]

    else:
        batch_size_per_iteration = batch_size
        total_samples = node_lower[internal_node_mask].shape[0]
        for start_idx in range(0, total_samples, batch_size_per_iteration):
            end_idx = min(start_idx + batch_size_per_iteration, total_samples)
            eval_result = eval_func(node_lower[internal_node_mask][start_idx:end_idx],
                                    node_upper[internal_node_mask][start_idx:end_idx],
                                    continue_splitting, clip_dimension=[1, 2])
            node_types_temp[start_idx:end_idx], node_split_dim_temp[start_idx:end_idx], node_split_val_temp[
                                                                                        start_idx:end_idx] = eval_result[:3]
            total_new_lower, total_new_upper = eval_result[3:]
            if continue_splitting:
                diff_idx = end_idx - start_idx
                newA_lower_temp[start_idx:end_idx] = total_new_lower[:diff_idx]
                newB_lower_temp[start_idx:end_idx] = total_new_lower[diff_idx:]
                newA_upper_temp[start_idx:end_idx] = total_new_upper[:diff_idx]
                newB_upper_temp[start_idx:end_idx] = total_new_upper[diff_idx:]

        node_types[internal_node_mask] = node_types_temp
        node_split_dim[internal_node_mask] = node_split_dim_temp
        node_split_val[internal_node_mask] = node_split_val_temp
        newA_lower[internal_node_mask] = newA_lower_temp
        newB_lower[internal_node_mask] = newB_lower_temp
        newA_upper[internal_node_mask] = newA_upper_temp
        newB_upper[internal_node_mask] = newB_upper_temp

    # split the unknown nodes to children
    ## now actually build the child nodes
    if continue_splitting:
        # (if split_children is False this will just not create any children at all)
        split_mask = torch.logical_and(internal_node_mask, node_types == SIGN_UNKNOWN)
        # concatenate the new children to form output arrays
        inter_split_mask = split_mask.repeat_interleave(2)
        node_lower_out[inter_split_mask] = torch.hstack(
            (newA_lower[split_mask], newB_lower[split_mask])).view(inter_split_mask.sum(), d)
        node_upper_out[inter_split_mask] = torch.hstack(
            (newA_upper[split_mask], newB_upper[split_mask])).view(inter_split_mask.sum(), d)

    return node_lower_out, node_upper_out, node_types, node_split_dim, node_split_val


def construct_full_non_uniform_unknown_levelset_tree(
        func,
        params,
        lower: Tensor,
        upper: Tensor,
        branching_method: str,
        split_depth: Union[float, None] = None,
        offset: float = 0.,
        batch_size: Union[float, None] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """

    Constructs a tree that is nonuniform due to the clipping procedure. Clipping is only supported for modes in CROWN. BaB is now conducted in the order:

    * Bound nodes
    * Split nodes
    * Apply clipping to reduce volume of input domains

    :param func:                The implicit function to evaluate nodes. Should be a CROWN mode
    :param params:
    :param lower:               Lower input domain for batches
    :param upper:               Upper input domain for batches
    :param branching_method:    Specified branching heuristic to decide which dimension to split upon
    :param split_depth:         Desired split depth granularity of the kD tree
    :param offset:              Spec to verify against. Typically, 0.
    :param batch_size:          If not None, nodes are processed in batches
    :return:
    """

    d = lower.shape[-1]

    print(f"\n == CONSTRUCTING LEVELSET TREE")

    # Initialize datas
    node_lower = [lower]
    node_upper = [upper]
    node_type = []
    split_dim = []
    split_val = []
    N_curr_nodes = 1
    N_func_evals = 0
    N_total_nodes = 1
    ## Recursively build the tree
    i_split = 0
    n_splits = split_depth + 1  # 1 extra because last round doesn't split
    for i_split in range(n_splits):
        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = i_split + 1 == n_splits
        do_continue_splitting = not quit_next

        print(
            f"Uniform levelset tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")

        total_n_valid = 0
        og_lower = lower.clone()
        og_upper = upper.clone()
        lower, upper, out_node_type, out_split_dim, out_split_val = (
            construct_full_non_uniform_unknown_levelset_tree_iter(func, params, do_continue_splitting,
                                                                  og_lower, og_upper, i_split, branching_method,
                                                                  offset=offset, batch_size=batch_size))

        # as a sanity check, we can compare our results without clipping to the original uniform kd tree
        if DEBUG_NONUNIFORM_KDTREE:
            uni_result = (
                construct_full_uniform_unknown_levelset_tree_iter(func, params, do_continue_splitting,
                                                                  og_lower, og_upper, i_split,
                                                                  offset=offset, batch_size=batch_size))
            uni_node_lower_out, uni_node_upper_out, uni_node_types, uni_node_split_dim, uni_node_split_val = uni_result

            def _allclose_ignore_nan(tensor1, tensor2, rtol=1e-05, atol=1e-08, verbose=False):
                tensor1_mask = ~torch.isnan(tensor1)
                tensor2_mask = ~torch.isnan(tensor2)
                result = False
                if (tensor1_mask != tensor2_mask).any():
                    # consider nans equal
                    if verbose: print(f"NaN elements are not equal")
                    result = False
                else:
                    mask = tensor1_mask
                    result = torch.allclose(tensor1[mask], tensor2[mask], rtol=rtol, atol=atol)
                    if not result and verbose: print("Real elements are not equal")

                if not result and verbose:
                    print(f"tensor1:")
                    print(tensor1.detach().cpu().numpy())
                    print(f"tensor2:")
                    print(tensor2.detach().cpu().numpy())

                return result

            assert _allclose_ignore_nan(out_split_dim, uni_node_split_dim), "Node split dim does not match"
            assert _allclose_ignore_nan(lower, uni_node_lower_out,
                                        verbose=i_split == 8), "Node lower out does not match"
            assert _allclose_ignore_nan(upper, uni_node_upper_out,
                                        verbose=i_split == 8), "Node upper out does not match"
            assert _allclose_ignore_nan(out_node_type, uni_node_types), "Node type out does not match"
            assert _allclose_ignore_nan(out_split_val, uni_node_split_val), "Node split value does not match"

        node_lower.append(lower)
        node_upper.append(upper)
        node_type.append(out_node_type)
        split_dim.append(out_split_dim)
        split_val.append(out_split_val)
        N_curr_nodes = torch.logical_not(lower[:, 0].isnan()).sum()

        N_total_nodes += N_curr_nodes
        if quit_next:
            break

    node_lower = torch.cat(node_lower)
    node_upper = torch.cat(node_upper)
    node_type = torch.cat(node_type)
    split_dim = torch.cat(split_dim)
    split_val = torch.cat(split_val)

    # for key in out_dict:
    #     print(key, out_dict[key][:10])
    print("Total number of nodes evaluated: ", N_total_nodes)
    return node_lower, node_upper, node_type, split_dim, split_val

def construct_non_uniform_unknown_levelset_tree_iter(
        func, params, continue_splitting,
        node_valid, node_lower, node_upper,
        ib, out_valid, out_lower, out_upper, out_n_valid,
        finished_interior_lower, finished_interior_upper, N_finished_interior,
        finished_exterior_lower, finished_exterior_upper, N_finished_exterior,
        branching_method='naive', offset=0.
        ):
    N_in = node_lower.shape[0]
    d = node_lower.shape[-1]

    def eval_batch_of_nodes(
            lower,
            upper,
            continue_splitting=False
    ):
        """

        :param lower:               The input lower bounds of the nodes
        :param upper:               The input upper bounds of the nodes
        :param continue_splitting:  True if this evaluation should split the nodes after bounding them
        :return:                    Returns the (verification status, split dimension, split value, new_lb, new_ub)
        """

        # bounds the nodes
        ret = func.classify_box(params, lower, upper, offset)
        node_type, crown_dict = ret
        node_type = node_type.squeeze(-1)

        unverified_mask = node_type == SIGN_UNKNOWN  # only interested in domains that are unverified from this point

        new_split_values = torch.full_like(node_type, torch.nan, device=node_type.device)
        new_lower = torch.full((node_type.shape[0] * 2, 3), torch.nan, device=node_type.device)
        new_upper = torch.full((node_type.shape[0] * 2, 3), torch.nan, device=node_type.device)

        # call a branching heuristic to select dimensions to split upon
        dm_lb = crown_dict.get('dm_lb')
        lA = crown_dict.get('lA')
        lbias = crown_dict.get('lbias')
        new_split_idx = input_split_branching(dm_lb, lower, upper, lA, torch.full_like(lbias, offset),
                                              branching_method=branching_method).squeeze(1)
        # dm_lb dimension: (batches, 1)
        # lA dimension: (batches, 1, 3)
        # lbias dimension: (batches, 1)
        # split_idx dimension: (batches, 1)

        if not continue_splitting:
            # nodes only need to be bounded, not split
            return node_type, new_split_idx.float(), new_split_values.float(), lower, upper, unverified_mask

        if not unverified_mask.any():
            return node_type.float(), new_split_idx.float(), new_split_values.float(), new_lower.float(), new_upper.float(), unverified_mask.repeat(2)

        # filter out domains that are already verified
        dm_lb = dm_lb[unverified_mask]
        lA = lA[unverified_mask]
        lbias = lbias[unverified_mask]
        split_idx = new_split_idx[unverified_mask]

        # FIXME: This should be removed but is here for debugging purposes
        check_lbias = deconstruct_lbias(lower[unverified_mask], upper[unverified_mask], lA, dm_lb)
        assert torch.allclose(lbias, check_lbias), "Deconstructed lbias does not match CROWN lbias"

        # produce the splits on the domains
        split_result = kd_split(lower[unverified_mask], upper[unverified_mask], split_idx)
        _newA_lower, _newA_upper, _newB_lower, _newB_upper, split_values = split_result

        # stack lower and upper bounds to work with clip_domains
        total_new_lower = torch.cat((_newA_lower, _newB_lower)).float()
        total_new_upper = torch.cat((_newA_upper, _newB_upper)).float()

        new_split_values[unverified_mask] = split_values

        # repeat along the batch dimension for newly split nodes
        lA = lA.repeat(2, 1, 1)
        lbias = lbias.repeat(2, 1)

        # apply clipping to reduce volume of input domain
        clip_results = clip_domains(
            total_new_lower, total_new_upper, offset, lA, None, lbias, True)
        clipped_total_new_lower, clipped_total_new_upper = clip_results
        # clipped_total_new_lower/upper dimension: (N_out, n)

        if DEBUG_NONUNIFORM_KDTREE:
            clipped_total_new_lower = total_new_lower
            clipped_total_new_upper = total_new_upper

        # Clipping may cause x_L > x_U which means that the domain is verified,
        # and domains that are verified have NaN bounds
        clip_verified = (clipped_total_new_lower > clipped_total_new_upper).any(1)
        clipped_total_new_lower[clip_verified] = torch.nan
        clipped_total_new_upper[clip_verified] = torch.nan

        # put clipped domains back into our new domain list
        new_lower[unverified_mask.repeat(2)] = clipped_total_new_lower.float()
        new_upper[unverified_mask.repeat(2)] = clipped_total_new_upper.float()
        new_valid = unverified_mask.repeat(2)
        new_valid[unverified_mask.repeat(2)] = torch.logical_not(clip_verified)
        return node_type.float(), new_split_idx.float(), new_split_values.float(), new_lower.float(), new_upper.float(), new_valid

    total_samples = node_valid.shape[0]
    newA_lower = torch.empty((total_samples, 3))
    newB_lower = torch.empty((total_samples, 3))
    newA_upper = torch.empty((total_samples, 3))
    newB_upper = torch.empty((total_samples, 3))
    newA_valid = []
    newB_valid = []
    if isinstance(func, CrownImplicitFunction):
        batch_size_per_iteration = 256
        total_samples = node_lower.shape[0]
        node_types = torch.empty((total_samples,))
        node_split_dim = torch.empty((total_samples,))
        for start_idx in range(0, total_samples, batch_size_per_iteration):
            end_idx = min(start_idx + batch_size_per_iteration, total_samples)
            node_types[start_idx:end_idx], node_split_dim[start_idx:end_idx], _, new_node_lower_batch, new_node_upper_batch, new_valid_batch \
                = eval_batch_of_nodes(node_lower[start_idx:end_idx], node_upper[start_idx:end_idx], continue_splitting=continue_splitting)
            if continue_splitting:
                diff_idx = end_idx - start_idx
                newA_lower[start_idx:end_idx] = new_node_lower_batch[:diff_idx]
                newB_lower[start_idx:end_idx] = new_node_lower_batch[diff_idx:]
                newA_upper[start_idx:end_idx] = new_node_upper_batch[:diff_idx]
                newB_upper[start_idx:end_idx] = new_node_upper_batch[diff_idx:]
                newA_valid.append(new_valid_batch[:diff_idx])
                newB_valid.append(new_valid_batch[diff_idx:])
        if continue_splitting:
            # newA_lower = torch.cat(newA_lower)
            # newB_lower = torch.cat(newB_lower)
            # newA_upper = torch.cat(newA_upper)
            # newB_upper = torch.cat(newB_upper)
            newA_valid = torch.cat(newA_valid)
            newB_valid = torch.cat(newB_valid)
    else:
        # evaluate the function inside nodes
        raise NotImplementedError

    # if requested, write out interior nodes
    if finished_interior_lower is not None:
        out_mask = torch.logical_and(node_valid, node_types == SIGN_NEGATIVE)
        out_inds = utils.enumerate_mask(out_mask) + N_finished_interior
        mask = (- 1 < out_inds) & (out_inds < finished_interior_lower.shape[0])
        out_inds = out_inds[mask]
        node_interior_lower = node_lower[mask].float()
        node_interior_upper = node_upper[mask].float()
        # finished_interior_lower = finished_interior_lower.at[out_inds,:].set(node_lower, mode='drop')
        # finished_interior_upper = finished_interior_upper.at[out_inds,:].set(node_upper, mode='drop')
        finished_interior_lower[out_inds, :] = node_interior_lower
        finished_interior_upper[out_inds, :] = node_interior_upper
        N_finished_interior += torch.sum(out_mask)

    # if requested, write out exterior nodes
    if finished_exterior_lower is not None:
        out_mask = torch.logical_and(node_valid, node_types == SIGN_POSITIVE)
        out_inds = utils.enumerate_mask(out_mask) + N_finished_exterior
        mask = (- 1 < out_inds) & (out_inds < finished_exterior_lower.shape[0])
        out_inds = out_inds[mask]
        node_exterior_lower = node_lower[mask].float()
        node_exterior_upper = node_upper[mask].float()
        # finished_exterior_lower = finished_exterior_lower.at[out_inds,:].set(node_lower, mode='drop')
        # finished_exterior_upper = finished_exterior_upper.at[out_inds,:].set(node_upper, mode='drop')
        finished_exterior_lower[out_inds, :] = node_exterior_lower
        finished_exterior_upper[out_inds, :] = node_exterior_upper
        N_finished_exterior += torch.sum(out_mask)

    # split the unknown nodes to children
    # (if split_children is False this will just not create any children at all)
    split_mask = utils.logical_and_all([node_valid, node_types == SIGN_UNKNOWN])
    N_new = torch.sum(split_mask)  # each split leads to two children (for a total of 2*N_new)
    ## now actually build the child nodes
    if continue_splitting:

        # extents of the new child nodes along each split dimension
        # new_lower = node_lower
        # new_upper = node_upper
        # new_mid = 0.5 * (new_lower + new_upper)
        # new_coord_mask = torch.arange(3)[None, :] == node_split_dim[:, None]
        # newA_lower = new_lower
        # newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
        # newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
        # newB_upper = new_upper
        #
        # # concatenate the new children to form output arrays
        new_node_valid = torch.cat((newA_valid, newB_valid))
        node_lower = torch.cat((newA_lower, newB_lower))
        node_upper = torch.cat((newA_upper, newB_upper))
        # node_valid = torch.logical_and(new_node_valid, torch.logical_not(node_lower.isnan().any(dim=1)))
        node_valid = torch.logical_and(split_mask.repeat(2), new_node_valid)
        # node_valid = torch.cat((split_mask, split_mask))
        # node_lower = new_node_lower
        # node_upper = new_node_upper
        # new_N_valid = 2 * N_new
        new_N_valid = torch.sum(node_valid)
        outL = out_valid.shape[1]


    else:
        node_valid = torch.logical_and(node_valid, node_types == SIGN_UNKNOWN)
        new_N_valid = torch.sum(node_valid)
        outL = node_valid.shape[0]

    # write the result in to arrays
    # utils.printarr(out_valid, node_valid, out_lower, node_lower, out_upper, node_upper)
    out_valid[ib, :outL] = node_valid
    out_lower[ib, :outL, :] = node_lower
    out_upper[ib, :outL, :] = node_upper
    out_n_valid = out_n_valid + new_N_valid

    return out_valid, out_lower, out_upper, out_n_valid, \
        finished_interior_lower, finished_interior_upper, N_finished_interior, \
        finished_exterior_lower, finished_exterior_upper, N_finished_exterior


def construct_non_uniform_unknown_levelset_tree(func, params, lower, upper, node_terminate_thresh=None, split_depth=None,
                                            compress_after=False, with_childern=False, with_interior_nodes=False,
                                            with_exterior_nodes=False, offset=0., batch_process_size=2048):
    # Validate input
    # ASSUMPTION: all of our bucket sizes larger than batch_process_size must be divisible by batch_process_size
    for b in bucket_sizes:
        if b > batch_process_size and (b // batch_process_size) * batch_process_size != b:
            raise ValueError(
                f"batch_process_size must be a factor of our bucket sizes, is not a factor of {b} (try a power of 2)")
    if node_terminate_thresh is None and split_depth is None:
        raise ValueError("must specify at least one of node_terminate_thresh or split_depth as a terminating condition")
    if node_terminate_thresh is None:
        node_terminate_thresh = 9999999999

    d = lower.shape[-1]
    B = batch_process_size

    print(f"\n == CONSTRUCTING LEVELSET TREE")

    # Initialize data
    node_lower = lower[None, :]
    node_upper = upper[None, :]
    node_valid = torch.ones((1,), dtype=torch.bool)
    N_curr_nodes = 1
    finished_interior_lower = torch.zeros((batch_process_size, 3)) if with_interior_nodes else None
    finished_interior_upper = torch.zeros((batch_process_size, 3)) if with_interior_nodes else None
    N_finished_interior = 0
    finished_exterior_lower = torch.zeros((batch_process_size, 3)) if with_exterior_nodes else None
    finished_exterior_upper = torch.zeros((batch_process_size, 3)) if with_exterior_nodes else None
    N_finished_exterior = 0
    N_func_evals = 0

    ## Recursively build the tree
    i_split = 0
    n_splits = 99999999 if split_depth is None else split_depth + 1  # 1 extra because last round doesn't split
    for i_split in range(n_splits):
        # Reshape in to batches of size <= B
        init_bucket_size = node_lower.shape[0]
        this_b = min(B, init_bucket_size)
        N_func_evals += node_lower.shape[0]
        # utils.printarr(node_valid, node_lower, node_upper)
        node_valid = torch.reshape(node_valid, (-1, this_b))
        node_lower = torch.reshape(node_lower, (-1, this_b, d))
        node_upper = torch.reshape(node_upper, (-1, this_b, d))
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(
            N_curr_nodes / this_b))  # only the batches which are occupied (since valid nodes are densely packed at the start)

        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = (N_curr_nodes >= node_terminate_thresh) or i_split + 1 == n_splits
        do_continue_splitting = not quit_next

        print(
            f"Uniform levelset tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  bucket size: {init_bucket_size}  batch size: {this_b}  number of batches: {nb}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")

        # enlarge the finished nodes if needed
        if with_interior_nodes:
            while finished_interior_lower.shape[0] - N_finished_interior < N_curr_nodes:
                finished_interior_lower = utils.resize_array_axis(finished_interior_lower,
                                                                  2 * finished_interior_lower.shape[0])
                finished_interior_upper = utils.resize_array_axis(finished_interior_upper,
                                                                  2 * finished_interior_upper.shape[0])
        if with_exterior_nodes:
            while finished_exterior_lower.shape[0] - N_finished_exterior < N_curr_nodes:
                finished_exterior_lower = utils.resize_array_axis(finished_exterior_lower,
                                                                  2 * finished_exterior_lower.shape[0])
                finished_exterior_upper = utils.resize_array_axis(finished_exterior_upper,
                                                                  2 * finished_exterior_upper.shape[0])

        # map over the batches
        out_valid = torch.zeros((nb, 2 * this_b), dtype=torch.bool)
        out_lower = torch.zeros((nb, 2 * this_b, 3))
        out_upper = torch.zeros((nb, 2 * this_b, 3))
        total_n_valid = 0
        for ib in range(n_occ):
            out_valid, out_lower, out_upper, total_n_valid, \
                finished_interior_lower, finished_interior_upper, N_finished_interior, \
                finished_exterior_lower, finished_exterior_upper, N_finished_exterior, \
                = \
                construct_non_uniform_unknown_levelset_tree_iter(func, params, do_continue_splitting, \
                                                             node_valid[ib, ...], node_lower[ib, ...],
                                                             node_upper[ib, ...], \
                                                             ib, out_valid, out_lower, out_upper, total_n_valid, \
                                                             finished_interior_lower, finished_interior_upper,
                                                             N_finished_interior, \
                                                             finished_exterior_lower, finished_exterior_upper,
                                                             N_finished_exterior, \
                                                             offset=offset)

        node_valid = out_valid
        node_lower = out_lower
        node_upper = out_upper
        N_curr_nodes = total_n_valid

        # flatten back out
        node_valid = torch.reshape(node_valid, (-1,))
        node_lower = torch.reshape(node_lower, (-1, d))


        node_upper = torch.reshape(node_upper, (-1, d))

        # Compactify and rebucket arrays
        target_bucket_size = get_next_bucket_size(total_n_valid)
        node_valid, N_curr_nodes, node_lower, node_upper = compactify_and_rebucket_arrays(node_valid,
                                                                                          target_bucket_size,
                                                                                          node_lower, node_upper)

        if quit_next:
            break


    # pack the output in to a dict to support optional outputs
    out_dict = {
        'unknown_node_valid': node_valid,
        'unknown_node_lower': node_lower,
        'unknown_node_upper': node_upper,
    }

    if with_interior_nodes:
        out_dict['interior_node_valid'] = torch.arange(finished_interior_lower.shape[0]) < N_finished_interior
        out_dict['interior_node_lower'] = finished_interior_lower
        out_dict['interior_node_upper'] = finished_interior_upper

    if with_exterior_nodes:
        out_dict['exterior_node_valid'] = torch.arange(finished_exterior_lower.shape[0]) < N_finished_exterior
        out_dict['exterior_node_lower'] = finished_exterior_lower
        out_dict['exterior_node_upper'] = finished_exterior_upper


    return out_dict


def sample_surface_iter(func, params, n_samples_per_round, width, rngkey,
                        u_node_valid, u_node_lower, u_node_upper,
                        found_sample_points, found_start_ind):
    ## Generate sample points in the nodes

    # pick which node to sample from
    # (uses the fact valid nodes will always be first N)
    n_node_valid = torch.sum(u_node_valid)
    rngkey, subkey = split_generator(rngkey)
    node_ind = torch.randint(low=0, high=n_node_valid, size=(n_samples_per_round,), generator=subkey)
    # fetch node data
    samp_lower = u_node_lower[node_ind, :]
    samp_upper = u_node_upper[node_ind, :]

    # sample points uniformly within the nodes
    rngkey, subkey = split_generator(rngkey)
    samp_pos = (samp_upper - samp_lower) * torch.rand((n_samples_per_round, 3), generator=subkey) + samp_lower

    # evaluate the function and reject samples outside of the specified width
    samp_val = vmap(partial(func, params))(samp_pos)
    samp_valid = torch.abs(samp_val) < width

    # append these new samples on to the output array 
    n_samp_valid = torch.sum(samp_valid)
    valid_inds = utils.enumerate_mask(samp_valid, fill_value=found_sample_points.shape[0])
    valid_inds_loc = valid_inds + found_start_ind
    mask = (- 1 < valid_inds_loc) & (valid_inds_loc < found_sample_points.shape[0])
    valid_inds_loc = valid_inds_loc[mask]
    samp_pos = samp_pos[mask, :]
    found_sample_points[valid_inds_loc, :] = samp_pos
    found_start_ind = torch.minimum(found_start_ind + n_samp_valid, torch.as_tensor(found_sample_points.shape[0]))

    return found_sample_points, found_start_ind


def sample_surface(func, params, lower, upper, n_samples, width, rngkey, n_node_thresh=4096):
    '''
    - Build tree over levelset (rather than a usual 0 bound, it needs to use += width, so we know for sure that the sample region is contained in unknown cells)
    - Uniformly rejection sample from the unknown cells 
    '''

    # Build a tree over the valid nodes
    # By definition returned nodes are all SIGN_UNKNOWN, and all the same size
    out_dict = construct_uniform_unknown_levelset_tree(func, params, lower, upper, node_terminate_thresh=n_node_thresh,
                                                       offset=width)
    node_valid = out_dict['unknown_node_valid']
    node_lower = out_dict['unknown_node_lower']
    node_upper = out_dict['unknown_node_upper']
    # sample from the unknown nodes until we get enough samples
    n_samples_per_round = min(3 * n_samples, 100000)  # enough that we usually finish in one round
    found_sample_points = torch.zeros((n_samples, 3))
    found_start_ind = 0
    round_count = 0
    while True:
        round_count += 1

        rngkey, subkey = split_generator(rngkey)
        found_sample_points, found_start_ind = sample_surface_iter(func, params, n_samples_per_round, width, subkey,
                                                                   node_valid, node_lower, node_upper,
                                                                   found_sample_points, found_start_ind)

        # NOTE: assumes all nodes are same size

        if found_start_ind == n_samples:
            break

    func_with_params = partial(func, params)
    print((vmap(func_with_params)(found_sample_points) ** 2).sum().sqrt() / n_samples)
    return found_sample_points


# This is here for comparison to the tree-based one above
def sample_surface_uniform_iter(func, params, n_samples_per_round, width, rngkey,
                                lower, upper,
                                found_sample_points, found_start_ind):
    ## Generate sample points in the nodes

    # sample points uniformly within the nodes
    rngkey, subkey = split_generator(rngkey)
    samp_pos = (upper - lower) * torch.rand((n_samples_per_round, 3), generator=subkey) + lower

    # evaluate the function and reject samples outside of the specified width
    samp_val = vmap(partial(func, params))(samp_pos)
    samp_valid = torch.abs(samp_val) < width

    # append these new samples on to the output array 
    n_samp_valid = torch.sum(samp_valid)
    valid_inds = utils.enumerate_mask(samp_valid, fill_value=found_sample_points.shape[0])
    valid_inds_loc = valid_inds + found_start_ind
    mask = (- 1 < valid_inds_loc) & (valid_inds_loc < found_sample_points.shape[0])
    valid_inds_loc = valid_inds_loc[mask]
    samp_pos = samp_pos[mask, :]
    found_sample_points[valid_inds_loc, :] = samp_pos
    # found_sample_points = found_sample_points.at[valid_inds_loc,:].set(samp_pos, mode='drop', indices_are_sorted=True)
    found_start_ind = torch.minimum(found_start_ind + n_samp_valid, torch.as_tensor(found_sample_points.shape[0]))

    return found_sample_points, found_start_ind


def sample_surface_uniform(func, params, lower, upper, n_samples, width, rngkey):
    # sample from the unknown nodes until we get enough samples
    n_samples_per_round = min(10 * n_samples, 100000)
    found_sample_points = torch.zeros((n_samples, 3))
    found_start_ind = 0
    round_count = 0
    while True:
        round_count += 1

        rngkey, subkey = split_generator(rngkey)
        found_sample_points, found_start_ind = sample_surface_uniform_iter(func, params, n_samples_per_round, width,
                                                                           subkey, lower, upper, found_sample_points,
                                                                           found_start_ind)

        if found_start_ind == n_samples:
            break

    return found_sample_points


def hierarchical_marching_cubes_extract_iter(func, params, mc_data, n_subcell_depth, node_valid, node_lower, node_upper,
                                             tri_pos_out, n_out_written):
    # run the extraction routine
    tri_verts, tri_valid = vmap(
        partial(extract_cell.extract_triangles_from_subcells, func, params, mc_data, n_subcell_depth))(node_lower,
                                                                                                       node_upper)
    tri_valid = torch.logical_and(tri_valid, node_valid[:, None])

    # flatten out the generated triangles
    tri_verts = torch.reshape(tri_verts, (-1, 3, 3))
    tri_valid = torch.reshape(tri_valid, (-1,))

    # write the result
    out_inds = utils.enumerate_mask(tri_valid, fill_value=tri_pos_out.shape[0])
    out_inds += n_out_written
    mask = (-1 < out_inds) & (out_inds < tri_pos_out.shape[0])
    out_inds = out_inds[mask]
    # tri_pos_out = tri_pos_out.at[out_inds,:,:].set(tri_verts, mode='drop')
    tri_verts = tri_verts[mask]
    tri_pos_out[out_inds, :, :] = tri_verts
    n_out_written += torch.sum(tri_valid)

    return tri_pos_out, n_out_written


def hierarchical_marching_cubes(func, params, lower, upper, depth, enable_clipping, n_subcell_depth=3,
                                extract_batch_max_tri_out=1000000):
    # Build a tree over the isosurface
    # By definition returned nodes are all SIGN_UNKNOWN, and all the same size
    if enable_clipping:
        out_dict = construct_non_uniform_unknown_levelset_tree(func, params, lower, upper,
                                                     split_depth=3 * (depth - n_subcell_depth))
    else:
        out_dict = construct_uniform_unknown_levelset_tree(func, params, lower, upper,
                                                               split_depth=3 * (depth - n_subcell_depth))

    node_valid = out_dict['unknown_node_valid']
    node_lower = out_dict['unknown_node_lower']
    node_upper = out_dict['unknown_node_upper']

    # fetch the extraction data
    mc_data = extract_cell.get_mc_data()

    # Extract triangle from the valid nodes (do it in batches in case there are a lot)
    extract_batch_size = extract_batch_max_tri_out // (5 * (2 ** n_subcell_depth) ** 3)
    extract_batch_size = get_next_bucket_size(extract_batch_size)
    N_cell = node_valid.shape[0]
    N_valid = int(torch.sum(node_valid))
    n_out_written = 0
    tri_pos_out = torch.zeros((1, 3, 3))

    init_bucket_size = node_lower.shape[0]
    this_b = min(extract_batch_size, init_bucket_size)
    node_valid = torch.reshape(node_valid, (-1, this_b))
    node_lower = torch.reshape(node_lower, (-1, this_b, 3))
    node_upper = torch.reshape(node_upper, (-1, this_b, 3))
    nb = node_lower.shape[0]
    print(N_valid, this_b)
    n_occ = int(math.ceil(
        N_valid / this_b))  # only the batches which are occupied (since valid nodes are densely packed at the start)
    max_tri_round = this_b * 5 * (2 ** n_subcell_depth) ** 3
    for ib in range(n_occ):

        print(f"Extract iter {ib} / {n_occ}. max_tri_round: {max_tri_round} n_out_written: {n_out_written}")

        # expand the output array only lazily as needed
        while (tri_pos_out.shape[0] - n_out_written < max_tri_round):
            tri_pos_out = utils.resize_array_axis(tri_pos_out, 2 * tri_pos_out.shape[0])

        tri_pos_out, n_out_written = hierarchical_marching_cubes_extract_iter(func, params, mc_data, n_subcell_depth,
                                                                              node_valid[ib, ...], node_lower[ib, ...],
                                                                              node_upper[ib, ...], tri_pos_out,
                                                                              n_out_written)

    # clip the result triangles
    # TODO bucket and mask here? need to if we want this in a JIT loop
    tri_pos_out = tri_pos_out[:n_out_written, :]

    return tri_pos_out


def find_any_intersection_iter(
        func_tuple, params_tuple, eps,
        node_lower, node_upper, N_curr_nodes,
        viz_nodes=False
):
    # N_curr_nodes --> the first N nodes are valid

    '''
    Algorithm:
    process_node():

        for each func:

            detect func in node as one of 4 categories (
                positive: (strictly positive via interval bound)
                negative: (strictly negative via interval bound)
                unknown:  (cannot bound via interval bound) 
                near_surface: (there is a sign change in +- eps/2 of node center and node width < eps)
                (near surface has highest precedence if it applies)

        if >= 2 are (negative or near_surface):
            return found intersection! 

        if >= 2 are (negative or unknown):
            recurse on subnodes

        else: 
            return exit -- no intersection
    '''

    N_in = node_lower.shape[0]
    d = node_lower.shape[-1]
    node_valid = torch.arange(node_lower.shape[0]) < N_curr_nodes

    if len(func_tuple) != 2:
        raise ValueError("intersection supports pairwise only as written")
    funcA = func_tuple[0]
    funcB = func_tuple[1]
    paramsA = params_tuple[0]
    paramsB = params_tuple[1]

    # the side of a cube such that all points are within `eps` of each other
    eps_cube_width = eps / torch.sqrt(torch.as_tensor(3))

    def process_node(valid, lower, upper):

        intersection_count = 0  # nodes which definitely have an intersection in this cell
        possible_intersection_count = 0  # nodes which _might_ have an intersection in this cell

        # intersection details
        found_intersection = False
        intersection_loc = torch.tensor((-777., -777., -777.))

        # Node geometry
        node_width = torch.max(upper - lower)
        node_split_dim = torch.argmax(upper - lower, dim=-1)
        node_is_small = node_width < eps_cube_width
        node_center = 0.5 * (lower + upper)
        sample_offsets = torch.concatenate((torch.zeros((1, d)), torch.eye(d), -torch.eye(d)), dim=0)
        sample_pts = node_center[None, :] + eps_cube_width * sample_offsets

        # classify the box
        node_interval_typeA = funcA.classify_box(paramsA, lower, upper)
        node_interval_typeB = funcB.classify_box(paramsB, lower, upper)

        # test the sample points nearby for convergence checking
        sample_valsA = vmap(partial(funcA, paramsA))(sample_pts)
        sample_valsB = vmap(partial(funcB, paramsB))(sample_pts)

        all_same_signA = utils.all_same_sign(sample_valsA)
        all_same_signB = utils.all_same_sign(sample_valsB)
        is_near_surfaceA = torch.logical_and(node_is_small, ~all_same_signA)
        is_near_surfaceB = torch.logical_and(node_is_small, ~all_same_signB)

        ## test if we definitely found an intersection
        # indices = (sample_valsA < 0).nonzero().view(-1)
        any_neg_indA = torch.argmin(sample_valsA, keepdim=True)
        print(any_neg_indA)
        any_is_negA = torch.any(sample_valsA < 0)
        any_neg_locA = sample_pts[any_neg_indA, :]
        indices = torch.nonzero(sample_valsB < 0).view(-1)
        any_neg_indB = indices[0].item() if indices.numel() > 0 else 0
        any_is_negB = sample_valsB[any_neg_indB] < 0
        any_neg_locB = sample_pts[any_neg_indB, :]
        have_near_neg_samples = utils.logical_and_all((node_is_small, any_is_negA, any_is_negB))
        found_intersection = torch.logical_or(found_intersection, have_near_neg_samples)
        intersection_loc = torch.where(have_near_neg_samples, 0.5 * (any_neg_locA + any_neg_locB), intersection_loc)

        # if some sample point is inside of both funcs
        # (no need to do anything for both SIGN_NEGATIVE, it will be caught by this)
        # (this criterion is tested second because we prefer it, it gives a point stricly inside instead
        # of in the blurry eps converged region)
        sample_both_neg = torch.logical_and(sample_valsA < 0, sample_valsB < 0)
        indices = torch.nonzero(sample_both_neg < 0).view(-1)
        both_neg_ind = indices[0].item() if indices.numel() > 0 else 0
        have_sample_both_neg = sample_both_neg[both_neg_ind]
        sample_both_neg_loc = sample_pts[both_neg_ind, :]
        found_intersection = torch.logical_or(found_intersection, have_sample_both_neg)
        intersection_loc = torch.where(have_sample_both_neg, sample_both_neg_loc, intersection_loc)

        ## test if we need to keep searching
        could_be_insideA = torch.logical_or(
            node_interval_typeA == SIGN_NEGATIVE,
            torch.logical_and(node_interval_typeA == SIGN_UNKNOWN, ~is_near_surfaceA)
        )
        could_be_insideB = torch.logical_or(
            node_interval_typeB == SIGN_NEGATIVE,
            torch.logical_and(node_interval_typeB == SIGN_UNKNOWN, ~is_near_surfaceB)
        )

        needs_subdiv = utils.logical_and_all((could_be_insideA, could_be_insideB, valid))
        found_intersection = torch.logical_and(found_intersection, valid)

        return found_intersection, intersection_loc, needs_subdiv, node_split_dim

    # evaluate the function inside nodes
    has_intersection, intersection_loc, needs_subdiv, node_split_dim = \
        vmap(process_node)(node_valid, node_lower, node_upper)

    # if there was any intersection, pull out its data right away
    indices = torch.nonzero(has_intersection < 0).view(-1)
    int_ind = indices[0].item() if indices.numel() > 0 else 0
    found_int = has_intersection[int_ind]
    found_int_loc = intersection_loc[int_ind, :]

    # no need to keep processing anything if we found an intersection
    needs_subdiv = torch.logical_and(needs_subdiv, ~found_int)

    if viz_nodes:
        # if requested, dump out all of the nodes that were searched, for visualization
        viz_nodes_mask = torch.logical_and(node_valid, ~needs_subdiv)
    else:
        viz_nodes_mask = None

    N_needs_sudiv = torch.sum(needs_subdiv)

    # get rid of all nodes that are not getting subdivided and compactify the rest
    N_new = torch.sum(needs_subdiv)  # before split, after splitting there will be 2*N_new nodes
    # compact_inds = torch.nonzero(needs_subdiv, size=needs_subdiv.shape[0], fill_value=INVALID_IND)[0]
    compact_inds = needs_subdiv.nonzero(as_tuple=True)[0]
    num_needed = needs_subdiv.shape[0]
    if compact_inds.numel() < num_needed:
        padding = torch.full((num_needed - compact_inds.numel(),), INVALID_IND, device=needs_subdiv.device,
                             dtype=torch.long)
        compact_inds = torch.cat((compact_inds, padding))
    node_lower = node_lower.at[compact_inds, :].get(mode='fill', fill_value=-777.)
    node_upper = node_upper.at[compact_inds, :].get(mode='fill', fill_value=-777.)
    node_split_dim = node_split_dim.at[compact_inds].get(mode='fill', fill_value=-777)

    ## now actually build the child nodes

    # extents of the new child nodes along each split dimension
    new_lower = node_lower
    new_upper = node_upper
    new_mid = 0.5 * (new_lower + new_upper)
    new_coord_mask = torch.arange(3)[None, :] == node_split_dim[:, None]
    newA_lower = new_lower
    newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
    newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
    newB_upper = new_upper

    # write the new children in to the arrays (this will have twice the size of the input)
    node_lower = utils.interleave_arrays((newA_lower, newB_lower))
    node_upper = utils.interleave_arrays((newA_upper, newB_upper))

    return node_lower, node_upper, 2 * N_new, found_int, 1, 2, found_int_loc, viz_nodes_mask


def find_any_intersection(func_tuple, params_tuple, lower, upper, eps, viz_nodes=False):
    d = lower.shape[-1]

    print(f"\n == SEARCHING FOR INTERSECTION")

    # Initialize data
    node_lower = lower[None, :]
    node_upper = upper[None, :]
    N_curr_nodes = 1
    N_nodes_processed = 0  # only actually nodes, does not count fake ones due to bucketing
    N_bucket_nodes_processed = 0  # includes real and fake nodes due to bucketing

    if viz_nodes:
        viz_nodes_lower = torch.zeros((0, 3))
        viz_nodes_upper = torch.zeros((0, 3))
        viz_nodes_type = torch.zeros((0,), dtype=int)
    else:
        viz_nodes_lower = None
        viz_nodes_upper = None
        viz_nodes_type = None

    ## Recursively search the space
    split_round = 0
    while (True):

        ## Call the function which does all the actual work
        # (the node_lower/node_upper arrays that come out are twice the size due to splits)

        N_nodes_processed += N_curr_nodes
        N_bucket_nodes_processed += node_lower.shape[0]

        print(
            f"Intersection search depth {split_round}. Searching {N_curr_nodes} nodes (bucket: {node_lower.shape[0]})")

        if (viz_nodes):
            # if requested, save visualization data
            # (back these up so we can use them below)
            node_lower_prev = node_lower
            node_upper_prev = node_upper
        node_lower, node_upper, N_curr_nodes, found_int, found_int_A, found_int_B, found_int_loc, viz_mask = \
            find_any_intersection_iter(func_tuple, params_tuple, eps, node_lower, node_upper, N_curr_nodes, viz_nodes)
        if (viz_nodes):
            # if requested, save visualization data
            node_lower_save = node_lower_prev[viz_mask, :]
            node_upper_save = node_upper_prev[viz_mask, :]

            # classify the nodes
            def process_node(lower, upper):
                node_interval_typeA = func_tuple[0].classify_box(params_tuple[0], lower, upper)
                node_interval_typeB = func_tuple[1].classify_box(params_tuple[1], lower, upper)
                type_count = 0
                type_count = torch.where(node_interval_typeA == SIGN_POSITIVE, 1, type_count)
                type_count = torch.where(node_interval_typeB == SIGN_POSITIVE, 2, type_count)
                return type_count

            node_type_save = vmap(process_node)(node_lower_save, node_upper_save)

            viz_nodes_lower = torch.concatenate((viz_nodes_lower, node_lower_save))
            viz_nodes_upper = torch.concatenate((viz_nodes_upper, node_upper_save))
            viz_nodes_type = torch.concatenate((viz_nodes_type, node_type_save))

        N_curr_nodes = int(N_curr_nodes)

        # quit because we found an intersection
        if found_int:
            print(
                f"Found intersection between funcs {found_int_A},{found_int_B} at {found_int_loc}. Processed {N_nodes_processed} nodes ({N_bucket_nodes_processed}).")
            if viz_nodes:
                return found_int, found_int_A, found_int_B, found_int_loc, viz_nodes_lower, viz_nodes_upper, viz_nodes_type
            else:
                return found_int, found_int_A, found_int_B, found_int_loc

        # quit because there can be no intersection
        if N_curr_nodes == 0:
            print(f"No intersection detected. Processed {N_nodes_processed} nodes ({N_bucket_nodes_processed}).")
            if viz_nodes:
                return False, 0, 0, torch.tensor(
                    (-777., -777., -777.)), viz_nodes_lower, viz_nodes_upper, viz_nodes_type
            else:
                return False, 0, 0, torch.tensor((-777., -777., -777.))

        # if the current nodes would fit in a smaller array, put them there
        new_bucket_size = get_next_bucket_size(N_curr_nodes)
        curr_bucket_size = node_lower.shape[0]
        if new_bucket_size < curr_bucket_size:
            node_lower = node_lower[:new_bucket_size, :]
            node_upper = node_upper[:new_bucket_size, :]

        split_round += 1


def closest_point_iter(func, params,
                       query_points, query_min_dist, query_min_loc,
                       work_query_id, work_node_lower, work_node_upper, work_stack_top,
                       eps, batch_process_size):
    # basic strategy:
    # - pop work items off queue
    # - discard inside/outside nodes
    # - discard nodes further than min dist
    # - for any node which spans, compute minimum distance
    # - reduce over minimum
    # - if node dist == min, set min location
    # - recurse into big nodes, push back on stack

    ## pop off some work to do
    B = batch_process_size
    Q = query_points.shape[0]
    d = query_points.shape[-1]
    pop_ind = torch.maximum(torch.as_tensor(work_stack_top - B), torch.as_tensor(0))
    # batch_query_id   = jax.lax.dynamic_slice_in_dim(work_query_id,   pop_ind, B)
    # batch_node_lower = jax.lax.dynamic_slice_in_dim(work_node_lower, pop_ind, B)
    # batch_node_upper = jax.lax.dynamic_slice_in_dim(work_node_upper, pop_ind, B)
    batch_query_id = work_query_id.narrow(0, pop_ind, B)
    batch_node_lower = work_node_lower.narrow(0, pop_ind, B)
    batch_node_upper = work_node_upper.narrow(0, pop_ind, B)
    batch_query_loc = query_points[batch_query_id, :]
    batch_query_min_dist = query_min_dist[batch_query_id]

    batch_valid = torch.arange(B) < work_stack_top
    work_stack_top = pop_ind

    import math
    eps_cube_width = eps / math.sqrt(d)

    d = work_node_lower.shape[-1]

    # process each node, computing closest point data
    def process_one(valid, query_id, lower, upper, query_loc, query_min_dist):
        # compute an upper bound on the distance to any point in the node
        node_width = torch.max(upper - lower)
        node_center = 0.5 * (lower + upper)
        node_center_dist_offset = torch.sqrt(
            torch.sum(torch.square(upper - lower)))  # maximum distance from the center to any point in the node
        max_dist_to_point_in_node = geometry.norm(query_loc - node_center) + node_center_dist_offset  # could be tighter
        nearest_point_in_node = torch.clip(query_loc, min=lower, max=upper)
        min_dist_to_point_in_node = geometry.norm(query_loc - node_center)
        node_split_dim = torch.argmax(upper - lower, dim=-1)
        is_small = torch.as_tensor(node_width < eps_cube_width)
        sample_offsets = torch.cat((torch.zeros((1, d)), torch.eye(d), -torch.eye(d)), dim=0)  # [7,3]
        sample_pts = node_center[None, :] + (upper - lower)[None, :] * sample_offsets
        # classify the box
        node_interval_type = func.classify_box(params, lower.unsqueeze(0), upper.unsqueeze(0))
        is_outside = torch.logical_or(node_interval_type == SIGN_NEGATIVE, node_interval_type == SIGN_POSITIVE)

        # test the sample points nearby for convergence checking
        sample_vals = vmap(partial(func, params))(sample_pts)
        spans_surface = torch.logical_and(torch.logical_not(utils.all_same_sign(sample_vals)), valid)

        # compute outputs
        this_closest_point_dist = torch.where(spans_surface, max_dist_to_point_in_node, float('inf'))
        needs_subdiv = utils.logical_and_all((valid, torch.logical_not(is_outside), torch.logical_not(is_small),
                                              min_dist_to_point_in_node < query_min_dist))

        return needs_subdiv, this_closest_point_dist, node_center, node_split_dim

    # batch_needs_subdiv, batch_this_closest_point_dist, batch_node_center, batch_node_split_dim = \
    #     vmap(process_one)(batch_valid, batch_query_id, batch_node_lower, batch_node_upper, batch_query_loc, batch_query_min_dist)

    batch_needs_subdiv, batch_this_closest_point_dist, batch_node_center, batch_node_split_dim = [], [], [], []

    for i in range(batch_valid.shape[0]):
        out_tup = process_one(batch_valid[i], batch_query_id[i], batch_node_lower[i], batch_node_upper[i],
                              batch_query_loc[i], batch_query_min_dist[i])
        batch_needs_subdiv.append(out_tup[0])
        batch_this_closest_point_dist.append(out_tup[1])
        batch_node_center.append(out_tup[2])
        batch_node_split_dim.append(out_tup[3])

    batch_needs_subdiv = torch.tensor(batch_needs_subdiv)
    batch_this_closest_point_dist = torch.tensor(batch_this_closest_point_dist)
    batch_node_center = torch.stack(batch_node_center)
    batch_node_split_dim = torch.tensor(batch_node_split_dim)
    print("out shapes: ", batch_needs_subdiv.shape, batch_this_closest_point_dist.shape, batch_node_center.shape,
          batch_node_split_dim.shape)

    # set any newly found closest values
    # query_min_dist = query_min_dist.at[batch_query_id].min(batch_this_closest_point_dist)
    query_min_dist[batch_query_id] = torch.min(query_min_dist[batch_query_id], batch_this_closest_point_dist)
    batch_query_new_min_dist = query_min_dist[batch_query_id]
    batch_has_new_min = (batch_this_closest_point_dist == batch_query_new_min_dist)
    batch_target_inds = torch.where(batch_has_new_min, batch_query_id, Q)
    # query_min_loc = query_min_loc.at[batch_target_inds,:].set(batch_node_center, mode='drop')
    mask = (-1 < batch_target_inds) & (batch_target_inds < query_min_loc.shape[0])
    valid_inds = batch_target_inds[mask]
    valid_node_center = batch_node_center[mask]
    query_min_loc[valid_inds] = valid_node_center

    # compactify the nodes which need to be subdivided
    N_new = torch.sum(batch_needs_subdiv)  # before split, after splitting there will be 2*N_new nodes
    nonzero_inds = torch.nonzero(batch_needs_subdiv, as_tuple=False).squeeze()
    compact_inds = torch.full((batch_needs_subdiv.shape[0],), INVALID_IND, dtype=torch.long)
    compact_inds[nonzero_inds] = nonzero_inds  # 1D
    # compact_inds = torch.nonzero(batch_needs_subdiv, size=batch_needs_subdiv.shape[0], fill_value=INVALID_IND)[0]
    # batch_node_lower = batch_node_lower.at[compact_inds,:].get(mode='fill', fill_value=-777.)
    # batch_node_upper = batch_node_upper.at[compact_inds,:].get(mode='fill', fill_value=-777.)
    # batch_query_id = batch_query_id.at[compact_inds].get(mode='fill', fill_value=-777.)
    # batch_node_split_dim = batch_node_split_dim.at[compact_inds].get(mode='fill', fill_value=-777)

    mask = (-1 < compact_inds) & (compact_inds < batch_node_lower.shape[0])
    valid_inds = compact_inds[mask]
    temp_bnl = torch.full((compact_inds.size(0), batch_node_lower.size(1)), -777.)
    temp_bnl[mask] = batch_node_lower[valid_inds]
    batch_node_lower = temp_bnl

    mask = (-1 < compact_inds) & (compact_inds < batch_node_upper.shape[0])
    valid_inds = compact_inds[mask]
    temp_bnu = torch.full((compact_inds.size(0), batch_node_upper.size(1)), -777.)
    temp_bnu[mask] = batch_node_upper[valid_inds]
    batch_node_upper = temp_bnu

    mask = (-1 < compact_inds) & (compact_inds < batch_query_id.shape[0])
    valid_inds = compact_inds[mask]
    temp_bqi = torch.full((compact_inds.size(0),), -777., dtype=batch_query_id.dtype)
    temp_bqi[mask] = batch_query_id[valid_inds]
    batch_query_id = temp_bqi

    mask = (-1 < compact_inds) & (compact_inds < batch_node_split_dim.shape[0])
    valid_inds = compact_inds[mask]
    temp_bnsd = torch.full((compact_inds.size(0),), -777, dtype=batch_node_split_dim.dtype)
    temp_bnsd[mask] = batch_node_split_dim[valid_inds]
    batch_node_split_dim = temp_bnsd

    ## now actually build the child nodes

    # extents of the new child nodes along each split dimension
    new_batch_lower = batch_node_lower
    new_batch_upper = batch_node_upper
    new_batch_mid = 0.5 * (new_batch_lower + new_batch_upper)
    new_batch_coord_mask = torch.arange(3)[None, :] == batch_node_split_dim[:, None]
    newA_lower = new_batch_lower
    newA_upper = torch.where(new_batch_coord_mask, new_batch_mid, new_batch_upper)
    newB_lower = torch.where(new_batch_coord_mask, new_batch_mid, new_batch_lower)
    newB_upper = new_batch_upper

    # write the new children in to the arrays (this will have twice the size of the input)
    new_node_lower = utils.interleave_arrays((newA_lower, newB_lower))
    new_node_upper = utils.interleave_arrays((newA_upper, newB_upper))
    new_node_query_id = utils.interleave_arrays((batch_query_id, batch_query_id))

    # TODO is this guaranteed to update in place like at[] does?
    # work_query_id   = jax.lax.dynamic_update_slice_in_dim(work_query_id, new_node_query_id, pop_ind, axis=0)
    # work_node_lower = jax.lax.dynamic_update_slice_in_dim(work_node_lower, new_node_lower, pop_ind, axis=0)
    # work_node_upper = jax.lax.dynamic_update_slice_in_dim(work_node_upper, new_node_upper, pop_ind, axis=0)
    work_query_id[pop_ind:pop_ind + new_node_query_id.size(0)] = new_node_query_id
    work_node_lower[pop_ind:pop_ind + new_node_lower.size(0), :] = new_node_lower
    work_node_upper[pop_ind:pop_ind + new_node_upper.size(0), :] = new_node_upper
    work_stack_top = work_stack_top + 2 * N_new

    return query_min_dist, query_min_loc, \
        work_query_id, work_node_lower, work_node_upper, work_stack_top,


def closest_point(func, params, lower, upper, query_points, eps=0.001, batch_process_size=2048):
    # working data
    B = batch_process_size
    Q = query_points.shape[0]
    work_node_lower = lower.unsqueeze(0).repeat(Q, 1)
    work_node_upper = upper.unsqueeze(0).repeat(Q, 1)
    work_query_id = torch.arange(Q)
    query_min_dist = torch.full((Q,), float('inf'))
    query_min_loc = torch.full((Q, 3), -777.)
    work_stack_top = query_points.shape[0]

    i_round = 0
    while work_stack_top > 0:

        # Ensure there is enough room on the stack (at most we will add B new entries if every node is subdivided)
        while work_node_lower.shape[0] < (work_stack_top + B):
            N = work_node_lower.shape[0]
            N_new = max(2 * N, 8 * B)
            work_node_lower = utils.resize_array_axis(work_node_lower, N_new)
            work_node_upper = utils.resize_array_axis(work_node_upper, N_new)
            work_query_id = utils.resize_array_axis(work_query_id, N_new)

        query_min_dist, query_min_loc, \
            work_query_id, work_node_lower, work_node_upper, work_stack_top = \
            closest_point_iter(func, params,
                               query_points, query_min_dist, query_min_loc,
                               work_query_id, work_node_lower, work_node_upper, work_stack_top,
                               eps=eps, batch_process_size=batch_process_size)

        work_stack_top = int(work_stack_top)

        i_round += 1

    return query_min_dist, query_min_loc


def bulk_properties_sample_mass(func, params, node_valid, node_lower, node_upper, n_samples, rngkey):
    # pick which node to sample from
    # (uses the fact valid nodes will always be first N)
    n_node_valid = torch.sum(node_valid)
    rngkey, subkey = split_generator(rngkey)
    node_ind = torch.randint(low=0, high=n_node_valid.item(), size=(n_samples,), generator=subkey)
    # fetch node data
    samp_lower = node_lower[node_ind, :]
    samp_upper = node_upper[node_ind, :]

    # sample points uniformly within the nodes
    rngkey, subkey = split_generator(rngkey)
    samp_pos = (samp_upper - samp_lower) * torch.rand((n_samples, 3), generator=subkey) + samp_lower

    # evaluate the function and reject samples outside of the specified width
    samp_val = vmap(partial(func, params))(samp_pos)
    samp_valid = samp_val < 0.

    # compute the contribution to mass and centroid
    areas = torch.prod(node_upper - node_lower, dim=-1)
    total_area = torch.sum(torch.where(node_valid, areas, 0.))
    vol_per_sample = total_area / n_samples

    mass = vol_per_sample * torch.sum(samp_valid)
    centroid = vol_per_sample * torch.sum(torch.where(samp_valid[:, None], samp_pos, 0.), dim=0)

    return mass, centroid


def bulk_properties(func, params, lower, upper, rngkey, n_expand=int(1e4), n_sample=int(1e6)):
    out_dict = construct_uniform_unknown_levelset_tree(func, params, lower, upper, with_interior_nodes=True,
                                                       node_terminate_thresh=n_expand)
    node_valid = out_dict['unknown_node_valid']
    node_lower = out_dict['unknown_node_lower']
    node_upper = out_dict['unknown_node_upper']
    interior_node_valid = out_dict['interior_node_valid']
    interior_node_lower = out_dict['interior_node_lower']
    interior_node_upper = out_dict['interior_node_upper']

    # Compute mass and centroid for this demo
    def compute_bulk_mass(lower, upper):
        mass = torch.prod(upper - lower)
        c = 0.5 * (lower + upper)
        return mass, mass * c

    mass_interior, centroid_interior = vmap(compute_bulk_mass)(interior_node_lower, interior_node_upper)

    mass_interior = torch.sum(torch.where(interior_node_valid, mass_interior, 0.))
    centroid_interior = torch.sum(torch.where(interior_node_valid[:, None], centroid_interior, 0.), dim=0)

    rngkey, subkey = split_generator(rngkey)
    mass_boundary, centroid_boundary = bulk_properties_sample_mass(func, params, node_valid, node_lower, node_upper,
                                                                   n_sample, subkey)

    mass = mass_interior + mass_boundary
    centroid = centroid_interior + centroid_boundary
    centroid = centroid / mass

    return mass, centroid


def generate_tree_viz_nodes_simple(node_lower, node_upper, shrink_factor=0.05):
    print("Generating viz nodes")

    # (global shrink)
    min_width = torch.min(node_upper - node_lower)
    shrink = shrink_factor * min_width
    node_lower += shrink
    node_upper -= shrink

    # Construct the 8 indices for each cell
    v0 = torch.stack((node_lower[:, 0], node_lower[:, 1], node_lower[:, 2]), dim=-1)
    v1 = torch.stack((node_upper[:, 0], node_lower[:, 1], node_lower[:, 2]), dim=-1)
    v2 = torch.stack((node_upper[:, 0], node_upper[:, 1], node_lower[:, 2]), dim=-1)
    v3 = torch.stack((node_lower[:, 0], node_upper[:, 1], node_lower[:, 2]), dim=-1)
    v4 = torch.stack((node_lower[:, 0], node_lower[:, 1], node_upper[:, 2]), dim=-1)
    v5 = torch.stack((node_upper[:, 0], node_lower[:, 1], node_upper[:, 2]), dim=-1)
    v6 = torch.stack((node_upper[:, 0], node_upper[:, 1], node_upper[:, 2]), dim=-1)
    v7 = torch.stack((node_lower[:, 0], node_upper[:, 1], node_upper[:, 2]), dim=-1)
    vs = [v0, v1, v2, v3, v4, v5, v6, v7]

    # (local shrink)
    centers = 0.5 * (node_lower + node_upper)
    for i in range(8):
        vs[i] = (1. - shrink_factor) * vs[i] + shrink_factor * centers

    verts = utils.interleave_arrays(vs)

    # Construct the index array
    inds = torch.arange(8 * v0.shape[0]).reshape((-1, 8))

    return verts, inds
