import jax

from functools import partial
import math

import numpy as np
from functorch import vmap
import utils
from bucketing import *
import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
import extract_cell
import geometry
import torch

from crown import CrownImplicitFunction

INVALID_IND = 2**30
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        node_type = func.classify_box(params, lower, upper, offset=offset)

        # use the largest length along any dimension as the split policy
        worst_dim = torch.argmax(upper-lower, dim=-1)
        return node_type, worst_dim

    def eval_batch_of_nodes(lower, upper):
        node_type = func.classify_box(params, lower, upper, offset=offset).squeeze(-1)
        worst_dim = torch.argmax(upper-lower, dim=-1)
        return node_type, worst_dim

    # print(type(func))
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

    # print("N_in: ", N_in)
    num_node_valid = node_valid.sum() - 1
    # first_node_invalid = min(num_node_valid + 1, N_in - 1)
    # print(num_node_valid, first_node_invalid)
    # print("node_valid: ", node_valid.sum())
    # print("negative nodes: ", torch.logical_and(node_valid, node_types == SIGN_NEGATIVE).int().sum())
    # print("positive nodes: ", torch.logical_and(node_valid, node_types == SIGN_POSITIVE).int().sum())
    # print("unknown nodes: ", torch.logical_and(node_valid, node_types == SIGN_UNKNOWN).int().sum())
    # print("node_lower: ", node_lower[num_node_valid], node_lower[first_node_invalid])
    # print("node_upper: ", node_upper[num_node_valid], node_upper[first_node_invalid])
    # print("node types: ", node_types[num_node_valid], node_types[first_node_invalid])
    # print("node split dim: ", node_split_dim[num_node_valid], node_split_dim[first_node_invalid])

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
    # print("node valid before split_mask: ", node_valid)
    split_mask = utils.logical_and_all([node_valid, node_types == SIGN_UNKNOWN])
    # print("split_mask: ", split_mask)
    N_new = torch.sum(split_mask) # each split leads to two children (for a total of 2*N_new)
    ## now actually build the child nodes
    if continue_splitting:

        # extents of the new child nodes along each split dimension
        new_lower = node_lower
        new_upper = node_upper
        new_mid = 0.5 * (new_lower + new_upper)
        new_coord_mask = torch.arange(3)[None,:] == node_split_dim[:,None]
        newA_lower = new_lower
        newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
        newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
        newB_upper = new_upper
        # print(newA_lower[new_coord_mask], newA_upper[new_coord_mask])
        # print(newB_lower[new_coord_mask], newB_upper[new_coord_mask])

        # concatenate the new children to form output arrays
        node_valid = torch.cat((split_mask, split_mask))
        node_lower = torch.cat((newA_lower, newB_lower))
        node_upper = torch.cat((newA_upper, newB_upper))
        new_N_valid = 2*N_new
        outL = out_valid.shape[1]


        # print("new_mid: ", new_mid[num_node_valid], new_mid[first_node_invalid])
    else:
        node_valid = torch.logical_and(node_valid, node_types == SIGN_UNKNOWN)
        new_N_valid = torch.sum(node_valid)
        outL = node_valid.shape[0]



    # write the result in to arrays
    # utils.printarr(out_valid, node_valid, out_lower, node_lower, out_upper, node_upper)
    out_valid[ib,:outL] = node_valid
    out_lower[ib,:outL,:] = node_lower
    out_upper[ib,:outL,:] = node_upper
    out_n_valid = out_n_valid + new_N_valid

    return out_valid, out_lower, out_upper, out_n_valid, \
           finished_interior_lower, finished_interior_upper, N_finished_interior, \
           finished_exterior_lower, finished_exterior_upper, N_finished_exterior


def construct_uniform_unknown_levelset_tree(func, params, lower, upper, node_terminate_thresh=None, split_depth=None, compress_after=False, with_childern=False, with_interior_nodes=False, with_exterior_nodes=False, offset=0., batch_process_size=2048):
       
    # Validate input
    # ASSUMPTION: all of our bucket sizes larger than batch_process_size must be divisible by batch_process_size
    for b in bucket_sizes:
        if b > batch_process_size and (b//batch_process_size)*batch_process_size != b:
            raise ValueError(f"batch_process_size must be a factor of our bucket sizes, is not a factor of {b} (try a power of 2)")
    if node_terminate_thresh is None and split_depth is None:
        raise ValueError("must specify at least one of node_terminate_thresh or split_depth as a terminating condition")
    if node_terminate_thresh is None:
        node_terminate_thresh = 9999999999

    d = lower.shape[-1]
    B = batch_process_size

    print(f"\n == CONSTRUCTING LEVELSET TREE")
    # print(f"  node thresh: {n_node_thresh}")n_node_thresh

    # Initialize data
    node_lower = lower[None,:]
    node_upper = upper[None,:]
    node_valid = torch.ones((1,), dtype=torch.bool)
    N_curr_nodes = 1
    finished_interior_lower = torch.zeros((batch_process_size,3)) if with_interior_nodes else None
    finished_interior_upper = torch.zeros((batch_process_size,3)) if with_interior_nodes else None
    N_finished_interior = 0
    finished_exterior_lower = torch.zeros((batch_process_size,3)) if with_exterior_nodes else None
    finished_exterior_upper = torch.zeros((batch_process_size,3)) if with_exterior_nodes else None
    N_finished_exterior = 0
    N_func_evals = 0

    ## Recursively build the tree
    i_split = 0
    n_splits = 99999999 if split_depth is None else split_depth+1 # 1 extra because last round doesn't split
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
        n_occ = int(math.ceil(N_curr_nodes / this_b)) # only the batches which are occupied (since valid nodes are densely packed at the start)

        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = (N_curr_nodes >= node_terminate_thresh) or i_split+1 == n_splits
        do_continue_splitting = not quit_next

        print(f"Uniform levelset tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  bucket size: {init_bucket_size}  batch size: {this_b}  number of batches: {nb}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")

        # enlarge the finished nodes if needed
        if with_interior_nodes:
            while finished_interior_lower.shape[0] - N_finished_interior < N_curr_nodes:
                finished_interior_lower = utils.resize_array_axis(finished_interior_lower, 2*finished_interior_lower.shape[0])
                finished_interior_upper = utils.resize_array_axis(finished_interior_upper, 2*finished_interior_upper.shape[0])
        if with_exterior_nodes:
            while finished_exterior_lower.shape[0] - N_finished_exterior < N_curr_nodes:
                finished_exterior_lower = utils.resize_array_axis(finished_exterior_lower, 2*finished_exterior_lower.shape[0])
                finished_exterior_upper = utils.resize_array_axis(finished_exterior_upper, 2*finished_exterior_upper.shape[0])

        # map over the batches
        out_valid = torch.zeros((nb, 2*this_b), dtype=torch.bool)
        out_lower = torch.zeros((nb, 2*this_b, 3))
        out_upper = torch.zeros((nb, 2*this_b, 3))
        total_n_valid = 0
        for ib in range(n_occ): 
            out_valid, out_lower, out_upper, total_n_valid, \
            finished_interior_lower, finished_interior_upper, N_finished_interior, \
            finished_exterior_lower, finished_exterior_upper, N_finished_exterior, \
            = \
            construct_uniform_unknown_levelset_tree_iter(func, params, do_continue_splitting, \
                    node_valid[ib,...], node_lower[ib,...], node_upper[ib,...], \
                    ib, out_valid, out_lower, out_upper, total_n_valid, \
                    finished_interior_lower, finished_interior_upper, N_finished_interior, \
                    finished_exterior_lower, finished_exterior_upper, N_finished_exterior, \
                    offset=offset)
            # print(out_valid.sum())

        node_valid = out_valid
        node_lower = out_lower
        node_upper = out_upper
        # print("N_curr_nodes: ", N_curr_nodes)
        # print("n_occ: ", n_occ)
        # print("total_n_valid: ", total_n_valid)
        N_curr_nodes = total_n_valid

        # flatten back out
        node_valid = torch.reshape(node_valid, (-1,))
        node_lower = torch.reshape(node_lower, (-1, d))


        node_upper = torch.reshape(node_upper, (-1, d))

        # Compactify and rebucket arrays
        target_bucket_size = get_next_bucket_size(total_n_valid)
        node_valid, N_curr_nodes, node_lower, node_upper = compactify_and_rebucket_arrays(node_valid, target_bucket_size, node_lower, node_upper)

        if quit_next:
            break


    # pack the output in to a dict to support optional outputs
    out_dict = {
            'unknown_node_valid' : node_valid,
            'unknown_node_lower' : node_lower,
            'unknown_node_upper' : node_upper,
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


def construct_uniform_unknown_levelset_tree_iter_new(
        func, params, continue_splitting,
        node_lower, node_upper,
        split_level,
        offset=0.
):
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
        node_type = func.classify_box(params, lower, upper, offset=offset).squeeze(-1)
        worst_dim = torch.argmax(upper - lower, dim=-1)
        return node_type, worst_dim

    # print(type(func))
    node_types_temp = node_types[internal_node_mask]
    node_split_dim_temp = node_split_dim[internal_node_mask]
    if isinstance(func, CrownImplicitFunction):
        batch_size_per_iteration = 256
        total_samples = node_lower[internal_node_mask].shape[0]
        # print(total_samples)
        for start_idx in range(0, total_samples, batch_size_per_iteration):
            end_idx = min(start_idx + batch_size_per_iteration, total_samples)
            node_types_temp[start_idx:end_idx], node_split_dim_temp[start_idx:end_idx] \
                = eval_batch_of_nodes(node_lower[internal_node_mask][start_idx:end_idx], node_upper[internal_node_mask][start_idx:end_idx])

        node_types[internal_node_mask] = node_types_temp
        node_split_dim[internal_node_mask] = node_split_dim_temp
    else:
        # evaluate the function inside nodes
        # node_types[internal_node_mask], node_split_dim[internal_node_mask] = vmap(eval_one_node)(node_lower[internal_node_mask], node_upper[internal_node_mask])
        batch_size_per_iteration = 256
        total_samples = node_lower[internal_node_mask].shape[0]
        # print(total_samples)
        for start_idx in range(0, total_samples, batch_size_per_iteration):
            end_idx = min(start_idx + batch_size_per_iteration, total_samples)
            node_types_temp[start_idx:end_idx], node_split_dim_temp[start_idx:end_idx] \
                = vmap(eval_one_node)(node_lower[internal_node_mask][start_idx:end_idx],
                                      node_upper[internal_node_mask][start_idx:end_idx])

        node_types[internal_node_mask] = node_types_temp
        node_split_dim[internal_node_mask] = node_split_dim_temp

    # split the unknown nodes to children
    # (if split_children is False this will just not create any children at all)
    # print("node valid before split_mask: ", node_valid)
    # print("internal_node_mask: ", internal_node_mask)
    # print("node_types: ", node_types)
    split_mask = torch.logical_and(internal_node_mask, node_types == SIGN_UNKNOWN)
    # print("split_mask: ", split_mask)
    ## now actually build the child nodes
    if continue_splitting:

        # extents of the new child nodes along each split dimension
        new_lower = node_lower
        new_upper = node_upper
        new_mid = 0.5 * (new_lower + new_upper)
        new_coord_mask = torch.arange(3)[None, :] == node_split_dim[:, None]
        newA_lower = new_lower
        # print(new_coord_mask, new_mid, new_upper, split_mask.sum())
        newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
        newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
        newB_upper = new_upper

        # concatenate the new children to form output arrays
        node_lower_out[split_mask.repeat_interleave(2)] = torch.hstack((newA_lower[split_mask].unsqueeze(1), newB_lower[split_mask].unsqueeze(1))).view(-1, d)
        node_upper_out[split_mask.repeat_interleave(2)] = torch.hstack((newA_upper[split_mask].unsqueeze(1), newB_upper[split_mask].unsqueeze(1))).view(-1, d)
        # print("node_split_dim: ", node_split_dim)
        node_split_val[split_mask] = new_mid[torch.arange(split_mask.sum()), node_split_dim[split_mask].long()]


    return node_lower_out, node_upper_out, node_types, node_split_dim, node_split_val


def construct_uniform_unknown_levelset_tree_new(func, params, lower, upper, split_depth=None,
                                            offset=0.):

    d = lower.shape[-1]

    print(f"\n == CONSTRUCTING LEVELSET TREE")
    # print(f"  node thresh: {n_node_thresh}")n_node_thresh

    # Initialize data
    node_lower = [lower]
    node_upper = [upper]
    node_type = []
    split_dim = []
    split_val = []
    N_curr_nodes = 1
    N_func_evals = 0

    ## Recursively build the tree
    i_split = 0
    n_splits = split_depth + 1  # 1 extra because last round doesn't split
    for i_split in range(n_splits):
        # print(i_split)
        # Reshape in to batches of size <= B
        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = i_split + 1 == n_splits
        do_continue_splitting = not quit_next

        print(
            f"Uniform levelset tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")


        # map over the batches
        total_n_valid = 0
        lower, upper, out_node_type, out_split_dim, out_split_val = (
            construct_uniform_unknown_levelset_tree_iter_new(func, params, do_continue_splitting,
                                                             lower, upper, i_split, offset=offset))
        # print(lower.shape, upper.shape, out_node_type.shape, out_split_dim.shape, out_split_val.shape)
        node_lower.append(lower)
        node_upper.append(upper)
        node_type.append(out_node_type)
        split_dim.append(out_split_dim)
        split_val.append(out_split_val)
        # print("N_curr_nodes: ", N_curr_nodes)
        # print("n_occ: ", n_occ)
        # print("total_n_valid: ", total_n_valid)
        N_curr_nodes = torch.logical_not(lower[:, 0].isnan()).sum()


        if quit_next:
            break

    node_lower = torch.cat(node_lower)
    node_upper = torch.cat(node_upper)
    node_type = torch.cat(node_type)
    split_dim = torch.cat(split_dim)
    split_val = torch.cat(split_val)


    # for key in out_dict:
    #     print(key, out_dict[key][:10])
    return node_lower, node_upper, node_type, split_dim, split_val


# @partial(jax.jit, static_argnames=("func", "n_samples_per_round"), donate_argnums=(8,9))
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
    samp_lower = u_node_lower[node_ind,:]
    samp_upper = u_node_upper[node_ind,:]

    # sample points uniformly within the nodes
    rngkey, subkey = split_generator(rngkey)
    samp_pos = (samp_upper-samp_lower) * torch.rand((n_samples_per_round,3), generator=subkey) + samp_lower
   
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
    found_sample_points[valid_inds_loc,:] = samp_pos
    found_start_ind = torch.minimum(found_start_ind + n_samp_valid, torch.as_tensor(found_sample_points.shape[0]))

    return found_sample_points, found_start_ind
    

def sample_surface(func, params, lower, upper, n_samples, width, rngkey, n_node_thresh=4096):

    '''
    - Build tree over levelset (rather than a usual 0 bound, it needs to use += width, so we know for sure that the sample region is contained in unknown cells)
    - Uniformly rejection sample from the unknown cells 
    '''

    # print(f"Sample surface: building level set tree with at least {n_node_thresh} nodes")

    # Build a tree over the valid nodes
    # By definition returned nodes are all SIGN_UNKNOWN, and all the same size
    # print(f"sample_surface n_node_thresh {n_node_thresh}")
    out_dict = construct_uniform_unknown_levelset_tree(func, params, lower, upper, node_terminate_thresh=n_node_thresh, offset=width)
    node_valid = out_dict['unknown_node_valid']
    node_lower = out_dict['unknown_node_lower']
    node_upper = out_dict['unknown_node_upper']
    # sample from the unknown nodes until we get enough samples
    n_samples_per_round = min(3*n_samples, 100000) # enough that we usually finish in one round
    found_sample_points = torch.zeros((n_samples,3))
    found_start_ind = 0
    round_count = 0
    while True:
        round_count += 1

        # print(f"Have {found_start_ind} / {n_samples} samples. Performing sample round")

        rngkey, subkey = split_generator(rngkey)
        found_sample_points, found_start_ind = sample_surface_iter(func, params, n_samples_per_round, width, subkey,
                node_valid, node_lower, node_upper, found_sample_points, found_start_ind)

        # NOTE: assumes all nodes are same size

        if found_start_ind == n_samples:
            break

    # print(f"Done! Sampling took {round_count} rounds")
    func_with_params = partial(func, params)
    print((vmap(func_with_params)(found_sample_points)**2).sum().sqrt() / n_samples)
    return found_sample_points


# This is here for comparison to the tree-based one above
def sample_surface_uniform_iter(func, params, n_samples_per_round, width, rngkey,
        lower, upper,
        found_sample_points, found_start_ind):

    ## Generate sample points in the nodes

    # sample points uniformly within the nodes
    rngkey, subkey = split_generator(rngkey)
    samp_pos = (upper - lower) * torch.rand((n_samples_per_round,3), generator=subkey) + lower

    # evaluate the function and reject samples outside of the specified width
    samp_val = vmap(partial(func, params))(samp_pos)
    samp_valid = torch.abs(samp_val) < width

    # append these new samples on to the output array 
    n_samp_valid  = torch.sum(samp_valid)
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
    n_samples_per_round = min(10*n_samples, 100000)
    found_sample_points = torch.zeros((n_samples,3))
    found_start_ind = 0
    round_count = 0
    while True:
        round_count += 1

        rngkey, subkey = split_generator(rngkey)
        found_sample_points, found_start_ind = sample_surface_uniform_iter(func, params, n_samples_per_round, width, subkey, lower, upper, found_sample_points, found_start_ind)

        if found_start_ind == n_samples:
            break


    return found_sample_points

def hierarchical_marching_cubes_extract_iter(func, params, mc_data, n_subcell_depth, node_valid, node_lower, node_upper,tri_pos_out, n_out_written):

    # run the extraction routine
    tri_verts, tri_valid = vmap(partial(extract_cell.extract_triangles_from_subcells, func, params, mc_data, n_subcell_depth))(node_lower, node_upper)
    tri_valid = torch.logical_and(tri_valid, node_valid[:,None])
   
    # flatten out the generated triangles
    tri_verts = torch.reshape(tri_verts, (-1,3,3))
    tri_valid = torch.reshape(tri_valid, (-1,))

    # write the result
    out_inds = utils.enumerate_mask(tri_valid, fill_value=tri_pos_out.shape[0])
    out_inds += n_out_written
    mask = (-1 < out_inds) & (out_inds < tri_pos_out.shape[0])
    out_inds = out_inds[mask]
    # tri_pos_out = tri_pos_out.at[out_inds,:,:].set(tri_verts, mode='drop')
    tri_verts = tri_verts[mask]
    tri_pos_out[out_inds,:,:] = tri_verts
    n_out_written += torch.sum(tri_valid)

    return tri_pos_out, n_out_written

def hierarchical_marching_cubes(func, params, lower, upper, depth, n_subcell_depth=3, extract_batch_max_tri_out=1000000):
    # Build a tree over the isosurface
    # By definition returned nodes are all SIGN_UNKNOWN, and all the same size
    out_dict = construct_uniform_unknown_levelset_tree(func, params, lower, upper, split_depth=3*(depth-n_subcell_depth))
    node_valid = out_dict['unknown_node_valid']
    node_lower = out_dict['unknown_node_lower']
    node_upper = out_dict['unknown_node_upper']

    # fetch the extraction data
    mc_data = extract_cell.get_mc_data()

    # Extract triangle from the valid nodes (do it in batches in case there are a lot)
    extract_batch_size = extract_batch_max_tri_out // (5 * (2**n_subcell_depth)**3)
    extract_batch_size = get_next_bucket_size(extract_batch_size)
    N_cell = node_valid.shape[0]
    N_valid = int(torch.sum(node_valid))
    n_out_written = 0
    tri_pos_out = torch.zeros((1, 3, 3))

    init_bucket_size = node_lower.shape[0]
    # print(extract_batch_size, init_bucket_size)
    this_b = min(extract_batch_size, init_bucket_size)
    node_valid = torch.reshape(node_valid, (-1, this_b))
    node_lower = torch.reshape(node_lower, (-1, this_b, 3))
    node_upper = torch.reshape(node_upper, (-1, this_b, 3))
    nb = node_lower.shape[0]
    print(N_valid, this_b)
    n_occ = int(math.ceil(N_valid / this_b)) # only the batches which are occupied (since valid nodes are densely packed at the start)
    max_tri_round = this_b * 5 * (2**n_subcell_depth)**3
    for ib in range(n_occ):

        print(f"Extract iter {ib} / {n_occ}. max_tri_round: {max_tri_round} n_out_written: {n_out_written}")

        # expand the output array only lazily as needed
        while(tri_pos_out.shape[0] - n_out_written < max_tri_round):
            tri_pos_out = utils.resize_array_axis(tri_pos_out, 2*tri_pos_out.shape[0])
        
        tri_pos_out, n_out_written = hierarchical_marching_cubes_extract_iter(func, params, mc_data, n_subcell_depth, node_valid[ib,...], node_lower[ib,...], node_upper[ib,...], tri_pos_out, n_out_written)

    # clip the result triangles
    # TODO bucket and mask here? need to if we want this in a JIT loop
    tri_pos_out = tri_pos_out[:n_out_written,:]

    return tri_pos_out

   
def find_any_intersection_iter(
        func_tuple, params_tuple, eps,
        node_lower, node_upper, N_curr_nodes,
        viz_nodes = False
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

        intersection_count = 0 # nodes which definitely have an intersection in this cell
        possible_intersection_count = 0 # nodes which _might_ have an intersection in this cell

        # intersection details
        found_intersection = False
        intersection_loc = torch.tensor((-777., -777., -777.))

        # Node geometry
        node_width = torch.max(upper-lower)
        node_split_dim = torch.argmax(upper-lower, dim=-1)
        node_is_small = node_width < eps_cube_width
        node_center = 0.5 * (lower + upper)
        sample_offsets = torch.concatenate((torch.zeros((1,d)) ,torch.eye(d), -torch.eye(d)), dim=0)
        sample_pts = node_center[None,:] + eps_cube_width * sample_offsets

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
        any_neg_locA = sample_pts[any_neg_indA,:]
        indices = torch.nonzero(sample_valsB < 0).view(-1)
        any_neg_indB = indices[0].item() if indices.numel() > 0 else 0
        any_is_negB = sample_valsB[any_neg_indB] < 0
        any_neg_locB = sample_pts[any_neg_indB,:]
        have_near_neg_samples = utils.logical_and_all((node_is_small, any_is_negA, any_is_negB))
        found_intersection = torch.logical_or(found_intersection, have_near_neg_samples)
        intersection_loc = torch.where(have_near_neg_samples, 0.5 * (any_neg_locA + any_neg_locB), intersection_loc)
        
        # if some sample point is inside of both funcs
        # (no need to do anything for both SIGN_NEGATIVE, it will be caught by this)
        # (this criterion is tested second because we prefer it, it gives a point stricly inside instead
        # of in the blurry eps converged region)
        sample_both_neg = torch.logical_and(sample_valsA < 0 , sample_valsB < 0)
        indices = torch.nonzero(sample_both_neg < 0).view(-1)
        both_neg_ind = indices[0].item() if indices.numel() > 0 else 0
        have_sample_both_neg = sample_both_neg[both_neg_ind]
        sample_both_neg_loc = sample_pts[both_neg_ind,:]
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
    N_new = torch.sum(needs_subdiv) # before split, after splitting there will be 2*N_new nodes
    # compact_inds = torch.nonzero(needs_subdiv, size=needs_subdiv.shape[0], fill_value=INVALID_IND)[0]
    compact_inds = needs_subdiv.nonzero(as_tuple=True)[0]
    num_needed = needs_subdiv.shape[0]
    if compact_inds.numel() < num_needed:
        padding = torch.full((num_needed - compact_inds.numel(),), INVALID_IND, device=needs_subdiv.device,
                             dtype=torch.long)
        compact_inds = torch.cat((compact_inds, padding))
    node_lower = node_lower.at[compact_inds,:].get(mode='fill', fill_value=-777.)
    node_upper = node_upper.at[compact_inds,:].get(mode='fill', fill_value=-777.)
    node_split_dim = node_split_dim.at[compact_inds].get(mode='fill', fill_value=-777)

    ## now actually build the child nodes

    # extents of the new child nodes along each split dimension
    new_lower = node_lower
    new_upper = node_upper
    new_mid = 0.5 * (new_lower + new_upper)
    new_coord_mask = torch.arange(3)[None,:] == node_split_dim[:,None]
    newA_lower = new_lower
    newA_upper = torch.where(new_coord_mask, new_mid, new_upper)
    newB_lower = torch.where(new_coord_mask, new_mid, new_lower)
    newB_upper = new_upper

    # write the new children in to the arrays (this will have twice the size of the input)
    node_lower = utils.interleave_arrays((newA_lower, newB_lower))
    node_upper = utils.interleave_arrays((newA_upper, newB_upper))

    return node_lower, node_upper, 2*N_new, found_int, 1, 2, found_int_loc, viz_nodes_mask


def find_any_intersection(func_tuple, params_tuple, lower, upper, eps, viz_nodes=False):
    
    d = lower.shape[-1]

    print(f"\n == SEARCHING FOR INTERSECTION")
    # print(f"  max depth: {max_depth}")

    # Initialize data
    node_lower = lower[None,:]
    node_upper = upper[None,:]
    N_curr_nodes = 1
    N_nodes_processed = 0 # only actually nodes, does not count fake ones due to bucketing
    N_bucket_nodes_processed = 0      # includes real and fake nodes due to bucketing

    if viz_nodes:
        viz_nodes_lower = torch.zeros((0,3))
        viz_nodes_upper = torch.zeros((0,3))
        viz_nodes_type = torch.zeros((0,), dtype=int)
    else:
        viz_nodes_lower = None
        viz_nodes_upper = None
        viz_nodes_type = None

    ## Recursively search the space
    split_round = 0
    while(True):
        
        ## Call the function which does all the actual work
        # (the node_lower/node_upper arrays that come out are twice the size due to splits)

        N_nodes_processed += N_curr_nodes
        N_bucket_nodes_processed += node_lower.shape[0]

        print(f"Intersection search depth {split_round}. Searching {N_curr_nodes} nodes (bucket: {node_lower.shape[0]})")
        
        if(viz_nodes):
            # if requested, save visualization data
            # (back these up so we can use them below)
            node_lower_prev = node_lower
            node_upper_prev = node_upper
        # print(node_lower, node_upper)
        node_lower, node_upper, N_curr_nodes, found_int, found_int_A, found_int_B, found_int_loc, viz_mask = \
            find_any_intersection_iter(func_tuple, params_tuple, eps, node_lower, node_upper, N_curr_nodes, viz_nodes)
        # print(node_lower, node_upper)
        if(viz_nodes):
            # if requested, save visualization data
            node_lower_save = node_lower_prev[viz_mask,:]
            node_upper_save = node_upper_prev[viz_mask,:]

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
            print(f"Found intersection between funcs {found_int_A},{found_int_B} at {found_int_loc}. Processed {N_nodes_processed} nodes ({N_bucket_nodes_processed}).")
            if viz_nodes:
                return found_int, found_int_A, found_int_B, found_int_loc, viz_nodes_lower, viz_nodes_upper, viz_nodes_type
            else:
                return found_int, found_int_A, found_int_B, found_int_loc
        
        # quit because there can be no intersection
        if N_curr_nodes == 0:
            print(f"No intersection detected. Processed {N_nodes_processed} nodes ({N_bucket_nodes_processed}).")
            if viz_nodes:
                return False, 0, 0, torch.tensor((-777., -777., -777.)), viz_nodes_lower, viz_nodes_upper, viz_nodes_type
            else:
                return False, 0, 0, torch.tensor((-777., -777., -777.))

        # if the current nodes would fit in a smaller array, put them there
        new_bucket_size = get_next_bucket_size(N_curr_nodes)
        curr_bucket_size = node_lower.shape[0]
        if new_bucket_size < curr_bucket_size:
            node_lower = node_lower[:new_bucket_size,:]
            node_upper = node_upper[:new_bucket_size,:]

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
    pop_ind = torch.maximum(torch.as_tensor(work_stack_top-B),torch.as_tensor(0))
    # batch_query_id   = jax.lax.dynamic_slice_in_dim(work_query_id,   pop_ind, B)
    # batch_node_lower = jax.lax.dynamic_slice_in_dim(work_node_lower, pop_ind, B)
    # batch_node_upper = jax.lax.dynamic_slice_in_dim(work_node_upper, pop_ind, B)
    batch_query_id = work_query_id.narrow(0, pop_ind, B)
    batch_node_lower = work_node_lower.narrow(0, pop_ind, B)
    batch_node_upper = work_node_upper.narrow(0, pop_ind, B)
    batch_query_loc = query_points[batch_query_id,:]
    batch_query_min_dist = query_min_dist[batch_query_id]

    batch_valid = torch.arange(B) < work_stack_top
    work_stack_top = pop_ind

    import math
    eps_cube_width = eps / math.sqrt(d)

    d = work_node_lower.shape[-1]

    # process each node, computing closest point data
    def process_one(valid, query_id, lower, upper, query_loc, query_min_dist):

        # compute an upper bound on the distance to any point in the node
        node_width = torch.max(upper-lower)
        node_center = 0.5 * (lower + upper)
        node_center_dist_offset = torch.sqrt(torch.sum(torch.square(upper-lower))) # maximum distance from the center to any point in the node
        max_dist_to_point_in_node = geometry.norm(query_loc - node_center) + node_center_dist_offset # could be tighter
        nearest_point_in_node = torch.clip(query_loc, min=lower, max=upper)
        min_dist_to_point_in_node = geometry.norm(query_loc - node_center)
        node_split_dim = torch.argmax(upper-lower, dim=-1)
        is_small = torch.as_tensor(node_width < eps_cube_width)
        sample_offsets = torch.cat((torch.zeros((1,d)) ,torch.eye(d), -torch.eye(d)), dim=0) # [7,3]
        sample_pts = node_center[None,:] + (upper-lower)[None,:] * sample_offsets
        # classify the box
        node_interval_type = func.classify_box(params, lower.unsqueeze(0), upper.unsqueeze(0))
        is_outside = torch.logical_or(node_interval_type==SIGN_NEGATIVE, node_interval_type==SIGN_POSITIVE)

        # test the sample points nearby for convergence checking
        sample_vals = vmap(partial(func, params))(sample_pts)
        spans_surface = torch.logical_and(torch.logical_not(utils.all_same_sign(sample_vals)), valid)

        # compute outputs
        this_closest_point_dist = torch.where(spans_surface, max_dist_to_point_in_node, float('inf'))
        needs_subdiv = utils.logical_and_all((valid, torch.logical_not(is_outside), torch.logical_not(is_small), min_dist_to_point_in_node < query_min_dist))

        return needs_subdiv, this_closest_point_dist, node_center, node_split_dim

    # print("all shapes: ", batch_valid.shape, batch_query_id.shape, batch_node_lower.shape, batch_node_upper.shape, batch_query_loc.shape, batch_query_min_dist.shape)

    # batch_needs_subdiv, batch_this_closest_point_dist, batch_node_center, batch_node_split_dim = \
    #     vmap(process_one)(batch_valid, batch_query_id, batch_node_lower, batch_node_upper, batch_query_loc, batch_query_min_dist)


    batch_needs_subdiv, batch_this_closest_point_dist, batch_node_center, batch_node_split_dim = [], [], [], []

    for i in range(batch_valid.shape[0]):
        out_tup = process_one(batch_valid[i], batch_query_id[i], batch_node_lower[i], batch_node_upper[i], batch_query_loc[i], batch_query_min_dist[i])
        batch_needs_subdiv.append(out_tup[0])
        batch_this_closest_point_dist.append(out_tup[1])
        batch_node_center.append(out_tup[2])
        batch_node_split_dim.append(out_tup[3])


    batch_needs_subdiv = torch.tensor(batch_needs_subdiv)
    batch_this_closest_point_dist = torch.tensor(batch_this_closest_point_dist)
    batch_node_center = torch.stack(batch_node_center)
    batch_node_split_dim = torch.tensor(batch_node_split_dim)
    print("out shapes: ", batch_needs_subdiv.shape, batch_this_closest_point_dist.shape, batch_node_center.shape, batch_node_split_dim.shape)

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
    N_new = torch.sum(batch_needs_subdiv) # before split, after splitting there will be 2*N_new nodes
    nonzero_inds = torch.nonzero(batch_needs_subdiv, as_tuple=False).squeeze()
    compact_inds = torch.full((batch_needs_subdiv.shape[0],), INVALID_IND, dtype=torch.long)
    compact_inds[nonzero_inds] = nonzero_inds # 1D
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
    temp_bqi = torch.full((compact_inds.size(0), ), -777., dtype=batch_query_id.dtype)
    temp_bqi[mask] = batch_query_id[valid_inds]
    batch_query_id = temp_bqi

    mask = (-1 < compact_inds) & (compact_inds < batch_node_split_dim.shape[0])
    valid_inds = compact_inds[mask]
    temp_bnsd = torch.full((compact_inds.size(0), ), -777, dtype=batch_node_split_dim.dtype)
    temp_bnsd[mask] = batch_node_split_dim[valid_inds]
    batch_node_split_dim = temp_bnsd

    ## now actually build the child nodes

    # extents of the new child nodes along each split dimension
    new_batch_lower = batch_node_lower
    new_batch_upper = batch_node_upper
    new_batch_mid = 0.5 * (new_batch_lower + new_batch_upper)
    new_batch_coord_mask = torch.arange(3)[None,:] == batch_node_split_dim[:,None]
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
    work_stack_top = work_stack_top + 2*N_new

    # print("cpi return shapes: ", query_min_dist.shape, query_min_loc.shape, work_query_id.shape, work_node_lower.shape, work_node_upper.shape, work_stack_top.shape)
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
    query_min_loc = torch.full((Q,3), -777.)
    work_stack_top = query_points.shape[0]
    
    i_round = 0
    while work_stack_top > 0:

        # Ensure there is enough room on the stack (at most we will add B new entries if every node is subdivided)
        while work_node_lower.shape[0] < (work_stack_top + B):
            N = work_node_lower.shape[0]
            N_new = max(2*N, 8*B)
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
    samp_lower = node_lower[node_ind,:]
    samp_upper = node_upper[node_ind,:]

    # sample points uniformly within the nodes
    rngkey, subkey = split_generator(rngkey)
    samp_pos = (samp_upper - samp_lower) * torch.rand((n_samples,3), generator=subkey) + samp_lower

    # evaluate the function and reject samples outside of the specified width
    samp_val = vmap(partial(func, params))(samp_pos)
    samp_valid = samp_val < 0.

    # compute the contribution to mass and centroid
    areas = torch.prod(node_upper-node_lower, dim=-1)
    total_area = torch.sum(torch.where(node_valid, areas, 0.))
    vol_per_sample = total_area / n_samples
    
    mass = vol_per_sample*torch.sum(samp_valid)
    centroid = vol_per_sample*torch.sum(torch.where(samp_valid[:,None], samp_pos, 0.), dim=0)

    return mass, centroid

def bulk_properties(func, params, lower, upper, rngkey, n_expand=int(1e4), n_sample=int(1e6)):

    out_dict = construct_uniform_unknown_levelset_tree(func, params, lower, upper, with_interior_nodes=True, node_terminate_thresh=n_expand)
    node_valid = out_dict['unknown_node_valid']
    node_lower = out_dict['unknown_node_lower']
    node_upper = out_dict['unknown_node_upper']
    interior_node_valid = out_dict['interior_node_valid']
    interior_node_lower = out_dict['interior_node_lower']
    interior_node_upper = out_dict['interior_node_upper']

    # Compute mass and centroid for this demo
    def compute_bulk_mass(lower, upper):
        mass = torch.prod(upper-lower)
        c = 0.5 * (lower + upper)
        return mass, mass * c

    mass_interior, centroid_interior = vmap(compute_bulk_mass)(interior_node_lower, interior_node_upper)

    mass_interior = torch.sum(torch.where(interior_node_valid, mass_interior, 0.))
    centroid_interior = torch.sum(torch.where(interior_node_valid[:,None], centroid_interior, 0.), dim=0)

    rngkey, subkey = split_generator(rngkey)
    mass_boundary, centroid_boundary = bulk_properties_sample_mass(func, params, node_valid, node_lower, node_upper, n_sample, subkey)

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
    v0 = torch.stack((node_lower[:,0], node_lower[:,1], node_lower[:,2]), dim=-1)
    v1 = torch.stack((node_upper[:,0], node_lower[:,1], node_lower[:,2]), dim=-1)
    v2 = torch.stack((node_upper[:,0], node_upper[:,1], node_lower[:,2]), dim=-1)
    v3 = torch.stack((node_lower[:,0], node_upper[:,1], node_lower[:,2]), dim=-1)
    v4 = torch.stack((node_lower[:,0], node_lower[:,1], node_upper[:,2]), dim=-1)
    v5 = torch.stack((node_upper[:,0], node_lower[:,1], node_upper[:,2]), dim=-1)
    v6 = torch.stack((node_upper[:,0], node_upper[:,1], node_upper[:,2]), dim=-1)
    v7 = torch.stack((node_lower[:,0], node_upper[:,1], node_upper[:,2]), dim=-1)
    vs = [v0, v1, v2, v3, v4, v5, v6, v7]
    
    # (local shrink)
    centers = 0.5 * (node_lower + node_upper)
    for i in range(8):
        vs[i] = (1. - shrink_factor) * vs[i] + shrink_factor * centers
        
    verts = utils.interleave_arrays(vs)

    # Construct the index array
    inds = torch.arange(8*v0.shape[0]).reshape((-1,8))

    return verts, inds

