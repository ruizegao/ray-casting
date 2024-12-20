#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import time
import os
import inspect
from collections import OrderedDict, defaultdict
from contextlib import ExitStack

import torch
from torch import optim, Tensor
import numpy as np
from typing import Tuple, Union, Optional, final
from math import ceil
from warnings import warn

from .perturbations import PerturbationLpNorm
from .bounded_tensor import BoundedTensor
from .beta_crown import print_optimized_beta
from .cuda_utils import double2float
from .utils import reduction_sum, multi_spec_keep_func_all
from .opt_pruner import OptPruner
from .clip_autoLiRPA import clip_domains
from .hyperplane_volume_intersection import batch_cube_intersection_with_plane, finalize_plots, fill_plots, \
    get_rows_cols, batch_cube_intersection_with_plane_with_constraints, batch_estimate_volume
### preprocessor-hint: private-section-start
from .adam_element_lr import AdamElementLR
### preprocessor-hint: private-section-end

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule

# FIXME: This is here temporarily for debugging, be sure to remove later
lb_iter, ub_iter, displayed_lb, displayed_ub, displayed_graph = None, None, False, False, False
ax_graph, fig_graph_lower, fig_graph_upper = None, None, None
un_mask = None
import matplotlib.pyplot as plt
from .hyperplane_volume_intersection import get_cube_vertices, unit_cube_faces, to_numpy, plot_cube, plot_intersection, plot_plane
out_path = "/home/jorgejc2/Documents/Research/ray-casting/alpha_crown_planes/"

default_optimize_bound_args = {
    'enable_alpha_crown': True,  # Enable optimization of alpha.
    'enable_beta_crown': False,  # Enable beta split constraint.

    'apply_output_constraints_to': [],  # Enable optimization w.r.t. output constraints.
    'tighten_input_bounds': False,  # Don't tighten input bounds
    # If output constraints are activated, use only bounds computed with them.
    'best_of_oc_and_no_oc': False,
    'directly_optimize': [],  # No layer should be directly optimized
    'oc_lr': 0.1,  # learning rate for dualized output constraints
    'share_gammas': False,

    'iteration': 20,  # Number of alpha/beta optimization iterations.
    # Share some alpha variables to save memory at the cost of slightly
    # looser bounds.
    'use_shared_alpha': False,
    # Optimizer used for alpha and beta optimization.
    'optimizer': 'adam',
    # Save best results of alpha/beta/bounds during optimization.
    'keep_best': True,
    # Only optimize bounds of last layer during alpha/beta CROWN.
    'fix_interm_bounds': True,
    # Learning rate for the optimizable parameter alpha in alpha-CROWN.
    'lr_alpha': 0.5,
    # Learning rate for the optimizable parameter beta in beta-CROWN.
    'lr_beta': 0.05,
    'lr_cut_beta': 5e-3,  # Learning rate for optimizing cut betas.
    # Initial alpha variables by calling CROWN once.
    'init_alpha': True,
    'lr_coeffs': 0.01,  # Learning rate for coeffs for refinement
    # Layers to be refined, separated by commas.
    # -1 means preactivation before last activation.
    'intermediate_refinement_layers': [-1],
    # When batch size is not 1, this reduction function is applied to
    # reduce the bounds into a scalar.
    'loss_reduction_func': reduction_sum,
    # Criteria function of early stop.
    'stop_criterion_func': lambda x: False,
    # Learning rate decay factor during bounds optimization.
    'lr_decay': 0.98,
    # Number of iterations that we will start considering early stop
    # if tracking no improvement.
    'early_stop_patience': 10,
    # Start to save optimized best bounds
    # when current_iteration > int(iteration*start_save_best)
    'start_save_best': 0.5,
    # Use double fp (float64) at the last iteration in alpha/beta CROWN.
    'use_float64_in_last_iteration': False,
    # Prune verified domain within iteration.
    'pruning_in_iteration': False,
    # Percentage of the minimum domains that can apply pruning.
    'pruning_in_iteration_threshold': 0.2,
    # For specification that will output multiple bounds for one
    # property, we use this function to prune them.
    'multi_spec_keep_func': multi_spec_keep_func_all,
    # Use the newly fixed loss function. By default, it is set to False
    # for compatibility with existing use cases.
    ### preprocessor-hint: private-section-start
    # See https://github.com/Verified-Intelligence/Verifier_Development/issues/170.
    ### preprocessor-hint: private-section-end
    # Try to ensure that the parameters always match with the optimized bounds.
    'deterministic': False,
    'max_time': 1e9,
    # When using the total volume loss function, a batch of unverified nodes may be visually analzyed further for
    # debugging and sanity checks
    'save_loss_graphs': False,
    'perpendicular_multiplier': None,
    # Use custom loss function instead of default -l (or u).
    # This is not supported by "keep_best" yet.
    'use_custom_loss': False,
    # use original loss function for some number of iterations then use custom loss function
    'swap_loss_iter': 0,
    # When both bound_upepr and bound_lower are set to True, instead of computing
    # optimized bounds one by one, we jointly optimize on both sides.
    # This is useful for some custom optimization objectives.
    'joint_optimization': False
}


def opt_reuse(self: 'BoundedModule'):
    for node in self.get_enabled_opt_act():
        node.opt_reuse()


def opt_no_reuse(self: 'BoundedModule'):
    for node in self.get_enabled_opt_act():
        node.opt_no_reuse()


def _set_alpha(optimizable_activations, parameters, alphas, lr):
    """Set best_alphas, alphas and parameters list."""
    for node in optimizable_activations:
        alphas.extend(list(node.alpha.values()))
        node.opt_start()
    # Alpha has shape (2, output_shape, batch_dim, node_shape)
    parameters.append({'params': alphas, 'lr': lr, 'batch_dim': 2})
    # best_alpha is a dictionary of dictionary. Each key is the alpha variable
    # for one activation layer, and each value is a dictionary contains all
    # activation layers after that layer as keys.
    best_alphas = OrderedDict()
    for m in optimizable_activations:
        best_alphas[m.name] = {}
        for alpha_m in m.alpha:
            best_alphas[m.name][alpha_m] = m.alpha[alpha_m].detach().clone()
            # We will directly replace the dictionary for each activation layer after
            # optimization, so the saved alpha might not have require_grad=True.
            m.alpha[alpha_m].requires_grad_()

    return best_alphas


def _set_gammas(nodes, parameters):
    """
    Adds gammas to parameters list
    """
    gammas = []
    gamma_lr = 0.1
    for node in nodes:
        if hasattr(node, 'gammas'):
            gammas.append(node.gammas_underlying_tensor)
            # The learning rate is the same for all layers
            gamma_lr = node.options['optimize_bound_args']['oc_lr']
    parameters.append({'params': gammas, 'lr': gamma_lr})

def _save_ret_first_time(bounds, fill_value, x, best_ret):
    """Save results at the first iteration to best_ret."""
    if bounds is not None:
        best_bounds = torch.full_like(
            bounds, fill_value=fill_value, device=x[0].device, dtype=x[0].dtype)
        best_ret.append(bounds.detach().clone())
    else:
        best_bounds = None
        best_ret.append(None)

    return best_bounds


def _to_float64(self: 'BoundedModule', C, x, aux_reference_bounds, interm_bounds):
    """
    Transfer variables to float64 only in the last iteration to help alleviate
    floating point error.
    """
    self.to(torch.float64)
    C = C.to(torch.float64)
    x = self._to(x, torch.float64)
    # best_intermediate_bounds is linked to aux_reference_bounds!
    # we only need call .to() for one of them
    self._to(aux_reference_bounds, torch.float64, inplace=True)
    interm_bounds = self._to(
        interm_bounds, torch.float64)

    return C, x, interm_bounds


def _to_default_dtype(self: 'BoundedModule', x, total_loss, full_ret, ret,
                      best_intermediate_bounds, return_A):
    """
    Switch back to default precision from float64 typically to adapt to
    afterwards operations.
    """
    total_loss = total_loss.to(torch.get_default_dtype())
    self.to(torch.get_default_dtype())
    x[0].to(torch.get_default_dtype())
    full_ret = list(full_ret)
    if isinstance(ret[0], Tensor):
        # round down lower bound
        full_ret[0] = double2float(full_ret[0], 'down')
    if isinstance(ret[1], Tensor):
        # round up upper bound
        full_ret[1] = double2float(full_ret[1], 'up')
    for _k, _v in best_intermediate_bounds.items():
        _v[0] = double2float(_v[0], 'down')
        _v[1] = double2float(_v[1], 'up')
        best_intermediate_bounds[_k] = _v
    if return_A:
        full_ret[2] = self._to(full_ret[2], torch.get_default_dtype())

    return total_loss, x, full_ret


def _get_idx_mask(idx, full_ret_bound, best_ret_bound, loss_reduction_func):
    """Get index for improved elements."""
    assert idx in [0, 1], (
        '0 means updating lower bound, 1 means updating upper bound')
    if idx == 0:
        idx_mask = (loss_reduction_func(full_ret_bound)
                    > loss_reduction_func(best_ret_bound)).view(-1)
    else:
        idx_mask = (loss_reduction_func(full_ret_bound)
                    < loss_reduction_func(best_ret_bound)).view(-1)
    improved_idx = None
    if idx_mask.any():
        # we only pick up the results improved in a batch
        improved_idx = idx_mask.nonzero(as_tuple=True)[0]
    return idx_mask, improved_idx


def _update_best_ret(full_ret_bound, best_ret_bound, full_ret, best_ret,
                     need_update, loss_reduction_func, idx, deterministic=False):
    """Update best_ret_bound and best_ret by comparing with new results."""
    assert idx in [0, 1], (
        '0 means updating lower bound, 1 means updating upper bound')
    idx_mask, improved_idx = _get_idx_mask(
        idx, full_ret_bound, best_ret_bound, loss_reduction_func)

    if improved_idx is not None:
        need_update = True
        compare = torch.max if idx == 0 else torch.min
        if not deterministic:
            best_ret_bound[improved_idx] = compare(
                full_ret_bound[improved_idx], best_ret_bound[improved_idx])
        else:
            best_ret_bound[improved_idx] = full_ret_bound[improved_idx]
        if full_ret[idx] is not None:
            if not deterministic:
                best_ret[idx][improved_idx] = compare(
                    full_ret[idx][improved_idx],
                    best_ret[idx][improved_idx])
            else:
                best_ret[idx][improved_idx] = full_ret[idx][improved_idx]

    return best_ret_bound, best_ret, need_update, idx_mask, improved_idx


def _update_optimizable_activations(
        optimizable_activations, interm_bounds,
        fix_interm_bounds, best_intermediate_bounds,
        reference_idx, idx, alpha, best_alphas, deterministic):
    """
    Update bounds and alpha of optimizable_activations.
    """
    for node in optimizable_activations:
        # Update best intermediate layer bounds only when they are optimized.
        # If they are already fixed in interm_bounds, then do
        # nothing.
        if node.name not in best_intermediate_bounds:
            continue
        if (interm_bounds is None
                or node.inputs[0].name not in interm_bounds
                or not fix_interm_bounds):
            if deterministic:
                best_intermediate_bounds[node.name][0][idx] = node.inputs[0].lower[reference_idx]
                best_intermediate_bounds[node.name][1][idx] = node.inputs[0].upper[reference_idx]
            else:
                best_intermediate_bounds[node.name][0][idx] = torch.max(
                    best_intermediate_bounds[node.name][0][idx],
                    node.inputs[0].lower[reference_idx])
                best_intermediate_bounds[node.name][1][idx] = torch.min(
                    best_intermediate_bounds[node.name][1][idx],
                    node.inputs[0].upper[reference_idx])
        if alpha:
            # Each alpha has shape (2, output_shape, batch, *shape) for act.
            # For other activation function this can be different.
            for alpha_m in node.alpha:
                best_alphas[node.name][alpha_m][:, :,
                    idx] = node.alpha[alpha_m][:, :, idx]


def update_best_beta(self: 'BoundedModule', enable_opt_interm_bounds, betas,
                     best_betas, idx):
    """
    Update best beta by given idx.
    """
    if enable_opt_interm_bounds and betas:
        for node in self.splittable_activations:
            for node_input in node.inputs:
                for key in node_input.sparse_betas.keys():
                    best_betas[node_input.name][key] = (
                        node_input.sparse_betas[key].val.detach().clone())
        if self.cut_used:
            for gbidx, general_betas in enumerate(self.cut_beta_params):
                # FIXME need to check if 'cut' is a node name
                best_betas['cut'][gbidx] = general_betas.detach().clone()
    else:
        for node in self.nodes_with_beta:
            best_betas[node.name][idx] = node.sparse_betas[0].val[idx]
        if self.cut_used:
            regular_beta_length = len(betas) - len(self.cut_beta_params)
            for cut_beta_idx in range(len(self.cut_beta_params)):
                # general cut beta crown general_betas
                best_betas['cut'][cut_beta_idx][:, :, idx,
                    :] = betas[regular_beta_length + cut_beta_idx][:, :, idx, :]


def _get_optimized_bounds(
        self: 'BoundedModule', x=None, aux=None, C=None, IBP=False,
        forward=False, method='backward', bound_side='lower',
        reuse_ibp=False, return_A=False, average_A=False, final_node_name=None,
        interm_bounds=None, reference_bounds=None,
        aux_reference_bounds=None, needed_A_dict=None, cutter=None,
        decision_thresh=None, epsilon_over_decision_thresh=1e-4, use_clip_domains=False, swap_loss=False,
        plane_constraints_lower:Optional[Tensor]=None, plane_constraints_upper:Optional[Tensor]=None,
        custom_loss_func_params:Optional[dict]=None
):
    """
    Optimize CROWN lower/upper bounds by alpha and/or beta.
    """

    opts = self.bound_opts['optimize_bound_args']
    iteration = opts['iteration']
    max_time = opts['max_time']
    beta = opts['enable_beta_crown']
    alpha = opts['enable_alpha_crown']
    apply_output_constraints_to = opts['apply_output_constraints_to']
    opt_choice = opts['optimizer']
    keep_best = opts['keep_best']
    save_loss_graphs = opts['save_loss_graphs']
    perpendicular_multiplier = opts['perpendicular_multiplier']
    fix_interm_bounds = opts['fix_interm_bounds']
    loss_reduction_func = opts['loss_reduction_func']
    stop_criterion_func = opts['stop_criterion_func']
    use_float64_in_last_iteration = opts['use_float64_in_last_iteration']
    early_stop_patience = opts['early_stop_patience']
    start_save_best = opts['start_save_best']
    multi_spec_keep_func = opts['multi_spec_keep_func']
    deterministic = opts['deterministic']
    enable_opt_interm_bounds = self.bound_opts.get(
        'enable_opt_interm_bounds', False)
    sparse_intermediate_bounds = self.bound_opts.get(
        'sparse_intermediate_bounds', False)
    verbosity = self.bound_opts['verbosity']
    use_custom_loss = opts['use_custom_loss']
    custom_loss_func = opts.get('custom_loss_func', None)
    swap_loss_iter = opts.get('swap_loss_iter')
    if custom_loss_func_params is None:
        custom_loss_func_params = defaultdict(set)

    if bound_side not in ['lower', 'upper', 'both']:
        raise ValueError(bound_side)
    bound_lower = bound_side == 'lower'
    bound_upper = bound_side == 'upper'
    if bound_side == 'both':
        if keep_best:
            raise NotImplementedError(
                'Keeping the best results is only supported for the default optimization objective!')
        bound_lower = True
        bound_upper = True

    if bound_lower and plane_constraints_lower is not None:
        plane_constraints = plane_constraints_lower.to(x[0])
    elif bound_upper and plane_constraints_upper is not None:
        plane_constraints = plane_constraints_upper.to(x[0])
    else:
        plane_constraints = None

    assert alpha or beta, (
        'nothing to optimize, use compute bound instead!')

    if C is not None:
        self.final_shape = C.size()[:2]
        self.bound_opts.update({'final_shape': self.final_shape})
    if opts['init_alpha']:
        # TODO: this should set up aux_reference_bounds.
        self.init_alpha(x, share_alphas=opts['use_shared_alpha'],
                        method=method, c=C, final_node_name=final_node_name)

    optimizable_activations = self.get_enabled_opt_act()

    alphas, parameters = [], []
    dense_coeffs_mask = []
    if alpha:
        best_alphas = _set_alpha(
            optimizable_activations, parameters, alphas, opts['lr_alpha'])
    else:
        best_alphas = None
    if beta:
        ret_set_beta = self.set_beta(
            enable_opt_interm_bounds, parameters,
            opts['lr_beta'], opts['lr_cut_beta'], cutter, dense_coeffs_mask)
        betas, best_betas, coeffs, dense_coeffs_mask = ret_set_beta[:4]
    if apply_output_constraints_to is not None and len(apply_output_constraints_to) > 0:
        _set_gammas(self.nodes(), parameters)

    start = time.time()

    if isinstance(decision_thresh, Tensor):
        if decision_thresh.dim() == 1:
            # add the spec dim to be aligned with compute_bounds return
            decision_thresh = decision_thresh.unsqueeze(-1)

    if opts['pruning_in_iteration']:
        if return_A:
            raise NotImplementedError(
                'Pruning in iteration optimization does not support '
                'return A yet. '
                'Please fix or discard this optimization by setting '
                '--disable_pruning_in_iteration '
                'or bab: pruning_in_iteration: false')
        if bound_side != 'lower':
            raise NotImplementedError(
                'Pruning in iteration optimization only supports computing lower bounds.'
            )
        pruner = OptPruner(
            x, threshold=opts['pruning_in_iteration_threshold'],
            multi_spec_keep_func=multi_spec_keep_func,
            loss_reduction_func=loss_reduction_func,
            decision_thresh=decision_thresh,
            epsilon_over_decision_thresh=epsilon_over_decision_thresh,
            fix_interm_bounds=fix_interm_bounds)
    else:
        pruner = None

    if opt_choice == 'adam-autolr':
        opt = AdamElementLR(parameters)
    elif opt_choice == 'adam':
        opt = optim.Adam(parameters)
    elif opt_choice == 'sgd':
        opt = optim.SGD(parameters, momentum=0.9)
    else:
        raise NotImplementedError(opt_choice)

    # Create a weight vector to scale learning rate.
    loss_weight = torch.ones(size=(x[0].size(0),), device=x[0].device)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, opts['lr_decay'])

    # best_intermediate_bounds is linked to aux_reference_bounds!
    best_intermediate_bounds = {}
    if (sparse_intermediate_bounds and aux_reference_bounds is None
            and reference_bounds is not None):
        aux_reference_bounds = {}
        for name, (lb, ub) in reference_bounds.items():
            aux_reference_bounds[name] = [
                lb.detach().clone(), ub.detach().clone()]
    if aux_reference_bounds is None:
        aux_reference_bounds = {}

    if len(apply_output_constraints_to) > 0:
        # INVPROP requires that all layers have cached bounds. This may not be the case
        # unless we explicitly compute them.
        self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = []
        with torch.no_grad():
            self.compute_bounds(
                x=x, C=C, method='backward', bound_lower=bound_lower,
                bound_upper=bound_upper, final_node_name=final_node_name,
                interm_bounds=interm_bounds)
        self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = (
            apply_output_constraints_to
        )

    global lb_iter, ub_iter, displayed_lb, displayed_ub, displayed_graph, ax_graph, fig_graph_lower, fig_graph_upper, un_mask
    if swap_loss and (bound_lower or not bound_upper):
        if lb_iter is None:
            lb_iter = 0
        else:
            lb_iter += 1
    if swap_loss and (bound_upper or not bound_lower):
        if ub_iter is None:
            ub_iter = 0
        else:
            ub_iter += 1

    need_grad = True
    patience = 0
    ret_0 = None

    # When using a custom loss function, we may need the A-matrix depending on
    # the implementation of the loss function. So return A anyway.
    # (Perhaps add one more option to specify this choice?)
    if use_custom_loss or use_clip_domains:
        actual_return_A = return_A
        actual_needed_A_dict = needed_A_dict.copy()
        output_name = self.output_name[0]
        input_name = self.input_name[0]
        if not return_A:
            needed_A_dict = defaultdict(set)
            needed_A_dict[output_name].add(input_name)
            return_A = True
        else:
            # Even if return_A is already true, we must make sure that A_matrix of the whole network is returned
            extra_entries = defaultdict(set)
            extra_entries[output_name].add(input_name)
            needed_A_dict.update(extra_entries)

    for i in range(iteration):
        if cutter:
            # cuts may be optimized by cutter
            self.cut_module = cutter.cut_module

        intermediate_constr = None

        if not fix_interm_bounds:
            # If we still optimize all intermediate neurons, we can use
            # interm_bounds as reference bounds.
            if reference_bounds is None:
                reference_bounds = {}
            if interm_bounds is not None:
                reference_bounds.update(interm_bounds)
            interm_bounds = {}

        if i == iteration - 1:
            # No grad update needed for the last iteration
            need_grad = False
            if (self.device == 'cuda'
                    and torch.get_default_dtype() == torch.float32
                    and use_float64_in_last_iteration):
                C, x, interm_bounds = self._to_float64(
                    C, x, aux_reference_bounds, interm_bounds)

        if pruner:
            # we will use last update preserve mask in caller functions to recover
            # lA, l, u, etc to full batch size
            self.last_update_preserve_mask = pruner.preserve_mask
            pruner.cache_full_sized_alpha(optimizable_activations)

        # If input bounds are tightened with output constraints, they depend on the
        # relaxations of all other layers. The current iteration will recompute them.
        # This involves concretizing them, so they will depend on themselves.
        # To avoid a loop of gradients, remove gradients here.
        tighten_input_bounds = (
                self.bound_opts['optimize_bound_args']['tighten_input_bounds']
        )
        if tighten_input_bounds:
            for root in self.roots():
                if hasattr(root, 'perturbation') and root.perturbation is not None:
                    root.perturbation.x_L = root.perturbation.x_L.detach()
                    root.perturbation.x_U = root.perturbation.x_U.detach()

        with torch.no_grad() if not need_grad else ExitStack():
            # ret is lb, ub or lb, ub, A_dict (if return_A is set to true)
            ret = self.compute_bounds(
                x, aux, C, method=method, IBP=IBP, forward=forward,
                bound_lower=bound_lower, bound_upper=bound_upper,
                reuse_ibp=reuse_ibp, return_A=return_A,
                final_node_name=final_node_name, average_A=average_A,
                # When intermediate bounds are recomputed, we must set it
                # to None
                interm_bounds=interm_bounds if fix_interm_bounds else None,
                # This is the currently tightest interval, which will be used to
                # pass split constraints when intermediate betas are used.
                reference_bounds=reference_bounds,
                # This is the interval used for checking for unstable neurons.
                aux_reference_bounds=aux_reference_bounds if sparse_intermediate_bounds else None,
                # These are intermediate layer beta variables and their
                # corresponding A matrices and biases.
                intermediate_constr=intermediate_constr,
                needed_A_dict=needed_A_dict,
                update_mask=pruner.preserve_mask if pruner else None,
                cache_bounds=len(apply_output_constraints_to) > 0,
            )
        # If output constraints are used, it's possible that no inputs satisfy them.
        # If one of the layer that uses output constraints realizes this, it sets
        # self.infeasible_bounds = True for this element in the batch.
        if self.infeasible_bounds is not None and torch.any(self.infeasible_bounds):
            if ret[0] is not None:
                ret = (
                    torch.where(
                        self.infeasible_bounds.unsqueeze(1),
                        torch.full_like(ret[0], float('inf')),
                        ret[0],
                    ),
                    ret[1],
                )
            if ret[1] is not None:
                ret = (
                    ret[0],
                    torch.where(
                        self.infeasible_bounds.unsqueeze(1),
                        torch.full_like(ret[1], float('-inf')),
                        ret[1],
                    ),
                )
        ret_l, ret_u = ret[0], ret[1]

        if pruner:
            pruner.recover_full_sized_alpha(optimizable_activations)

        if (self.cut_used and i % cutter.log_interval == 0
                and len(self.cut_beta_params) > 0):
            # betas[-1]: (2(0 lower, 1 upper), spec, batch, num_constrs)
            if ret_l is not None:
                print(i, 'lb beta sum:',
                      f'{self.cut_beta_params[-1][0].sum() / ret_l.size(0)},',
                      f'worst {ret_l.min()}')
            if ret_u is not None:
                print(i, 'lb beta sum:',
                      f'{self.cut_beta_params[-1][1].sum() / ret_u.size(0)},',
                      f'worst {ret_u.min()}')

        if i == 0:
            # save results at the first iteration
            best_ret = []
            best_ret_l = _save_ret_first_time(
                ret[0], float('-inf'), x, best_ret)
            best_ret_u = _save_ret_first_time(
                ret[1], float('inf'), x, best_ret)
            ret_0 = ret[0].detach().clone() if bound_lower else ret[1].detach().clone()

            for node in optimizable_activations:
                if node.inputs[0].lower is None and node.inputs[0].upper is None:
                    continue
                new_intermediate = [node.inputs[0].lower.detach().clone(),
                                    node.inputs[0].upper.detach().clone()]
                best_intermediate_bounds[node.name] = new_intermediate
                if sparse_intermediate_bounds:
                    # Always using the best bounds so far as the reference
                    # bounds.
                    aux_reference_bounds[node.inputs[0].name] = new_intermediate

        def _reformat_batches(_x: tuple, _dm_lb: Tensor, _lA: Tensor, _lbias: Tensor
                              ) -> Tuple[Tensor, int, int, int, Tensor, Tensor]:
            """
            Returns reformatted lA with other necessary parameters
            :param _x:
            :param _dm_lb:
            :param _lA:
            :param _lbias:
            :return:
            """
            _batch_lA = _lA.flatten(2)
            _batches, _num_spec, _input_dim = _batch_lA.shape

            _x_L = _x[0].ptb.x_L
            _x_U = _x[0].ptb.x_U

            return _batch_lA, _batches, _num_spec, _input_dim, _x_L, _x_U

        if use_custom_loss and i >= swap_loss_iter:
            try:
                sig = inspect.signature(custom_loss_func)
                param_names = list(sig.parameters.keys())
                # We assume the keys of the functions are the names of local variables
                variables = {}
                for name in param_names:

                    if name in custom_loss_func_params:
                        # safer to check parameters here first
                        variables[name] = custom_loss_func_params[name]
                    elif name in locals():
                        variables[name] = locals()[name]
                    else:
                        raise ValueError(f"Cannot find {name} in local variables or custom_loss_func_params.")
                total_loss = custom_loss_func(**variables)
                # print(total_loss[0].item())
                stop_criterion = False
                full_ret = ret
            except Exception as e:
                warn("An error occurred. Please make sure that the implementation of the loss function is correct and")
                warn("pass it to BoundedModule using the key bound_opts['optimize_bound_args']['custom_loss_func']\n")
                raise Exception(e)

            # display the results:
            # parse to get lA and lbias
            ret_A = ret[2]
            if bound_lower:
                dm_lb = ret_l
                lA = ret_A[self.output_name[0]][self.input_name[0]]['lA']
                lbias = ret_A[self.output_name[0]][self.input_name[0]]['lbias']
            else:
                dm_lb = ret_u
                lA = ret_A[self.output_name[0]][self.input_name[0]]['uA']
                lbias = ret_A[self.output_name[0]][self.input_name[0]]['ubias']

            form_ret = _reformat_batches(x, dm_lb, lA, lbias)
            [batch_lA, batches, num_spec, input_dim, x_L, x_U] = form_ret
            volumes = total_loss.detach().clone()  # for viewing

            if save_loss_graphs and num_spec == 1 and input_dim == 3 and (
                    (not displayed_lb and bound_lower) or (not displayed_ub and bound_upper)):
                plot_args = {
                    'i': i - swap_loss_iter, 'bound_lower': bound_lower, 'bound_upper': bound_upper, 'x_L': x_L, 'x_U': x_U,
                    'dm_lb': dm_lb, 'batches': batches, 'num_spec': num_spec, 'input_dim': input_dim,
                    'volumes': volumes, 'batch_lA': batch_lA, 'lbias': lbias, 'forward_fn': self.forward,
                    'fig_graph_lower': fig_graph_lower, 'fig_graph_upper': fig_graph_upper, 'un_mask': un_mask,
                    'plane_constraints': plane_constraints, 'bev_dict': None,
                }
                fig_graph_lower, fig_graph_upper, un_mask = plot_domains(**plot_args)

        # if swap_loss and bound_lower:
        elif swap_loss and i >= swap_loss_iter:
            # parse to get lA and lbias
            ret_A = ret[2]
            if bound_lower:
                dm_lb = ret_l
                lA = ret_A[self.output_name[0]][self.input_name[0]]['lA']
                lbias = ret_A[self.output_name[0]][self.input_name[0]]['lbias']
            else:
                dm_lb = ret_u
                lA = ret_A[self.output_name[0]][self.input_name[0]]['uA']
                lbias = ret_A[self.output_name[0]][self.input_name[0]]['ubias']
            stop_criterion = False
            full_ret = ret

            form_ret = _reformat_batches(x, dm_lb, lA, lbias)
            [batch_lA, batches, num_spec, input_dim, x_L, x_U] = form_ret

            # Get the threshold in the proper format to use for edge case when calculating volume
            if isinstance(decision_thresh, (float, int)):
                threshold = torch.full_like(dm_lb, decision_thresh)
            elif isinstance(decision_thresh, Tensor):
                threshold = decision_thresh.clone()
            else:
                raise Exception("Make sure that 'decision_thresh' is specified.")

            # def _concretize(xhat, eps, lA, lbias, sgn):
            #     bound = lA.bmm(xhat) + sgn * lA.abs().bmm(eps) + lbias
            #     return bound


            # xhat = (x_U + x_L) / 2
            # eps = (x_U - x_L) / 2
            # sgn = -1 if bound_lower else +1
            # xhat_vect = xhat.unsqueeze(2)  # to column vectors
            # eps_vect = eps.unsqueeze(2)  # to column vectors

            # reshape tensors to be compatible with the volume calculation in batches
            # lA_reshape = -1 * batch_lA.reshape(batches * num_spec, input_dim) if bound_lower else batch_lA.reshape(batches * num_spec, input_dim)
            lA_reshape = batch_lA.reshape(batches * num_spec, input_dim)
            x_L_reshape = x_L.reshape(batches, input_dim)
            x_U_reshape = x_U.reshape(batches, input_dim)
            x_L_reshape = x_L_reshape.expand(batches * num_spec, input_dim)
            x_U_reshape = x_U_reshape.expand(batches * num_spec, input_dim)
            # lbias_reshape = -1 * lbias.reshape(batches * num_spec, 1) if bound_lower else lbias.reshape(batches * num_spec, 1)
            lbias_reshape = lbias.reshape(batches * num_spec, 1)

            # try out heuristic volume loss
            total_loss, bev_dict = batch_estimate_volume(
                x_L_reshape, x_U_reshape, lA_reshape, lbias_reshape, not bound_lower, plane_constraints,
                perpendicular_multiplier=perpendicular_multiplier, return_dict=True
            )

            # try out exact volume loss
            # if plane_constraints is not None:
            #     plane_constraints = plane_constraints.to(x_L_reshape)
            #     total_loss = batch_cube_intersection_with_plane_with_constraints(x_L_reshape, x_U_reshape, lA_reshape,
            #                                                                      lbias_reshape, plane_constraints,
            #                                                                      bound_lower, False)
            # else:
            #     total_loss = batch_cube_intersection_with_plane(x_L_reshape, x_U_reshape, lA_reshape, lbias_reshape,
            #                                                     dm_lb.reshape(batches * num_spec, -1), threshold,
            #                                                     lb_iter,
            #                                                     bound_lower)

            volumes = total_loss.detach().clone()  # for viewing
            # total_loss *= -1  # to maximize (maximize for exact volume, minimize for heuristic)

            # for specific scenarios (1 specification 3D input space) it is possible to graph the inputs and planes
            if save_loss_graphs and num_spec == 1 and input_dim == 3 and (
                    (not displayed_lb and bound_lower) or (not displayed_ub and bound_upper)):
                plot_args = {
                    'i': i - swap_loss_iter, 'bound_lower': bound_lower, 'bound_upper': bound_upper, 'x_L': x_L, 'x_U': x_U,
                    'dm_lb': dm_lb, 'batches': batches, 'num_spec': num_spec, 'input_dim': input_dim,
                    'volumes': volumes, 'batch_lA': batch_lA, 'lbias': lbias, 'forward_fn': self.forward,
                    'fig_graph_lower': fig_graph_lower, 'fig_graph_upper': fig_graph_upper, 'un_mask': un_mask,
                    'plane_constraints': plane_constraints, 'bev_dict': bev_dict,
                }
                fig_graph_lower, fig_graph_upper, un_mask = plot_domains(**plot_args)
        else:
            l = ret_l
            # Reduction over the spec dimension.
            if ret_l is not None and ret_l.shape[1] != 1:
                l = loss_reduction_func(ret_l)
            u = ret_u
            if ret_u is not None and ret_u.shape[1] != 1:
                u = loss_reduction_func(ret_u)

            # full_l, full_ret_l and full_u, full_ret_u is used for update the best
            full_ret_l, full_ret_u = ret_l, ret_u
            full_l = l
            full_ret = ret

            if pruner:
                (x, C, full_l, full_ret_l, full_ret_u,
                full_ret, stop_criterion) = pruner.prune(
                    x, C, ret_l, ret_u, ret, full_l, full_ret_l, full_ret_u,
                    full_ret, interm_bounds, aux_reference_bounds, reference_bounds,
                    stop_criterion_func, bound_lower)
            else:
                stop_criterion = (stop_criterion_func(full_ret_l) if bound_lower
                                else stop_criterion_func(-full_ret_u))

            total_loss = -l if bound_lower else u
            directly_optimize_layers = self.bound_opts['optimize_bound_args']['directly_optimize']
            for directly_optimize_layer_name in directly_optimize_layers:
                total_loss += (
                    self[directly_optimize_layer_name].upper.sum()
                    - self[directly_optimize_layer_name].lower.sum()
                )

        if type(stop_criterion) == bool:
            loss = total_loss.sum() * (not stop_criterion)
        else:
            assert total_loss.shape == stop_criterion.shape
            loss = (total_loss * stop_criterion.logical_not()).sum()

        stop_criterion_final = isinstance(
            stop_criterion, Tensor) and stop_criterion.all()

        if i == iteration - 1:
            best_ret = list(best_ret)
            if best_ret[0] is not None:
                best_ret[0] = best_ret[0].to(torch.get_default_dtype())
            if best_ret[1] is not None:
                best_ret[1] = best_ret[1].to(torch.get_default_dtype())

        if (i == iteration - 1 and self.device == 'cuda'
                and torch.get_default_dtype() == torch.float32
                and use_float64_in_last_iteration):
            total_loss, x, full_ret = self._to_default_dtype(
                x, total_loss, full_ret, ret, best_intermediate_bounds, return_A)

        with torch.no_grad():
            # for lb and ub, we update them in every iteration since updating
            # them is cheap
            need_update = False
            improved_idx = None
            if keep_best:
                if best_ret_u is not None:
                    best_ret_u, best_ret, need_update, idx_mask, improved_idx = _update_best_ret(
                        full_ret_u, best_ret_u, full_ret, best_ret, need_update,
                        loss_reduction_func, idx=1, deterministic=deterministic)
                if best_ret_l is not None:
                    best_ret_l, best_ret, need_update, idx_mask, improved_idx = _update_best_ret(
                        full_ret_l, best_ret_l, full_ret, best_ret, need_update,
                        loss_reduction_func, idx=0, deterministic=deterministic)
            else:
                # Not saving the best, just keep the last iteration.
                if full_ret[0] is not None:
                    best_ret[0] = full_ret[0]
                if full_ret[1] is not None:
                    best_ret[1] = full_ret[1]
            if return_A:
                if not use_custom_loss or actual_return_A:
                    # FIXME: A should also be updated by idx.
                    # FIXME: When using custom loss and actual return_A is True,
                    # we might return more entries than those in actual_needed_A_dict
                    best_ret = [best_ret[0], best_ret[1], full_ret[2]]

            if need_update or not keep_best:
                patience = 0  # bounds improved, reset patience
            else:
                patience += 1

            time_spent = time.time() - start

            if keep_best:
                # Save variables if this is the best iteration.
                # To save computational cost, we only check keep_best at the first
                # (in case divergence) and second half iterations
                # or before early stop by either stop_criterion or
                # early_stop_patience reached
                if (i < 1 or i > int(iteration * start_save_best) or deterministic
                        or stop_criterion_final or patience == early_stop_patience
                        or time_spent > max_time):

                    # compare with the first iteration results and get improved indexes
                    if bound_lower:
                        if deterministic:
                            idx = improved_idx
                            idx_mask = None
                        else:
                            idx_mask, idx = _get_idx_mask(
                                0, full_ret_l, ret_0, loss_reduction_func)
                        ret_0[idx] = full_ret_l[idx]
                    else:
                        if deterministic:
                            idx = improved_idx
                            idx_mask = None
                        else:
                            idx_mask, idx = _get_idx_mask(
                                1, full_ret_u, ret_0, loss_reduction_func)
                        ret_0[idx] = full_ret_u[idx]

                    if idx is not None:
                        # for update propose, we condition the idx to update only
                        # on domains preserved
                        if pruner:
                            reference_idx, idx = pruner.prune_idx(idx_mask, idx, x)
                        else:
                            reference_idx = idx

                        _update_optimizable_activations(
                            optimizable_activations, interm_bounds,
                            fix_interm_bounds, best_intermediate_bounds,
                            reference_idx, idx, alpha, best_alphas, deterministic)

                        if beta:
                            self.update_best_beta(enable_opt_interm_bounds, betas,
                                                best_betas, idx)

        # udpate x_L, x_U
        # with torch.no_grad():
        #
        #     # Now that we have the bounds, we can perform clipping and udpate the input bounds
        #     if use_clip_domains and bound_lower:
        #         if ret[2] is None:
        #             raise Exception("Using clip domains requires A_dict but A_dict was not returned")
        #
        #         # FIXME: Now need to consider lb and ub, this is only correct for lb
        #         clip_ret = clip_domains(
        #             x_L,
        #             x_U,
        #             decision_thresh,
        #             lA,
        #             ret[0],
        #             lbias,
        #             False
        #         )
        #         [new_x_L, new_x_U] = clip_ret
        #
        #         # Form the new input bounded tensor
        #         data = x[0].data.clone()
        #         ptb = PerturbationLpNorm(
        #             norm=x[0].ptb.norm, eps=x[0].ptb.eps, x_L=new_x_L, x_U=new_x_U)
        #         new_x = BoundedTensor(data, ptb).to(data.device)  # the value of data doesn't matter, only ptb
        #
        #         x = (new_x,)  # input x must be a tuple


        if os.environ.get('AUTOLIRPA_DEBUG_OPT', False):
            print(f'****** iter [{i}]',
                  f'loss: {loss.item()}, lr: {opt.param_groups[0]["lr"]}',
                  (' pruning_in_iteration open status: '
                     f'{pruner.pruning_in_iteration}') if pruner else '')

        if stop_criterion_final:
            print(f'\nall verified at {i}th iter')
            break

        if patience > early_stop_patience:
            print(f'Early stop at {i}th iter due to {early_stop_patience}'
                  ' iterations no improvement!')
            break

        if time_spent > max_time:
            print(f'Early stop at {i}th iter due to exceeding the time limit '
                  f'for the optimization (time spent: {time_spent})')
            break

        if i != iteration - 1 and not loss.requires_grad:
            assert i == 0, (i, iteration)
            print('[WARNING] No optimizable parameters found. Will skip optimiziation. '
                  'This happens e.g. if all optimizable layers are freezed or the '
                  'network has no optimizable layers.')
            break

        opt.zero_grad(set_to_none=True)

        if verbosity > 2:
            current_lr = [param_group['lr'] for param_group in opt.param_groups]
            print(f'*** iter [{i}]\n', f'loss: {loss.item()}',
                  total_loss.squeeze().detach().cpu().numpy(), 'lr: ',
                  current_lr)
            if beta:
                print_optimized_beta(optimizable_activations)
            if beta and i == 0 and verbosity > 2:
                breakpoint()

        if i != iteration - 1:
            # we do not need to update parameters in the last step since the
            # best result already obtained
            loss.backward()

            # All intermediate variables are not needed at this point.
            self._clear_and_set_new(
                None,
                cache_bounds=len(apply_output_constraints_to) > 0,
            )
            if opt_choice == 'adam-autolr':
                opt.step(lr_scale=[loss_weight, loss_weight])
            else:
                opt.step()

        if beta:
            for b in betas:
                b.data = (b >= 0) * b.data
            for dmi in range(len(dense_coeffs_mask)):
                # apply dense mask to the dense split coeffs matrix
                coeffs[dmi].data = (
                    dense_coeffs_mask[dmi].float() * coeffs[dmi].data)

        ### preprocessor-hint: private-section-start
        # Possibly update cuts if they are parameterized and optimzied
        if self.cut_used:
            cutter.update_cuts()
        ### preprocessor-hint: private-section-end

        if alpha:
            for m in optimizable_activations:
                m.clip_alpha()
        if apply_output_constraints_to is not None and len(apply_output_constraints_to) > 0:
            for m in self.nodes():
                m.clip_gammas()

        scheduler.step()

        if pruner:
            pruner.next_iter()

    if pruner:
        best_ret = pruner.update_best(full_ret_l, full_ret_u, best_ret)

    if verbosity > 3:
        breakpoint()

    if keep_best:
        # Set all variables to their saved best values.
        with torch.no_grad():
            for idx, node in enumerate(optimizable_activations):
                if node.name not in best_intermediate_bounds:
                    continue
                if alpha:
                    # Assigns a new dictionary.
                    node.alpha = best_alphas[node.name]
                # Update best intermediate layer bounds only when they are
                # optimized. If they are already fixed in
                # interm_bounds, then do nothing.
                best_intermediate = best_intermediate_bounds[node.name]
                node.inputs[0].lower.data = best_intermediate[0].data
                node.inputs[0].upper.data = best_intermediate[1].data
            if beta:
                for node in self.nodes_with_beta:
                    assert getattr(node, 'sparse_betas', None) is not None
                    if enable_opt_interm_bounds:
                        for key in node.sparse_betas.keys():
                            node.sparse_betas[key].val.copy_(
                                best_betas[node.name][key])
                    else:
                        node.sparse_betas[0].val.copy_(best_betas[node.name])
            if self.cut_used:
                for ii in range(len(self.cut_beta_params)):
                    self.cut_beta_params[ii].data = best_betas['cut'][ii].data

    if interm_bounds is not None and not fix_interm_bounds:
        for l in self._modules.values():
            if (l.name in interm_bounds.keys()
                    and l.is_lower_bound_current()):
                l.lower = torch.max(l.lower, interm_bounds[l.name][0])
                l.upper = torch.min(l.upper, interm_bounds[l.name][1])
                infeasible_neurons = l.lower > l.upper
                if infeasible_neurons.any():
                    print(f'Infeasibility detected in layer {l.name}.',
                          infeasible_neurons.sum().item(),
                          infeasible_neurons.nonzero()[:, 0])

    if verbosity > 0:
        if best_ret_l is not None:
            # FIXME: unify the handling of l and u.
            print('best_l after optimization:', best_ret_l.sum().item())
            if beta:
                print('beta sum per layer:', [p.sum().item() for p in betas])
        print('alpha/beta optimization time:', time.time() - start)

    for node in optimizable_activations:
        node.opt_end()

    if pruner:
        pruner.update_ratio(full_l, full_ret_l)
        pruner.clean_full_sized_alpha_cache()

    if os.environ.get('AUTOLIRPA_DEBUG_OPT', False):
        print()

    if save_loss_graphs and not displayed_ub and bound_upper and swap_loss:
        displayed_ub = True
        figs = [fig for (fig, _) in fig_graph_upper]
        title = f"/home/jorgejc2/Documents/Research/ray-casting/temp_figs/loss_upper_fig"
        titles = [title + '_' + str(i) for i in range(len(fig_graph_upper))]
        finalize_plots(figs, title, titles, pickle_data=True)

    elif save_loss_graphs and not displayed_lb and bound_lower and swap_loss:
        displayed_lb = True
        figs = [fig for (fig, _) in fig_graph_lower]
        title = f"/home/jorgejc2/Documents/Research/ray-casting/temp_figs/loss_lower_fig"
        titles = [title + '_' + str(i) for i in range(len(fig_graph_lower))]
        finalize_plots(figs, title, titles, pickle_data=True)

    return best_ret


def init_alpha(self: 'BoundedModule', x, share_alphas=False, method='backward',
               c=None, bound_lower=True, bound_upper=True, final_node_name=None,
               interm_bounds=None, activation_opt_params=None,
               skip_bound_compute=False):
    self(*x) # Do a forward pass to set perturbed nodes
    final = (self.final_node() if final_node_name is None
             else self[final_node_name])
    self._set_used_nodes(final)

    optimizable_activations = self.get_enabled_opt_act()
    for node in optimizable_activations:
        # TODO(7/6/2023) In the future, we may need to enable alpha sharing
        # automatically by consider the size of all the optimizable nodes in the
        # graph. For now, only an adhoc check in MatMul is added.
        node._all_optimizable_activations = optimizable_activations

        # initialize the parameters
        node.opt_init()

    apply_output_constraints_to = (
        self.bound_opts['optimize_bound_args']['apply_output_constraints_to']
    )
    if (not skip_bound_compute or interm_bounds is None or
            activation_opt_params is None or not all(
                [act.name in activation_opt_params
                 for act in self.optimizable_activations])):
        skipped = False
        # if new interval is None, then CROWN interval is not present
        # in this case, we still need to redo a CROWN pass to initialize
        # lower/upper
        with torch.no_grad():
            # We temporarilly deactivate output constraints
            self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = []
            l, u = self.compute_bounds(
                x=x, C=c, method=method, bound_lower=bound_lower,
                bound_upper=bound_upper, final_node_name=final_node_name,
                interm_bounds=interm_bounds)
            self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = (
                apply_output_constraints_to
            )
            if len(apply_output_constraints_to) > 0:
                # Some layers, such as the BoundTanh layer, do some of their initialization
                # in the forward pass. We need to call the forward pass again to ensure
                # that they are initialized for the output constraints, too.
                l, u = self.compute_bounds(
                    x=x, C=c, method=method, bound_lower=bound_lower,
                    bound_upper=bound_upper, final_node_name=final_node_name,
                    interm_bounds=interm_bounds, cache_bounds=True)
    else:
        # we skip, but we still would like to figure out the "used",
        # "perturbed", "backward_from" of each note in the graph
        skipped = True
        # this set the "perturbed" property
        self.set_input(*x, interm_bounds=interm_bounds)
        self.backward_from = {node: [final] for node in self._modules}
        l = u = None

    final_node_name = final_node_name or self.final_name

    init_intermediate_bounds = {}
    for node in optimizable_activations:
        start_nodes = []
        if method in ['forward', 'forward+backward']:
            start_nodes.append(('_forward', 1, None, False))
        if method in ['backward', 'forward+backward']:
            start_nodes += self.get_alpha_crown_start_nodes(
                node,
                c=c,
                share_alphas=share_alphas,
                final_node_name=final_node_name,
            )
        if not start_nodes:
            continue
        if skipped:
            node.restore_optimized_params(activation_opt_params[node.name])
        else:
            node.init_opt_parameters(start_nodes)
        if node in self.splittable_activations:
            for i in node.requires_input_bounds:
                input_node = node.inputs[i]
                if (not input_node.perturbed
                        or node.inputs[i].lower is None
                        and node.inputs[i].upper is None):
                    continue
                init_intermediate_bounds[node.inputs[i].name] = (
                    [node.inputs[i].lower.detach(),
                    node.inputs[i].upper.detach()])
    if (
        apply_output_constraints_to is not None
        and len(apply_output_constraints_to) > 0
        and hasattr(self, 'constraints')
    ):
        # self.constraints.shape = (batch_size, num_constraints, num_output_neurons)
        # For abCROWN we know that:
        # If the output constraints are a conjunction, the shape is (1, num_constraints, *)
        # If the output constraints are a disjunction, the shape is (num_constraints, 1, *)
        # Checking which entry is 1 allows to discern both cases.
        # If auto_LiRPA is used directly, we could have batches of inputs with more than one
        # constraint. This is currently not supported.
        if self.constraints.size(0) == 1:
            num_gammas = self.constraints.size(1)
        elif self.constraints.size(1) == 1:
            num_gammas = self.constraints.size(0)
        else:
            raise NotImplementedError(
                'To use output constraints, either have a batch size of 1 or use only one '
                'output constraint'
            )
        for node in self.nodes():
            node.init_gammas(num_gammas)

    if self.bound_opts['verbosity'] >= 1:
        print('Optimizable variables initialized.')
    if skip_bound_compute:
        return init_intermediate_bounds
    else:
        return l, u, init_intermediate_bounds

def plot_domains(i: int, bound_lower: bool, bound_upper: bool, x_L: Tensor, x_U: Tensor, dm_lb: Tensor, batches: int,
                 num_spec: int, input_dim: int, volumes: Tensor, batch_lA: Tensor, lbias: Tensor, forward_fn,
                 fig_graph_lower: list, fig_graph_upper: list, un_mask: Tensor,
                 plane_constraints: Optional[Tensor], bev_dict: Optional[dict] = None
                 ) -> Tuple[list, list, Tensor]:
    """
    Helper function to fill in plots in verification problems that can be visually shown.
    :param i:                   Epoch iteration
    :param bound_lower:         If the current bound is a lower bound
    :param bound_upper:         If the current bound is an upper bound
    :param x_L:                 Domain input lower bound
    :param x_U:                 Domain input upper bound
    :param dm_lb:               Network specification lower bound (or upper bound)
    :param batches:             Number of batches
    :param num_spec:            Specification dimension
    :param input_dim:           Input dimension
    :param volumes:             Volume results from loss function
    :param batch_lA:            lA matrix with shape (batches, num_spec, input_dim)
    :param lbias:               lbias with shape (batches, num_spec)
    :param forward_fn:          Forward pass function of the original network
    :param fig_graph_lower:     List of figures holding the plots for the lower bound
    :param fig_graph_upper:     List of figures holding the plots for the lower bound
    :param un_mask:             Mask to pull out the unverified instances
    :param plane_constraints:   If given, the plane constraints used in the optimization
    :param bev_dict:            If given, displays more detailed info from the loss function
    :return:
    """
    assert num_spec == 1 and input_dim == 3, "Dimensions don't fit, cannot plot these nodes"
    # for specific scenarios (1 specification 3D input space) it is possible to graph the inputs and planes
    with torch.no_grad(): # to be safe that gradients do not get affected
        ## graph set up
        print(f"i {i} bev_dict: {bev_dict if bev_dict is not None else 'N/A'}")

        # get a set of fixed masks to analyze a set of unverified nodes
        if un_mask is None:
            if bound_lower:
                un_mask = (dm_lb < 0).squeeze()
            elif bound_upper:
                un_mask = (dm_lb > 0).squeeze()
            un_mask_offset = 12 * 3  # start idx of nodes to plot
            un_mask[:un_mask_offset] = False
        num_unverified = un_mask.to(dtype=torch.int).sum()
        print(
            f"{num_unverified} unverified nodes out of {batches} nodes | lb_iter {lb_iter}, ub_iter {ub_iter}")

        [rows, cols] = get_rows_cols(num_unverified)

        # create/retrieve list of figures that will be manipulated
        if fig_graph_lower is None:
            fig_graph_lower = []

        if fig_graph_upper is None:
            fig_graph_upper = []

        if bound_lower:
            if len(fig_graph_lower) == i:
                fig, axs = plt.subplots(rows, cols, figsize=(12, 8), subplot_kw={'projection': '3d'})
                axs = axs.flatten()
                fig_graph_lower.append((fig, axs))
            else:
                fig, axs = fig_graph_lower[i]
        else:
            if len(fig_graph_upper) == i:
                fig, axs = plt.subplots(rows, cols, figsize=(12, 8), subplot_kw={'projection': '3d'})
                axs = axs.flatten()
                fig_graph_upper.append((fig, axs))
            else:
                fig, axs = fig_graph_upper[i]

        ## start of actual plotting

        # keep entries that were unverified on the first iteration of alpha crown
        nv_x_L = x_L[un_mask]  # nv for not verified
        nv_x_U = x_U[un_mask]
        # nv_batch_lA = -1 * batch_lA[un_mask] if bound_lower else batch_lA[un_mask]
        # nv_lbias = -1 * lbias[un_mask] if bound_lower else lbias[un_mask]
        nv_batch_lA = batch_lA[un_mask]
        nv_lbias = lbias[un_mask]
        nv_volumes = volumes[un_mask]
        nv_dm_lb = dm_lb[un_mask]
        plot_batches = len(axs)   # number of plots
        nv_plane_constraints = None
        if plane_constraints is not None:
            # nv_plane_constraints = -1 * plane_constraints[un_mask] if bound_lower else plane_constraints[un_mask]
            nv_plane_constraints = plane_constraints[un_mask]

        fill_plots(
            axs, forward_fn,
            nv_x_L, nv_x_U, nv_batch_lA, nv_lbias, nv_volumes, plot_batches, num_unverified, num_spec,
            nv_dm_lb, nv_plane_constraints, True, i)

        return fig_graph_lower, fig_graph_upper, un_mask