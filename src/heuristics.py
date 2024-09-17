#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
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
"""
Old branching heuristics, must be removed very soon (assigned to Kaidi).
"""

import torch
from typing import Tuple

@torch.no_grad()
def input_split_branching(dom_lb, x_L, x_U, lA, thresholds,
                          branching_method, split_depth=1, arguments=None):
    """
    Produce input split according to branching methods.
    """

    if arguments is None:
        arguments = {
            'sb_coeff_thresh': 0.001,  # Clamp values of coefficient matrix (A matrix) for sb branching heuristic.
            'sb_margin_weight': 1.0,  # Weight for the margin term in the sb heuristic.
            'sb_sum': False,  # Use sum for multiple specs in sb.
            'sb_primary_spec': None,
            'touch_zero_score': 0  # A touch-zero score in BF.
        }

    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)

    if branching_method == 'naive':
        # we just select the longest edge
        return torch.topk(x_U - x_L, split_depth, -1).indices
    elif branching_method == 'sb':
        return input_split_heuristic_sb(
            x_L, x_U, dom_lb, thresholds, lA, split_depth, arguments)
    else:
        raise NameError(f'Unsupported branching method "{branching_method}" for input splits.')

def input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth, arguments) -> Tuple[torch.Tensor]:
    """

    Smart branching where the sensitivities for each input is calculated as a score. More sensitive inputs are split.

    :param x_L:         The lower bound on the inputs of the subdomains
    :param x_U:         The upper bound on the inputs of the subdomains
    :param dom_lb:      The lower bound on the outputs of the subdomains
    :param thresholds:  The specification threshold where dom_lb > thresholds implies the subdomain is verified
    :param lA:          CROWN lA for subdomains
    :param split_depth: How many splits we wish to consider for all subdomains where split_depth <= input_dim
    :param arguments:   Input indices to split on for each batch
    :return:
    """
    lA_clamping_thresh = arguments.get('sb_coeff_thresh')
    sb_margin_weight = arguments.get('sb_margin_weight')
    sb_sum = arguments('sb_sum')
    sb_primary_spec = arguments('sb_primary_spec')
    touch_zero_score = arguments('touch_zero_score')

    lA = lA.flatten(2)
    # lA shape: (batch, spec, # inputs)
    perturb = (x_U - x_L).unsqueeze(-2)
    # perturb shape: (batch, 1, # inputs)
    # dom_lb shape: (batch, spec)
    # thresholds shape: (batch, spec)
    assert lA_clamping_thresh >= 0

    if sb_sum:
        score = lA.abs().clamp(min=lA_clamping_thresh) * perturb / 2
        score = score.sum(dim=-2)
        if touch_zero_score:
            touch_zero = torch.logical_or(x_L == 0, x_U == 0)
            score = score + touch_zero * (x_U - x_L) * touch_zero_score
    else:
        score = (lA.abs().clamp(min=lA_clamping_thresh) * perturb / 2
                + (dom_lb.to(lA.device).unsqueeze(-1)
                    - thresholds.unsqueeze(-1)) * sb_margin_weight)
        if sb_primary_spec is not None:
            assert score.ndim == 3
            score = score[:, sb_primary_spec, :]
        else:
            score = score.amax(dim=-2)
    # note: the k (split_depth) in topk <= # inputs, because split_depth is computed as
    # min(max split depth, # inputs).
    # 1) If max split depth <= # inputs, then split_depth <= # inputs.
    # 2) If max split depth > # inputs, then split_depth = # inputs.
    return torch.topk(score, split_depth, -1).indices