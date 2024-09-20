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

import torch
from torch import Tensor
from typing import Union, Tuple

uniform_grid = False
all_dims = None


def clip_domains(
        x_L: Tensor,
        x_U: Tensor,
        thresholds: Union[Tensor, float],
        lA: Tensor,
        dm_lb: Union[Tensor, None] = None,
        lbias: Union[Tensor, None] = None,
        calculate_dm_lb: bool = False,
        clip_dimensions: Union[Tensor, int, None] = None,
        split_depth_meta = None
) -> Tuple[Tensor, Tensor]:
    """
    Takes subdomains (or original domain) and shrinks along dimensions to remove verified portions of the input domain
    to remove redundancy and allow for more effective splits.
    :param x_L:                 The lower bound on the inputs of the subdomains
    :param x_U:                 The upper bound on the inputs of the subdomains
    :param thresholds:          The specification threshold where dom_lb > thresholds implies the subdomain is verified
    :param lA:                  CROWN lA for subdomains
    :param dm_lb:               The lower bound on the outputs of the domains
    :param lbias:               CROWN lbias for subdomains. Needed to concretize dm_lb if dm_lb is not given/incorrect
    :param calculate_dm_lb:     If set to true, dm_lb is assumed to be None or incorrect. lbias is then needed
    :return:                    The new x_L, x_U
    """
    if calculate_dm_lb:
        assert isinstance(lbias, Tensor), "lbias is needed to concretize dm_lb"
    else:
        assert isinstance(dm_lb, Tensor), "dm_lb was not given"

    global uniform_grid, all_dims
    if uniform_grid is False and split_depth_meta is not None:
        split_depth = split_depth_meta.get("split_depth")
        og_x_L = split_depth_meta.get("og_x_L")
        og_x_U = split_depth_meta.get("og_x_U")
        points = 2 ** split_depth

        dim1 = torch.linspace(og_x_L[0, 0], og_x_U[0, 0], points).unsqueeze(0)
        dim2 = torch.linspace(og_x_L[0, 1], og_x_U[0, 1], points).unsqueeze(0)
        dim3 = torch.linspace(og_x_L[0, 2], og_x_U[0, 2], points).unsqueeze(0)
        all_dims = [dim1, dim2, dim3]
        uniform_grid = True

    # save original shapes
    x_L_shape = x_L.shape
    x_U_shape = x_U.shape

    # Get initial variables and correct views
    lA = lA.flatten(2)
    batches, num_spec, input_dim = lA.shape
    x_L = x_L.clone().view(batches, input_dim)
    x_U = x_U.clone().view(batches, input_dim)
    if isinstance(thresholds, float):
        # transform thresholds into a Tensor if given as a single float
        thresholds = torch.full((batches, num_spec), thresholds, dtype=x_L.dtype, device=x_L.device)
    # x_L/x_U shape: (batch, input_dim)
    # lA shape: (batch, num_spec, input_dim)
    # dm_lb shape: (batch, num_spec)
    # lbias shape: (batch, num_spec)
    # thresholds shape: (batch, num_spec)

    # shapes (batch, input_dim)
    xhat = (x_U + x_L) / 2
    eps = (x_U - x_L) / 2

    if calculate_dm_lb:
        # use lbias to concretize dm_lb for the subdomains
        # transform to vectors that have shape (batch, _, 1)
        lbias = lbias.flatten(1)
        xhat_vect = xhat.unsqueeze(2)
        eps_vect = eps.unsqueeze(2)
        lbias_vect = lbias.unsqueeze(2)
        # shape (batch, num_spec, 1)
        dm_lb = lA.bmm(xhat_vect) - (lA.abs()).bmm(eps_vect) + lbias_vect
        # squeeze out singleton dimension
        dm_lb = dm_lb.squeeze(2)

    # ensures we only evaluate the domains that are not already verified from splitting
    # as shrinking these domains is simply redundant
    not_verified = (dm_lb <= thresholds).all(1)
    original_areas = (x_U[not_verified] - x_L[not_verified]).prod(1)
    original_total_area = original_areas.sum(0).item()

    # Solve for x in parallel
    # Solving for x in one dimension while concretizing the rest gives solutions of shape (batch, num_spec)
    # Repeating this process over all dimensions gives final shape of
    # concrete_minus_one and curr_x: (batch, num_spec, input_dim)
    concrete_minus_one = dm_lb.unsqueeze(2) - lA * xhat.unsqueeze(1) + lA.abs() * eps.unsqueeze(1)
    curr_x = (thresholds.unsqueeze(2) - concrete_minus_one) / lA

    # Sort solutions appropriately
    x_U_candidates = torch.where(lA > 0, curr_x, torch.inf)
    x_L_candidates = torch.where(lA < 0, curr_x, -torch.inf)

    # Update new_x_U(L)
    # min returns tuple (min, min_indices), so we use [0] to only get min values
    # we care about min across output dimension assuming at least one specification output must be verified
    x_L_temp, x_U_temp = x_L.clone(), x_U.clone()
    if clip_dimensions is None:
        x_U_temp = torch.min(x_U_candidates.min(dim=1)[0], x_U)
        x_L_temp = torch.max(x_L_candidates.max(dim=1)[0], x_L)
    else:
        x_U_temp[:, clip_dimensions] = torch.min(x_U_candidates.min(dim=1)[0], x_U)[:, clip_dimensions]
        x_L_temp[:, clip_dimensions] = torch.max(x_L_candidates.max(dim=1)[0], x_L)[:, clip_dimensions]

    if uniform_grid is False:
        x_L = x_L_temp
        x_U = x_U_temp
    else:
        curr_all_dims = [dim.repeat(batches, 1) for dim in all_dims]
        for i in range(3):
            # batches x 1
            # all_dims[i] is 1 x 2 ** split_depth
            # curr_all_dims[i] is batches x 2 ** split_depth
            upper_choices = torch.where(curr_all_dims[i] >= x_U_temp[:, i].unsqueeze(1), curr_all_dims[i], torch.inf)
            lower_choices = torch.where(curr_all_dims[i] <= x_L_temp[:, i].unsqueeze(1), curr_all_dims[i], -torch.inf)
            x_U[:, i] = upper_choices.min(dim=1)[0]
            x_L[:, i] = lower_choices.max(dim=1)[0]

        assert not torch.isinf(x_U).any(), "x_U has inf element"
        assert not torch.isinf(x_L).any(), "x_L has inf element"


    # Get the entries where domains were not already verified to perform evaluation metrics
    x_L_nv, x_U_nv = x_L[not_verified], x_U[not_verified]

    # performs evaluation metrics
    num_shrunken_and_verified = (x_L_nv > x_U_nv).any(1).sum(0).item()
    new_areas = torch.clamp(x_U_nv - x_L_nv, min=0.).prod(1)
    new_total_area = new_areas.sum(0).item()
    shrunken_areas = original_areas - new_areas
    shrunken_total_area = shrunken_areas.sum(0).item()
    num_shrunken = (new_areas < original_areas).to(dtype=torch.int).sum(0).item()
    print(
        f"Domain clipping: area new/prev {new_total_area:.4f}/{original_total_area:.4f} ({100 * (shrunken_total_area / original_total_area) if original_total_area > 0 else 0.:.2f}%), domains verified after shrinking {num_shrunken_and_verified}, shrunken {num_shrunken} of {batches} batches ({100*(num_shrunken / batches) if batches > 0 else 0.:.2f}%)")

    # reshape x_L,x_U to originally given shape and discover how many batches were shrunken
    x_L, x_U = x_L.view(x_L_shape), x_U.view(x_U_shape)

    return x_L, x_U
