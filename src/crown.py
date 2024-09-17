import gc
from functools import partial
import dataclasses
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

import utils
import torch
import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.DoubleTensor)

batch_size_per_iteration = 100000

# === Crown utility methods

def deconstruct_lbias(_x_L, _x_U, _lA, _dm_lb):
    """

    Given the input region, Crown's lA matrix and domain lower bounds, deconstructing lbias is very trivial.

    :param _x_L:
    :param _x_U:
    :param _lA:
    :param _dm_lb:
    :return:
    """
    _lA = _lA.flatten(2) # (batch, spec_dim, in_dim)
    xhat_vect = ((_x_U + _x_L) / 2).flatten(1) # (batch, in_dim)
    xhat_vect = xhat_vect.unsqueeze(2) # (batch, in_dim, 1)
    eps_vect = ((_x_U - _x_L) / 2).flatten(1) # (batch, in_dim)
    eps_vect = eps_vect.unsqueeze(2) # (batch, in_dim, 1)
    dm_lb_vect = _dm_lb.unsqueeze(2) # (batch, spec_dim, 1)
    _lbias = dm_lb_vect - (_lA.bmm(xhat_vect) - _lA.abs().bmm(eps_vect))
    return _lbias.squeeze(2) # (batch, spec_dim)

# === Function wrappers

class CrownImplicitFunction(implicit_function.ImplicitFunction):

    def __init__(self, implicit_func, crown_func, crown_mode='CROWN', enable_clipping=False):
        super().__init__("classify-and-distance")
        self.implicit_func = implicit_func
        self.bounded_func = BoundedModule(crown_func, torch.empty((batch_size_per_iteration, 3)), bound_opts={"relu": "same-slope"})
        self.crown_mode = crown_mode
        self._enable_clipping = enable_clipping
        print(self.crown_mode)

    def __call__(self, params, x):
        # x_device = x.to(device)
        # self.crown_func.to(device)
        # return self.crown_func(x)
        return self.implicit_func(params, x)


    def call_implicit_func(self, params, x):
        return self.implicit_func(params, x)

    # the parent class automatically delegates to this
    # def classify_box(self, params, box_lower, box_upper):
    # pass

    def classify_general_box(self, params, box_center, box_vecs, offset=0.):
        # evaluate the function
        ptb = PerturbationLpNorm(x_L=box_center-box_vecs, x_U=box_center+box_vecs)
        bounded_x = BoundedTensor(box_center, ptb)
        return_A = self._enable_clipping
        result = self.bounded_func.compute_bounds(x=(bounded_x,), method=self.crown_mode,
                                                                    return_A=return_A) # dynamic forward
        if return_A:
            may_lower, may_upper, A_dict = result
        else:
            may_lower, may_upper = result
        output_type = torch.full_like(may_lower, SIGN_UNKNOWN)
        output_type = output_type.where(may_lower <= offset, torch.full_like(may_lower, SIGN_POSITIVE))
        output_type = output_type.where(may_upper >= -offset, torch.full_like(may_lower, SIGN_NEGATIVE))

        return output_type


    def classify_box(self, params, box_lower, box_upper, offset=0.):
        ptb = PerturbationLpNorm(x_L=box_lower.float(), x_U=box_upper.float())
        bounded_x = BoundedTensor(box_lower.float(), ptb)
        return_A = self._enable_clipping

        # prepare A_dict to retrieve final lA
        if return_A:
            needed_A_dict = defaultdict(set)
            needed_A_dict[self.bounded_func.output_name[0]].add(self.bounded_func.input_name[0])
        else:
            needed_A_dict = None

        result = self.bounded_func.compute_bounds(x=(bounded_x,), method=self.crown_mode, bound_upper=False,
                                                                return_A=return_A, needed_A_dict=needed_A_dict)

        if return_A:
            may_lower, may_upper, A_dict = result
            lA = A_dict[self.bounded_func.output_name[0]][self.bounded_func.input_name[0]]['lA']
            lbias = A_dict[self.bounded_func.output_name[0]][self.bounded_func.input_name[0]]['lbias']
        else:
            may_lower, may_upper = result
            lA = None
            lbias = None

        output_type = torch.full_like(may_lower, SIGN_UNKNOWN)
        output_type = output_type.where(may_lower <= offset, torch.full_like(may_lower, SIGN_POSITIVE))
        # output_type = output_type.where(may_upper >= -offset, torch.full_like(may_lower, SIGN_NEGATIVE))

        if return_A:
            crown_ret = {
                "dm_lb": may_lower.detach(),
                "lA": lA,
                "lbias": lbias
            }
            return output_type, crown_ret
        else:
            return output_type, None

def change_mode(self, new_mode):
        self.crown_mode = new_mode
