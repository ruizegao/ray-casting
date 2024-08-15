import gc
from functools import partial
import dataclasses
from dataclasses import dataclass

import numpy as np

import utils
import torch
import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

batch_size_per_iteration = 5000

# === Function wrappers

class CrownImplicitFunction(implicit_function.ImplicitFunction):

    def __init__(self, implicit_func, crown_func, crown_mode='CROWN'):
        super().__init__("classify-and-distance")
        self.implicit_func = implicit_func
        # self.crown_func = crown_func
        self.bounded_func = BoundedModule(crown_func, torch.empty((batch_size_per_iteration, 3)))
        self.crown_mode = crown_mode
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
        # print("box_vecs:" , box_vecs)
        ptb = PerturbationLpNorm(x_L=box_center-box_vecs, x_U=box_center+box_vecs)
        # eps = torch.norm(box_vecs, dim=1, keepdim=True) / torch.norm(box_center, dim=1, keepdim=True)
        # ptb = PerturbationLpNorm(eps=eps, norm=2.)
        bounded_x = BoundedTensor(box_center, ptb)
        may_lower, may_upper = self.bounded_func.compute_bounds(x=(bounded_x,), method=self.crown_mode)
        # may_lower = may_lower.flatten()
        # may_upper = may_upper.flatten()
        # print(may_lower.shape, may_upper.shape)
        output_type = torch.full_like(may_lower, SIGN_UNKNOWN)
        output_type = output_type.where(may_lower <= offset, torch.full_like(may_lower, SIGN_POSITIVE))
        output_type = output_type.where(may_upper >= -offset, torch.full_like(may_lower, SIGN_NEGATIVE))
        # print("lower: ", may_lower.flatten())
        # print("upper: ", may_upper.flatten())
        # print("output: ", output_type.flatten())

        return output_type


    # def classify_box(self, params, box_lower, box_upper, offset=0.):
    #     center = 0.5 * (box_lower + box_upper)
    #     pos_vec = box_upper - center
    #     # vecs = torch.diag(pos_vec)
    #     return self.classify_general_box(params, center, pos_vec, offset=offset)

    def classify_box(self, params, box_lower, box_upper, offset=0.):
        ptb = PerturbationLpNorm(x_L=box_lower.float(), x_U=box_upper.float())
        bounded_x = BoundedTensor(box_lower.float(), ptb)
        may_lower, may_upper = self.bounded_func.compute_bounds(x=(bounded_x,), method=self.crown_mode, bound_upper=False)
        output_type = torch.full_like(may_lower, SIGN_UNKNOWN)
        output_type = output_type.where(may_lower <= offset, torch.full_like(may_lower, SIGN_POSITIVE))
        # output_type = output_type.where(may_upper >= -offset, torch.full_like(may_lower, SIGN_NEGATIVE))

        return output_type

    def change_mode(self, new_mode):
        self.crown_mode = new_mode
