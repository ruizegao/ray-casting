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
        if crown_mode.lower() == 'dynamic-forward' or crown_mode.lower() == 'dynamic-forward+backward':
            self.bounded_func = BoundedModule(crown_func, torch.empty((batch_size_per_iteration, 3)), bound_opts={"dynamic_forward": True})
            if crown_mode.lower() == 'dynamic-forward':
                crown_mode = 'forward'
            else:
                crown_mode = 'forward+backward'
        else:
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
        ptb = PerturbationLpNorm(x_L=box_center-box_vecs, x_U=box_center+box_vecs)
        bounded_x = BoundedTensor(box_center, ptb)
        may_lower, may_upper = self.bounded_func.compute_bounds(x=(bounded_x,), method=self.crown_mode) # dynamic forward
        output_type = torch.full_like(may_lower, SIGN_UNKNOWN)
        output_type = output_type.where(may_lower <= offset, torch.full_like(may_lower, SIGN_POSITIVE))
        output_type = output_type.where(may_upper >= -offset, torch.full_like(may_lower, SIGN_NEGATIVE))

        return output_type


    def classify_box(self, params, box_lower, box_upper, offset=0.):
        ptb = PerturbationLpNorm(x_L=box_lower.float(), x_U=box_upper.float())
        bounded_x = BoundedTensor(box_lower.float(), ptb)
        may_lower, may_upper = self.bounded_func.compute_bounds(x=(bounded_x,), method=self.crown_mode, bound_upper=False)
        output_type = torch.full_like(may_lower, SIGN_UNKNOWN)
        output_type = output_type.where(may_lower <= offset, torch.full_like(may_lower, SIGN_POSITIVE))
        # output_type = output_type.where(may_upper >= -offset, torch.full_like(may_lower, SIGN_NEGATIVE))

        return output_type

    def bound_box(self, params, box_lower, box_upper):
        box_lower = box_lower.unsqueeze(0)
        box_upper = box_upper.unsqueeze(0)
        ptb = PerturbationLpNorm(x_L=box_lower, x_U=box_upper)
        bounded_x = BoundedTensor(box_lower, ptb)
        may_lower, may_upper = self.bounded_func.compute_bounds(x=(bounded_x,),
                                                                method=self.crown_mode)  # dynamic forward
        return may_lower, may_upper

def change_mode(self, new_mode):
        self.crown_mode = new_mode
