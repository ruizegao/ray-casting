import gc
from functools import partial
import dataclasses
import itertools
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
torch.set_default_tensor_type(torch.cuda.FloatTensor)

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

    def __init__(self, implicit_func, crown_func, crown_mode='CROWN', enable_clipping=False, obj_name=''):
        super().__init__("classify-and-distance")
        self.implicit_func = implicit_func
        self.torch_model = crown_func
        if crown_mode.lower() == 'alpha-crown':
            self.reuse_alpha = True
            self.bounded_func = BoundedModule(crown_func, torch.empty((batch_size_per_iteration, 3)), bound_opts={'optimize_bound_args': {'iteration': 20}})#, 'relu': 'same-slope'})
        else:
            self.reuse_alpha = False
            self.bounded_func = BoundedModule(crown_func, torch.empty((batch_size_per_iteration, 3)))#, bound_opts={'relu': 'same-slope'})

        self.crown_mode = crown_mode
        self._enable_clipping = enable_clipping
        if enable_clipping:
            self.bounding_method = crown_mode+'_clipping'
        else:
            self.bounding_method = crown_mode
        self.obj_name = obj_name
        print(self.crown_mode, enable_clipping)

    def __call__(self, params, x):
        # x_device = x.to(device)
        # self.crown_func.to(device)
        # return self.crown_func(x)
        return self.implicit_func(params, x)

    def torch_forward(self, x):
        return self.torch_model(x)

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
                                                                    return_A=return_A, reuse_alpha=self.reuse_alpha) # dynamic forward
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
        may_lower, may_upper = self.bounded_func.compute_bounds(x=(bounded_x,), method=self.crown_mode, bound_upper=True)
        # unstable_counts = []
        # for k, v in bound_dict.items():
        #     if 'input' in k:
        #         unstable_counts.append(torch.logical_and(v[0] < 0, v[1] > 0).sum().item())
        ptb = PerturbationLpNorm(x_L=box_lower.float(), x_U=box_upper.float())
        bounded_x = BoundedTensor(box_lower.float(), ptb)
        # return_A = self._enable_clipping
        return_A = True
        # prepare A_dict to retrieve final lA
        if return_A:
            needed_A_dict = defaultdict(set)
            needed_A_dict[self.bounded_func.output_name[0]].add(self.bounded_func.input_name[0])
        else:
            needed_A_dict = None

        result = self.bounded_func.compute_bounds(x=(bounded_x,), method=self.crown_mode, bound_upper=True,
                                                                return_A=return_A, needed_A_dict=needed_A_dict)

        if return_A:
            may_lower, may_upper, A_dict = result
            lA = A_dict[self.bounded_func.output_name[0]][self.bounded_func.input_name[0]]['lA']
            lbias = A_dict[self.bounded_func.output_name[0]][self.bounded_func.input_name[0]]['lbias']
            uA = A_dict[self.bounded_func.output_name[0]][self.bounded_func.input_name[0]]['uA']
            ubias = A_dict[self.bounded_func.output_name[0]][self.bounded_func.input_name[0]]['ubias']
        else:
            may_lower, may_upper = result
            lA = None
            lbias = None
            uA = None
            ubias = None

        torch.set_printoptions(threshold=float('inf'), precision=4)
        # output_bounds = [may_lower.item(), may_upper.item()]
        # print(*(output_bounds + unstable_counts))
        output_type = torch.full_like(may_lower, SIGN_UNKNOWN)
        output_type = output_type.where(may_lower <= offset, torch.full_like(may_lower, SIGN_POSITIVE))
        output_type = output_type.where(may_upper >= -offset, torch.full_like(may_lower, SIGN_NEGATIVE))

        if return_A:
            crown_ret = {
                "dm_lb": may_lower.detach(),
                "lA": lA,
                "lbias": lbias,
                "uA": uA,
                "ubias": ubias
            }
            return output_type, crown_ret
        else:
            return output_type, None

    def change_mode(self, new_mode):
            self.crown_mode = new_mode

    def enable_mesh_optimization(self):
        print("mesh optimization enabled")
        def get_domain_loss(x, ret, output_name, input_name):
            r"""
            x_L: (batch_size, input_size)
            x_U: (batch_size, input_size)
            A: {
                'lower_A' (and 'upper_A'): (batch_size, output_size, input_size)
                'lower_b' (and 'upper_b'): (batch_size, output_size)
                }
            """
            x_L = x[0].ptb.x_L
            x_U = x[0].ptb.x_U
            A = ret[2][output_name][input_name]

            xhat = ((x_U + x_L) / 2).unsqueeze(-1)
            eps = ((x_U - x_L) / 2).unsqueeze(-1)

            # Instead of directly using A['lbias'], we compute lbias from lower bounds to make use of the gradient
            n_lower = A['lA'].permute(0, 2, 1)
            d_lower = (ret[0].unsqueeze(-1) - A['lA'].bmm(xhat) + A['lA'].abs().bmm(eps)).squeeze(-1)
            # assert torch.allclose(d_lower, A['lbias'])
            n_upper = A['uA'].permute(0, 2, 1)
            d_upper = (ret[1].unsqueeze(-1) - A['uA'].bmm(xhat) - A['uA'].abs().bmm(eps)).squeeze(-1)
            # assert torch.allclose(d_upper, A['ubias'])

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

                average_intersections = torch.einsum('bij, bi -> bij', edge_intersections, valid_intersections).mean(
                    dim=1)

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

        self.bounded_func = BoundedModule(self.torch_model, torch.empty((batch_size_per_iteration, 3)), device=device,
                                          bound_opts={'sparse_intermediate_bounds': False,
                                                      'sparse_features_alpha': False,
                                                      'optimize_bound_args': {'keep_best': False,'iteration': 5,
                                                                              'use_custom_loss': True,
                                                                              'custom_loss_func': get_domain_loss,
                                                                              'joint_optimization': True
                                                                              }
                                                      })
        self.crown_mode = 'CROWN-Optimized'