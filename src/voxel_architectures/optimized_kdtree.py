import torch
import numpy as np
from functools import vmap
import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE

from crown import CrownImplicitFunction
from crown import deconstruct_lbias
from heuristics import input_split_branching

from split import kd_split

from math import log2

# typing
from torch import Tensor
from numpy import ndarray
from typing import Union, Tuple, Optional

torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ShortStackKDTree:
    """
    An implementation of optimized KD tree from 'Interactive k-D Tree GPU Raytracing'
    """

    def __init__(self):
        self._nodes = None
        self._hyperplanes = None
        self._root = None

        self._node_lower: Optional[Tensor] = None
        self._node_upper: Optional[Tensor] = None
        self._node_type: Optional[Tensor] = None
        self._split_dim: Optional[Tensor] = None
        self._split_val: Optional[Tensor] = None
        self._num_nodes: int = 0
        self._num_leaf_nodes: int = 0


    def _eval_one_node(
            self,
            func,
            params,
            offset: float,
            lower: Tensor,
            upper: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Bounds a single node
        :param func:
        :param params:
        :param offset:
        :param lower:
        :param upper:
        :return:
        """
        # perform an affine evaluation
        node_type = func.classify_box(params, lower, upper, offset=offset)
        # use the largest length along any dimension as the split policy
        worst_dim = torch.argmax(upper - lower, dim=-1)
        return node_type.float(), worst_dim.float()

    def _eval_batch_of_nodes(
            self,
            func,
            params,
            offset: float,
            lower: Tensor,
            upper: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Bounds a batch of nodes
        :param func:
        :param params:
        :param offset:
        :param lower:
        :param upper:
        :return:
        """
        node_type, _ = func.classify_box(params, lower, upper, offset=offset)
        node_type = node_type.squeeze(-1)
        worst_dim = torch.argmax(upper - lower, dim=-1)
        return node_type.float(), worst_dim.float()

    def _construct_full_uniform_unknown_levelset_tree_iter(
            self,
            func,
            params,
            continue_splitting,
            node_lower,
            node_upper,
            split_level,
            offset=0.,
            batch_size=None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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

    def build(
            self,
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
            lower, upper, out_node_type, out_split_dim, out_split_val = self._construct_full_uniform_unknown_levelset_tree_iter(
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
                # do not add the new split nodes (node_lower, node_upper, etc.)
                break

        node_lower = torch.cat(node_lower)
        node_upper = torch.cat(node_upper)
        node_type = torch.cat(node_type)
        split_dim = torch.cat(split_dim)
        split_val = torch.cat(split_val)

        # save the above tree to this class
        self._node_lower = node_lower
        self._node_upper = node_upper
        self._node_type = node_type
        self._split_dim = split_dim
        self._split_val = split_val

        # for key in out_dict:
        #     print(key, out_dict[key][:10])
        print("Total number of nodes evaluated: ", N_total_nodes)
        # if save_to:
        #     tree = {}
        #     tree['node_lower'] = node_lower.detach().cpu().numpy()
        #     tree['node_upper'] = node_upper.detach().cpu().numpy()
        #     tree['node_type'] = node_type.detach().cpu().numpy()
        #     tree['split_dim'] = split_dim.detach().cpu().numpy()
        #     tree['split_val'] = split_val.detach().cpu().numpy()
        #     np.savez(save_to, **tree)
        return node_lower, node_upper, node_type, split_dim, split_val

    ########################
    ####    TRAVERSAL   ####
    ########################

    @staticmethod
    def calc_child_indices(left: bool, p_idx: int, c_idx: int) -> Tuple[int, int, int, int]:
        split_depth = int(log2(p_idx))  # use the parent idx to calculate the current depth
        new_split_depth = split_depth + 1  # the split depth of the child idx
        new_p_idx = 2 ** new_split_depth  # the idx of the new parent of the child

        # calculate the idx of the new child based on if we want the left or right child
        if left:
            new_c_idx = 2 * (c_idx - p_idx) + new_p_idx
        else:
            new_c_idx = 2 * (c_idx - p_idx) + new_p_idx + 1

        # calculate the children of the child index
        new_l_idx = 2 * (new_c_idx - new_p_idx) + 2 ** (new_split_depth + 1)
        new_r_idx = new_l_idx + 1

        return new_p_idx, new_c_idx, new_l_idx, new_r_idx

    class _Node:
        def __init__(self, p_idx: int, n_idx: int, l_idx: int, r_idx: int):
            self._node: list[int] = [p_idx, n_idx, l_idx, r_idx]

        def __iter__(self):
            return iter(self._node)

        # properties
        @property
        def p_idx(self):
            return self._node[0]
        @property
        def c_idx(self):
            return self._node[1]
        @property
        def l_idx(self):
            return self._node[2]
        @property
        def r_idx(self):
            return self._node[3]
        @property
        def left(self):
            new_idx = ShortStackKDTree.calc_child_indices(True, *self[:2])
            return self.__init__(*new_idx)
        @property
        def right(self):
            new_idx = ShortStackKDTree.calc_child_indices(False, *self[:2])
            return self.__init__(*new_idx)

    class _StackObj:
        def __init__(self, node, tMin_v: Tensor, tMax_v: Tensor, live_v: Tensor):
            self._node = node
            self._tMin_v: Tensor = tMin_v
            self._tMax_v: Tensor = tMax_v
            self._live_v: Tensor = live_v

        def __iter__(self):
            return iter((self._node, self._tMin_v, self._tMax_v, self._live_v))

        # properties
        @property
        def node(self):
            return self._node
        @property
        def tMin_v(self):
            return self._tMin_v
        @property
        def tMax_v(self):
            return self._tMax_v
        @property
        def live_v(self):
            return self._live_v

    def _isLeaf(self, node: int):
        return node >= self._num_leaf_nodes

    def traverse(self, rays_v: Tensor):
        n = len(rays_v)  # number of rays
        # separate ray information into origin and direction
        rays_v_origin, rays_v_dir = rays_v[0], rays_v[1]
        sceneMin, sceneMax = self._node_lower[0], self._node_upper[0]  # the bounds of the scene
        tMin_v = torch.full((n,), sceneMin)
        tMax_v = torch.full((n,), sceneMax)
        tHit_v = torch.full((n,), torch.inf)
        live_v = torch.ones((n,), dtype=torch.bool)
        done_v = torch.zeros(n, dtype=torch.bool)
        didstack = False

        # root = self._Node(0, 0, 1, 2)  # get root of the kd tree
        root = 0
        # stack = [self._StackObj(root, sceneMin, sceneMax, live_v)]
        stack = [(root, sceneMin, sceneMax, live_v.clone())]

        while (didstack or not done_v.all()) and not (tHit_v < tMax_v).all():

            node = root
            if not (len(stack) == 0):
                didstack = True
                node_v, tMin_v, tMax_v, live_v = stack.pop()
                pushdown = False
            else:
                didstack = False
                node = root
                live_v = torch.logical_and(tMin_v <= tMax_v, ~done_v)
                pushdown = True

            while not self._isLeaf(node):

                a = self._split_dim[node]  # get split axis
                value = self._split_val[node]  # get split value
                tSplit_v = (value - rays_v_origin[a]) / rays_v_dir[a]
                first, sec = order(rays_v_dir[a], node.left, node.right)
                wantNear_v = torch.logical_and(tSplit_v > tMin_v, live_v)
                wantFar_v = torch.logical_and(tSplit_v <= tMax_v, live_v)

                if torch.logical_or(wantNear_v, ~live_v).all() and not wantFar_v.all():
                    node = first
                elif torch.logical_or(wantFar_v, ~live_v).all() and not wantNear_v.any():
                    node = second
                else:
                    pushdown = False
                    node = first
                    top_live_v = torch.logical_and(live_v, wantFar_v)

                    top_tMin_v = tMin_v.clone()
                    top_tMin_v[top_live_v] = torch.max(tMin_v[top_live_v], tSplit_v[top_live_v])

                    stack.append(self._StackObj(second, top_tMin_v, tMax_v, top_live_v))

                    live_v = wantNear_v.clone()
                    tMax_v[live_v] = torch.min(tSplit_v[live_v], tMax_v[live_v])

                if pushdown:
                    root = node

            for tri in node.triangles:
                tHit_v = torch.min(tHit_v, tri.intersect(ray))

                if (tHit_v < tMax_v).all():
                    return tHit_v

            done_v = torch.logical_or(done_v, tHit_v <= tMax_v)
            tMin_v = tMax_v
            tMax_v = sceneMax

        return tHit_v