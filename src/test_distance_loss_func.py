import torch
import itertools
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 80),
            nn.ReLU(),
            nn.Linear(80, 1)
        )
    
    def forward(self, x):
        return self.model(x)

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
    #indices:
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
        zeros_column = torch.zeros((2**(ndim - 1), 1), dtype=int, device=device)
        new_matrix = torch.cat((indices[:,:i], zeros_column, indices[:, i:]), dim=1)
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
    index_to_check_in_box = (all_indices == 0).nonzero()[:,2].reshape(bound_to_check_in_box.shape[0], -1)
    bound_to_check_in_box[:,:,0] = torch.gather(x_L, dim=1, index=index_to_check_in_box)
    bound_to_check_in_box[:,:,1] = torch.gather(x_U, dim=1, index=index_to_check_in_box)

    # All vertices of the box (batch_size, 2^input_size, input_size)
    binary_numbers = [list(map(int, bits)) for bits in itertools.product('12', repeat=ndim)]
    vertices_indices = torch.tensor(binary_numbers, device=device).unsqueeze(0).repeat(x_L.shape[0], 1, 1)
    all_vertices = torch.gather(input_domain, dim=1, index=vertices_indices)

    def _get_hook(n, d):
        # temp_cofficients[b][i][j] = input_domain[b][all_indices[b][i][j]][j]
        temp_edge_intersections = torch.gather(input_domain, dim=1, index=all_indices)
        temp_edge = torch.zeros_like(temp_edge_intersections, device=temp_edge_intersections.device)

        denominators = n.repeat(1, 1, 2**(ndim - 1)).flatten(1)
        intersections = -(torch.bmm(temp_edge_intersections, n).squeeze(-1) + d) / denominators
        temp_edge[all_indices == 0] = intersections.flatten()
        edge_intersections = temp_edge_intersections + temp_edge

        valid_intersections = torch.logical_and(intersections >= bound_to_check_in_box[:,:,0],
                                                intersections <= bound_to_check_in_box[:,:,1])

        average_intersections = torch.einsum('bij, bi -> bij', edge_intersections, valid_intersections).mean(dim=1)
    
        # Now compute the distances from vertices to planes
        # distance = (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2) (signed)
        all_distances = (torch.bmm(all_vertices, n).squeeze(-1) + d) / torch.norm(n, dim=1)

        completely_outside = torch.logical_or(torch.all(all_distances >= 0, dim=1), torch.all(all_distances <= 0, dim=1))

        shortest_distance, shortest_index = torch.min(torch.abs(all_distances), dim=1)

        # x_h = x - (ax + by + cz + d)/(a^2 + b^2 + c^2) * a
        feet_perpendicular = all_vertices - (all_distances / torch.norm(n, dim=1)).unsqueeze(-1) * n.unsqueeze(1).squeeze(-1)
        shortest_feet = feet_perpendicular[torch.arange(feet_perpendicular.shape[0]), shortest_index]

        chosen_feet = torch.einsum('bj, b -> bj', shortest_feet, completely_outside)

        hook = average_intersections + chosen_feet
        return hook

    hook_lower = _get_hook(n_lower, d_lower)
    hook_upper = _get_hook(n_upper, d_upper)
    domain_loss = torch.norm(hook_lower - hook_upper, dim=1)
    return domain_loss

def get_test_loss(ret, use_custom_loss):
    print("computing loss")
    print(ret)
    print(use_custom_loss)

if __name__ == '__main__':    
    device = torch.device('cuda')
    torch.manual_seed(100)

    # Define the model
    model = Model().to(device)
    # torch.save(model.state_dict(), 'test_model.pth')
    # model.load_state_dict(torch.load('test_model.pth'))

    input_width = model.model[0].in_features
    output_width = model.model[-1].out_features
    batch_size = 2
    eps = 1

    # Random test input
    x = torch.rand(batch_size, input_width, device=device)

    # Bound the input
    x_L = x - eps
    x_U = x + eps
    ptb = PerturbationLpNorm(x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x, ptb)

    # Bound the model
    lirpa_model = BoundedModule(model, torch.empty_like(x), device=device,
                                bound_opts={'sparse_intermediate_bounds': False,
                                            'sparse_features_alpha': False,
                                            'optimize_bound_args': {
                                                'keep_best': False,
                                                'iteration': 80,
                                                'use_custom_loss': True,
                                                'custom_loss_func': get_domain_loss,
                                                'joint_optimization': True
                                            }})

    for method in ['CROWN', 'CROWN-Optimized']:
        print("Bounding method: ", method)
        lb, ub, A_dict = lirpa_model.compute_bounds(x=(bounded_x,), method=method, return_A=True)
        for i in range(batch_size):
            for j in range(output_width):
                print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                    j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
            print('---------------------------------------------------------')
        print()