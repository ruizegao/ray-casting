import torch
import itertools
import argparse
import numpy as np

def get_distance(x_L, x_U, n_lower, d_lower, n_upper, d_upper):
    r"""
    x_L: (batch_size, input_size)
    x_U: (batch_size, input_size)

    n_lower (n_upper): (batch_size, 3, 1)
    d_lower (d_upper): (batch_size, 1)

    plane equation: n[0]*x + n[1]*y + n[2]*z + d = 0
    """
    device = x_L.device
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

        average_intersections = torch.einsum('bij, bi -> bij', edge_intersections, valid_intersections).mean(dim=1)

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("load_from", type=str)
    args = parser.parse_args()

    node_lower, node_upper, mAs, mbs, lAs, lbs, uAs, ubs = [torch.from_numpy(val).to(device) for val in np.load(args.load_from).values()]
    distance = get_distance(node_lower, node_upper, uAs.unsqueeze(-1), ubs.unsqueeze(-1), lAs.unsqueeze(-1), lbs.unsqueeze(-1))
    print(torch.sqrt(torch.norm(node_upper - node_lower, dim=1)))
    print(node_upper[0] - node_lower[0])
    print(torch.sqrt(distance).mean())
    print(torch.sqrt(distance).std())

if __name__ == '__main__':
    main()
