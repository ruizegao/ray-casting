import time
import shapely
import shapely.set_operations
import random

from neural_utils import load_net_object
from bouncing_letters import carve, scale_polygon
import torch
import torch.nn as nn
import numpy as np


@torch.jit.script
def detect_circle_polygon_collision_batch(circle_centers: torch.Tensor, circle_radius: torch.Tensor,
                                          polygon_vertices: torch.Tensor) -> torch.Tensor:
    """
    Detects if a batch of circles collides with a polygon (either intersects or is contained).

    Args:
        circle_centers: Tensor of shape (B, 2) where B is the batch size, representing circle centers.
        circle_radius: Scalar or Tensor of shape (B,) representing the radii of the circles.
        polygon_vertices: Tensor of shape (N, 2) representing polygon vertices.

    Returns:
        collision_mask: Boolean tensor of shape (B,) indicating whether each circle in the batch collides with the polygon.
    """
    B = circle_centers.shape[0]  # Batch size
    N = polygon_vertices.shape[0]  # Number of vertices

    # Shift vertices to form edges
    edges_start = polygon_vertices
    edges_end = torch.roll(polygon_vertices, shifts=-1, dims=0)  # Shift vertices to form edges

    # Compute closest points on edges to the circle centers
    vw = edges_end - edges_start
    pv = circle_centers.unsqueeze(1) - edges_start

    t = (pv * vw).sum(dim=-1) / (vw * vw).sum(dim=-1).clamp(min=1e-6)
    t = torch.clamp(t, 0, 1)  # Clamp to segment

    closest_points = edges_start + t.unsqueeze(-1) * vw  # Closest points

    # Compute distances from closest points to the circle centers
    distances = torch.norm(closest_points - circle_centers.unsqueeze(1), dim=-1)

    # Check intersection: if any closest point distance < radius
    intersection = distances < circle_radius.unsqueeze(-1)

    # Check containment using ray-casting method
    v1, v2 = edges_start, edges_end

    condition1 = (v1[:, 1] > circle_centers[:, 1].unsqueeze(1)) != (v2[:, 1] > circle_centers[:, 1].unsqueeze(1))
    slope = (v2[:, 0] - v1[:, 0]) / (v2[:, 1] - v1[:, 1] + 1e-6)
    x_intersect = v1[:, 0] + slope.unsqueeze(0) * (circle_centers[:, 1].unsqueeze(1) - v1[:, 1])

    # Ignore horizontal edges
    non_horizontal = v1[:, 1] != v2[:, 1]
    condition2 = circle_centers[:, 0].unsqueeze(1) < x_intersect

    intersections = ((condition1 & condition2) & non_horizontal).sum(dim=1)  # Count ray crossings

    contained = intersections % 2 == 1 # Odd crossings and no intersection

    collision_mask = intersection.any(dim=1) | contained  # Union of intersection and containment

    return collision_mask


set_t = {
    'dtype': torch.float32,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

def main():
    c_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    c_net = c_net.to(device=set_t['device'])
    c_polygon_l = carve(c_net, deep=False, smoothify=False, return_merged=True)
    c_polygon_t = carve(c_net, deep=True, smoothify=False, return_merged=True)
    i_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/I_MLP.pth', 'mlp')
    i_net = i_net.to(device=set_t['device'])
    i_polygon_t = carve(i_net, deep=True, smoothify=True, return_merged=True)
    # avg_time_mesh, avg_time_mlp, wrong_results = measure_intersection_time(c_polygon_t, i_polygon_t)
    # print(f'Average intersection check time: {avg_time_mesh:.6f} seconds')
    #
    # # avg_time_mlp = measure_intersection_time_mlp(num_trials=1000)
    # print(f'Average intersection check time with MLP: {avg_time_mlp:.6f} seconds')
    # print(f'Number of incorrect checks: {wrong_results}')

    c_polygon_t_vertices = torch.tensor(c_polygon_t.exterior.coords, device=set_t['device'])
    c_polygon_l_vertices = torch.tensor(c_polygon_l.exterior.coords, device=set_t['device'])
    bbox = c_polygon_t.envelope
    bbox_vertices = torch.tensor(bbox.exterior.coords, device=set_t['device'])
    # test_circle_centers = torch.rand(10000, 2) - 0.5
    # test_circle_radius = torch.rand(10000,) * 0.5
    num_trials = 1000000
    test_circle_centers = torch.from_numpy(np.random.uniform(-0.5, 0.5, (num_trials, 2))).float().cuda()
    test_circle_radius = torch.from_numpy(np.random.uniform(0.01, 0.1, (num_trials,))).float().cuda()
    # test_circle_centers.cuda()
    # test_circle_radius.cuda()
    compiled_fn = torch.compile(detect_circle_polygon_collision_batch)
    # Warmup
    mesh_results = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_t_vertices)
    mesh_results = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_t_vertices)
    mesh_results = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_t_vertices)
    mesh_results_not_early_outs = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_l_vertices)
    mesh_results_early_outs = ~mesh_results_not_early_outs
    mesh_results_ = compiled_fn(test_circle_centers[~mesh_results_early_outs],
                                test_circle_radius[~mesh_results_early_outs], c_polygon_t_vertices)
    mesh_results = ~mesh_results_early_outs
    mesh_results[mesh_results.clone()] = mesh_results_
    mesh_results_not_early_outs = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_l_vertices)
    mesh_results_early_outs = ~mesh_results_not_early_outs
    mesh_results_ = compiled_fn(test_circle_centers[~mesh_results_early_outs],
                                test_circle_radius[~mesh_results_early_outs], c_polygon_t_vertices)
    mesh_results = ~mesh_results_early_outs
    mesh_results[mesh_results.clone()] = mesh_results_
    mesh_results_not_early_outs = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_l_vertices)
    mesh_results_early_outs = ~mesh_results_not_early_outs
    mesh_results_ = compiled_fn(test_circle_centers[~mesh_results_early_outs],
                                test_circle_radius[~mesh_results_early_outs], c_polygon_t_vertices)
    mesh_results = ~mesh_results_early_outs
    mesh_results[mesh_results.clone()] = mesh_results_
    time_0 = time.perf_counter()
    mesh_results_not_early_outs = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_l_vertices)
    mesh_results_early_outs = ~mesh_results_not_early_outs
    mesh_results_ = compiled_fn(test_circle_centers[~mesh_results_early_outs], test_circle_radius[~mesh_results_early_outs], c_polygon_t_vertices)
    print(mesh_results.sum(), (~mesh_results).sum())
    mesh_results = ~mesh_results_early_outs
    mesh_results[mesh_results.clone()] = mesh_results_
    # mesh_results = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_t_vertices)

    time_1 = time.perf_counter()
    mesh_time = time_1 - time_0
    print(f'mesh time {mesh_time * 1000:.3f}')
    print(f'mesh num of early outs {mesh_results_early_outs.sum()}')
    # scripted_c_net = torch.jit.script(c_net)
    scripted_c_net = c_net
    # Warmup
    _ = compiled_fn(test_circle_centers, test_circle_radius, bbox_vertices)

    sdf_distances = scripted_c_net(test_circle_centers)
    sdf_distances = scripted_c_net(test_circle_centers)
    sdf_distances = scripted_c_net(test_circle_centers)
    sdf_distances = scripted_c_net(test_circle_centers)
    time_2 = time.perf_counter()
    sdf_results_not_early_outs = compiled_fn(test_circle_centers, test_circle_radius, bbox_vertices)
    sdf_results_early_outs = ~sdf_results_not_early_outs
    sdf_distances = scripted_c_net(test_circle_centers).squeeze(-1)
    time_3 = time.perf_counter()
    mlp_time = time_3 - time_2
    print(f'mlp time {mlp_time * 1000:.3f}')
    print(f'mlp num of early outs {sdf_results_early_outs.sum()}')
    print(mlp_time / mesh_time)
    sdf_results = sdf_distances <= test_circle_radius
    mesh_results = mesh_results.detach().cpu().numpy()
    print((~mesh_results).sum())
    sdf_results = sdf_results.detach().cpu().numpy()
    print(np.sum(mesh_results != sdf_results))
if __name__ == "__main__":
    main()
