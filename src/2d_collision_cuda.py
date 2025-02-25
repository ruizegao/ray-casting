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
def point_to_polygon_distance(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """
    Computes the minimum distances from a batch of points to a single polygon using matrix operations.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing N points.
        polygon (torch.Tensor): Tensor of shape (M, 2) representing M vertices of the polygon.

    Returns:
        torch.Tensor: Tensor of shape (N,) containing the minimum distances for each point.
    """
    # Get polygon edges
    edges_start = polygon  # (M, 2)
    edges_end = torch.roll(polygon, shifts=-1, dims=0)  # Shift to form edges (M, 2)

    # Vector from edge start to end
    edge_vec = edges_end - edges_start  # (M, 2)
    edge_length_sq = torch.sum(edge_vec ** 2, dim=1, keepdim=True)  # (M, 1)

    # Expand dimensions for batch processing
    points_exp = points.unsqueeze(1)  # (N, 1, 2)
    edges_start_exp = edges_start.unsqueeze(0)  # (1, M, 2)
    edge_vec_exp = edge_vec.unsqueeze(0)  # (1, M, 2)

    # Vector from edge start to points
    start_to_points = points_exp - edges_start_exp  # (N, M, 2)

    # Project points onto the edges
    proj = torch.sum(start_to_points * edge_vec_exp, dim=2, keepdim=True) / (edge_length_sq + 1e-10)  # (N, M, 1)
    proj = torch.clamp(proj, 0, 1)  # Clamp to edge segment

    # Closest points on the edges
    closest = edges_start_exp + proj * edge_vec_exp  # (N, M, 2)

    # Compute distances
    distances = torch.norm(points_exp - closest, dim=2)  # (N, M)

    # Minimum distance for each point
    min_distances, _ = torch.min(distances, dim=1)  # (N,)

    return min_distances


@torch.jit.script
def closest_point_on_segment(p: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Compute the closest point on the segment vw to point p.
    """
    vw = w - v  # Edge vector
    pv = p - v  # Vector from v to circle center

    t = (pv * vw).sum(dim=-1) / (vw * vw).sum(dim=-1).clamp(min=1e-6)
    t = torch.clamp(t, 0, 1)  # Clamp to segment

    return v + t.unsqueeze(-1) * vw  # Closest point


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

def generate_random_offset(shape):
    xoff = random.uniform(-1., 1.)
    yoff = random.uniform(-1., 1.)
    return shapely.affinity.translate(shape, xoff, yoff)

def measure_intersection_time(polygon_1, polygon_2, num_trials=1000):
    total_time_mesh = 0
    total_time_mlp = 0
    wrong_check_count = 0
    for i in range(num_trials):
        polygon_1_temp = generate_random_offset(polygon_1)
        polygon_2_temp = generate_random_offset(polygon_2)
        start_time_mesh = time.perf_counter()
        # mesh_flags = shapely.intersects(polygon_1_temp, polygon_2_temp)
        mesh_intersection = shapely.intersection(polygon_1_temp, polygon_2_temp)
        total_time_mesh += time.perf_counter() - start_time_mesh
        # start_time_mlp = time.perf_counter()
        # mlp_flags = check_intersection_mlp(centers_torch, radii_torch, c_net)
        # total_time_mlp += time.perf_counter() - start_time_mlp

    return total_time_mesh / num_trials, total_time_mlp, wrong_check_count

def main():
    c_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    c_net = c_net.to(device=set_t['device'])
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
    test_circle_centers = torch.rand(10000, 2) - 0.5
    test_circle_radius = torch.rand(10000,) * 0.5
    test_circle_centers.cuda()
    test_circle_radius.cuda()
    compiled_fn = torch.compile(detect_circle_polygon_collision_batch)
    mesh_results = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_t_vertices)
    mesh_results = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_t_vertices)
    mesh_results = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_t_vertices)
    mesh_results = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_t_vertices)
    time_0 = time.perf_counter()


    mesh_results = compiled_fn(test_circle_centers, test_circle_radius, c_polygon_t_vertices)
    # distances = point_to_polygon_distance(test_points, c_polygon_t_vertices)
    time_1 = time.perf_counter()
    mesh_time = time_1 - time_0
    print('mesh time', mesh_time)

    # scripted_c_net = torch.jit.script(c_net)
    scripted_c_net = c_net

    sdf_distances = scripted_c_net(test_circle_centers)
    sdf_distances = scripted_c_net(test_circle_centers)
    sdf_distances = scripted_c_net(test_circle_centers)
    sdf_distances = scripted_c_net(test_circle_centers)
    time_2 = time.perf_counter()
    sdf_distances = scripted_c_net(test_circle_centers).squeeze(-1)
    time_3 = time.perf_counter()
    mlp_time = time_3 - time_2
    print('mlp time', mlp_time)
    print(mlp_time / mesh_time)
    sdf_results = sdf_distances <= test_circle_radius
    mesh_results = mesh_results.detach().cpu().numpy()
    sdf_results = sdf_results.detach().cpu().numpy()
    print(np.sum(mesh_results != sdf_results))
if __name__ == "__main__":
    main()
