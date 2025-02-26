import pymunk
import random
import time
import shapely
import shapely.set_operations

from neural_utils import load_net_object
from bouncing_letters import carve, scale_polygon
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

global c_net, C_COMP_L, C_COMP_T

set_t = {
    'dtype': torch.float32,
    'device': torch.device('cpu'),
}

def generate_circles(N=10000):
    centers_x = np.random.uniform(-3, 3, (N, 1))
    centers_y = np.random.uniform(-1, 1, (N, 1))
    centers = np.hstack((centers_x, centers_y))
    radii = np.random.uniform(0.01, 0.05, (N,))
    return centers, radii

def detect_circles_polygons_collision(centers, radii, polygons):
    return shapely.distance(centers, polygons) <= radii

def check_intersection_mlps(centers, radii, nets):
    return torch.tensor([net(center).squeeze() <= radius for center, radius, net in zip(centers, radii, nets)])

def measure_intersection_time(polygons, bb_strtree, mlps, num_trials=1000):
    total_time_mesh = 0
    total_time_mlp = 0
    centers, radii = generate_circles(N=num_trials)
    centers_torch, radii_torch = torch.from_numpy(centers).float().cpu(), torch.from_numpy(radii).float().cpu()
    centers = np.array([shapely.Point(p) for p in centers])
    start_time_mesh = time.perf_counter()
    nearest_indices, nearest_distances = bb_strtree.query_nearest(centers, all_matches=False, return_distance=True)
    mesh_flags = nearest_distances <= radii
    mesh_flags[mesh_flags] = detect_circles_polygons_collision(centers[mesh_flags], radii[mesh_flags], polygons[nearest_indices[1][mesh_flags]])
    total_time_mesh += time.perf_counter() - start_time_mesh
    start_time_mlp = time.perf_counter()
    nearest_indices, nearest_distances = bb_strtree.query_nearest(centers, all_matches=False, return_distance=True)
    mlp_flags = nearest_distances <= radii
    mlp_flags_torch = torch.from_numpy(mlp_flags)
    mlp_flags[mlp_flags] = check_intersection_mlps(centers_torch[mlp_flags_torch], radii_torch[mlp_flags_torch], mlps[nearest_indices[1][mlp_flags]]).detach().cpu().numpy()
    total_time_mlp += time.perf_counter() - start_time_mlp
    wrong_check_count = (mesh_flags != mlp_flags).sum()

    return total_time_mesh / num_trials, total_time_mlp / num_trials, wrong_check_count

def main():
    i_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/I_MLP.pth', 'mlp')
    i_polygon_t = carve(i_net, deep=True, smoothify=False, return_merged=True)
    i_polygon_t = shapely.affinity.translate(i_polygon_t, -1.5, 0)
    c_net_left = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    c_polygon_t_left = carve(c_net_left, deep=True, smoothify=False, return_merged=True)
    c_polygon_t_left = shapely.affinity.translate(c_polygon_t_left, -0.5, 0)
    c_net_right = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    c_polygon_t_right = carve(c_net_right, deep=True, smoothify=False, return_merged=True)
    c_polygon_t_right = shapely.affinity.translate(c_polygon_t_right, 0.5, 0)
    v_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/V_MLP.pth', 'mlp')
    v_polygon_t = carve(v_net, deep=True, smoothify=False, return_merged=True)
    v_polygon_t = shapely.affinity.translate(v_polygon_t, 1.5, 0)

    polygons = np.array([i_polygon_t, c_polygon_t_left, c_polygon_t_right, v_polygon_t])
    polygons_bb = np.array([shapely.envelope(poly) for poly in polygons])
    strtree = shapely.STRtree(polygons)
    bb_strtree = shapely.STRtree(polygons_bb)
    i_net.cpu()
    c_net_left.cpu()
    c_net_right.cpu()
    v_net.cpu()
    mlps = np.array([i_net, c_net_left, c_net_right, v_net])
    avg_time_mesh, avg_time_mlp, wrong_results = measure_intersection_time(polygons, bb_strtree, mlps)
    print(f'Average intersection check time: {avg_time_mesh:.6f} seconds')

    # avg_time_mlp = measure_intersection_time_mlp(num_trials=1000)
    print(f'Average intersection check time with MLP: {avg_time_mlp:.6f} seconds')
    print(f'Number of incorrect checks: {wrong_results}')

if __name__ == "__main__":
    main()
