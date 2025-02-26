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
    centers = np.random.uniform(-2, 2, (N, 2))
    radii = np.random.uniform(0.1, 0.8, (N,))
    return centers, radii

def detect_circles_polygon_collision(centers, radii, polygon, bb):
    if shapely.distance(centers, bb) > radii:
        return np.array(False)

    distance = shapely.distance(centers, polygon)
    return distance <= radii

def check_intersection(shape, shapes):
    for s in shapes:
        if shape.shapes_collide(s).points:
            return True
    return False

def check_intersection_robust(shape, shapes_l, shapes_t, net):
    for s in shapes_l:
        if shape.shapes_collide(s).points:
            if check_intersection(shape, shapes_t):
                if check_intersection_mlp(net, shape.body.position):
                    return True
                # return check_intersection_mlp(net, shape.body.position)
                # return True
    return False

def check_intersection_mlp(centers, centers_shape, radii, radii_numpy, net, bb):
    if shapely.distance(centers_shape, bb) > radii_numpy:
        return torch.tensor(False)
    distance = net(centers).squeeze()
    return distance <= radii

def measure_intersection_time(polygon_t, bb, num_trials=10000):
    total_time_mesh = 0
    total_time_mlp = 0
    centers, radii = generate_circles(N=num_trials)
    centers_torch, radii_torch = torch.from_numpy(centers).float().cpu(), torch.from_numpy(radii).float().cpu()
    centers = np.array([shapely.Point(p) for p in centers])
    start_time_mesh = time.perf_counter()
    mesh_flags = np.array([detect_circles_polygon_collision(center, radius, polygon_t, bb) for center, radius in zip(centers, radii)])
    total_time_mesh += time.perf_counter() - start_time_mesh
    start_time_mlp = time.perf_counter()
    mlp_flags = torch.tensor([check_intersection_mlp(center_torch, center, radius_torch, radius, c_net, bb) for center_torch, center, radius_torch, radius in zip(centers_torch, centers, radii_torch, radii)]).float()
    total_time_mlp += time.perf_counter() - start_time_mlp
    wrong_check_count = (mesh_flags != mlp_flags.detach().cpu().numpy()).sum()

    return total_time_mesh / num_trials, total_time_mlp / num_trials, wrong_check_count

def main():
    global c_net
    c_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    # c_net = StarSDF()
    c_polygon_t = carve(c_net, deep=True, smoothify=False, return_merged=True)
    bounding_box = shapely.envelope(c_polygon_t)
    c_net.cpu()
    avg_time_mesh, avg_time_mlp, wrong_results = measure_intersection_time(c_polygon_t, bounding_box)
    print(f'Average intersection check time: {avg_time_mesh:.6f} seconds')

    # avg_time_mlp = measure_intersection_time_mlp(num_trials=1000)
    print(f'Average intersection check time with MLP: {avg_time_mlp:.6f} seconds')
    print(f'Number of incorrect checks: {wrong_results}')

if __name__ == "__main__":
    main()
