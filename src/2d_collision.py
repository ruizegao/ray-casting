import time
import shapely
import shapely.set_operations
import random
from neural_utils import load_net_object
from bouncing_letters import carve, scale_polygon
import torch
import torch.nn as nn
import numpy as np

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
    avg_time_mesh, avg_time_mlp, wrong_results = measure_intersection_time(c_polygon_t, i_polygon_t)
    print(f'Average intersection check time: {avg_time_mesh:.6f} seconds')

    # avg_time_mlp = measure_intersection_time_mlp(num_trials=1000)
    print(f'Average intersection check time with MLP: {avg_time_mlp:.6f} seconds')
    print(f'Number of incorrect checks: {wrong_results}')

if __name__ == "__main__":
    main()
