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


global c_net, C_COMP_L, C_COMP_T

set_t = {
    'dtype': torch.float32,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}


class StarSDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1))  # Inner radius
        self.b = nn.Parameter(torch.tensor(0.2))  # Outer radius
        self.freq = nn.Parameter(torch.tensor(5.0))  # Number of star spikes

    def forward(self, x):
        """
        x: (batch_size, 2) tensor with (x, y) coordinates
        Returns: (batch_size,) tensor with signed distances
        """
        r = torch.sqrt(x.pow(2).sum(dim=-1))  # Compute r = sqrt(x^2 + y^2)

        # Approximate cos(freq * theta) without atan2
        s = torch.sin(self.freq * torch.abs(x[:, 1]) / (torch.abs(x[:, 0]) + 1e-8))  # Avoid div by zero

        # Star SDF using the estimated shape function
        sdf = r - (self.a + self.b * s)
        # sdf = sdf.unsqueeze(-1)
        # print(sdf.shape)
        return sdf

def generate_circles(N=10000):
    centers = np.random.uniform(-0.5, 0.5, (N, 2))
    radii = np.random.uniform(0.04, 0.05, (N,))
    return centers, radii

def detect_circles_polygon_collision(centers, radii, polygon):
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

def check_intersection_mlp(centers, radii, net):
    distance = net(centers).squeeze()
    return distance <= radii

def measure_intersection_time(polygon_t, num_trials=1):
    total_time_mesh = 0
    total_time_mlp = 0
    centers, radii = generate_circles(N=num_trials)
    centers_torch, radii_torch = torch.from_numpy(centers).float().cuda(), torch.from_numpy(radii).float().cuda()
    centers = np.array([shapely.Point(p) for p in centers])
    start_time_mesh = time.perf_counter()
    mesh_flags = detect_circles_polygon_collision(centers, radii, polygon_t)
    total_time_mesh += time.perf_counter() - start_time_mesh
    start_time_mlp = time.perf_counter()
    mlp_flags = check_intersection_mlp(centers_torch, radii_torch, c_net)
    total_time_mlp += time.perf_counter() - start_time_mlp
    print(mesh_flags.shape, mlp_flags.shape)
    wrong_check_count = (mesh_flags != mlp_flags.detach().cpu().numpy()).sum()

    return total_time_mesh, total_time_mlp, wrong_check_count

def main():
    global c_net
    c_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    # c_net = StarSDF()
    c_net = c_net.to(device=set_t['device'])
    c_polygon_l = carve(c_net, deep=False, smoothify=False, return_merged=True)
    # global C_COMP_L
    # C_COMP_L = [shapely.geometry.Polygon(vertices) for vertices in c_components]
    # c_polygon_l = shapely.set_operations.union_all(C_COMP_L)
    c_polygon_t = carve(c_net, deep=True, smoothify=False, return_merged=True)
    # global C_COMP_T
    # C_COMP_T = [shapely.geometry.Polygon(vertices) for vertices in c_components]
    # c_polygon_t = shapely.set_operations.union_all(C_COMP_T)

    avg_time_mesh, avg_time_mlp, wrong_results = measure_intersection_time(c_polygon_t)
    print(f'Average intersection check time: {avg_time_mesh:.6f} seconds')

    # avg_time_mlp = measure_intersection_time_mlp(num_trials=1000)
    print(f'Average intersection check time with MLP: {avg_time_mlp:.6f} seconds')
    print(f'Number of incorrect checks: {wrong_results}')

if __name__ == "__main__":
    main()
