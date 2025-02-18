import pymunk
import random
import time
import shapely
from neural_utils import load_net_object
from bouncing_letters import carve, scale_polygon
import torch
import torch.nn as nn
import pygame
import pymunk.pygame_util

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


def create_static_polygons(space):
    poly_body_l = pymunk.Body(body_type=pymunk.Body.STATIC)
    shapes_l = []
    for poly in C_COMP_L:
        poly = pymunk.Poly(poly_body_l, list(poly.exterior.coords))
        # poly.sensor = True  # Keeps it as a sensor if needed
        shapes_l.append(poly)

    space.add(poly_body_l, *shapes_l)


    poly_body_t = pymunk.Body(body_type=pymunk.Body.STATIC)

    shapes_t = []

    for poly in C_COMP_T:
        poly = pymunk.Poly(poly_body_t, list(poly.exterior.coords))
        # poly.sensor = True  # Keeps it as a sensor if needed
        shapes_t.append(poly)

    space.add(poly_body_t, *shapes_t)
    # visualize(space)

    return shapes_l, shapes_t

def generate_random_circle(space, radius=0.05):
    x, y = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = (x, y)
    shape = pymunk.Circle(body, radius)
    space.add(body, shape)
    return shape, (x, y)

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

def check_intersection_space(space, shape):
    for s in space.shapes:
        if s != shape:
            collision_points = shape.shapes_collide(s).points
            if collision_points:
                return True
    return False

def check_intersection_mlp(net, circle_coords, circle_radius=0.05):
    distance = net(torch.tensor(circle_coords)).item()
    # print(circle_coords, distance)
    return distance <= circle_radius

def measure_intersection_time(space, shapes_l, shapes_t, num_trials=10000):
    total_time_mesh = 0
    total_time_mlp = 0
    wrong_check_count = 0
    for i in range(num_trials):
        shape, circle_coords = generate_random_circle(space)
        # circle_coords = generate_random_circle_coords()
        start_time_mesh = time.perf_counter()
        # mesh_flag = check_intersection_space(space, shape)
        mesh_flag = check_intersection_robust(shape, shapes_l, shapes_t, c_net)
        total_time_mesh += time.perf_counter() - start_time_mesh
        start_time_mlp = time.perf_counter()
        mlp_flag = check_intersection_mlp(c_net, circle_coords=circle_coords)
        total_time_mlp += time.perf_counter() - start_time_mlp
        if mesh_flag != mlp_flag:
            # print(mesh_flag, mlp_flag)
            wrong_check_count += 1
        # print(mesh_flag, mlp_flag)
        # if i != num_trials - 1:
        # visualize(space)
        space.remove(shape.body, shape)

    return total_time_mesh / num_trials, total_time_mlp / num_trials, wrong_check_count

def visualize(space):
    pygame.init()
    screen = pygame.display.set_mode((100, 100))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

def main():
    global c_net
    # c_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    c_net = StarSDF()
    c_net = c_net.to(device=set_t['device'])
    c_components = carve(c_net, deep=False)
    global C_COMP_L
    C_COMP_L = [shapely.geometry.Polygon(vertices) for vertices in c_components]
    c_components = carve(c_net, deep=True)
    global C_COMP_T
    C_COMP_T = [shapely.geometry.Polygon(vertices) for vertices in c_components]

    space = pymunk.Space()
    shapes_l, shapes_t = create_static_polygons(space)
    # print(len(shapes_l), len(shapes_t))
    avg_time_mesh, avg_time_mlp, wrong_results = measure_intersection_time(space, shapes_l, shapes_t)
    print(f'Average intersection check time: {avg_time_mesh:.6f} seconds')
    # visualize(space)

    # avg_time_mlp = measure_intersection_time_mlp(num_trials=1000)
    print(f'Average intersection check time with MLP: {avg_time_mlp:.6f} seconds')
    print(f'Number of incorrect checks: {wrong_results}')

if __name__ == "__main__":
    main()
