import pymunk
import random
import time
import shapely
from neural_utils import load_net_object
from bouncing_letters import carve, scale_polygon
import torch
import pygame
import pymunk.pygame_util

global i_net, c_net, v_net, C_COMP_L, C_COMP_L, V_COMP_L, C_COMP_T, C_COMP_T, V_COMP_T

set_t = {
    'dtype': torch.float32,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

def shift_polygon(vertices, d_x=0, d_y=0):
    return [(x + d_x, y + d_y) for x, y in vertices]

def vertically_flip_polygon(vertices, flip_axis=0):
    return [(x, 2 * flip_axis - y) for x, y in vertices]

def create_static_polygons(space):
    # scaled_i = [scale_polygon(poly.exterior.coords, 80) for poly in I_COMP_L]
    # scaled_c = [scale_polygon(poly.exterior.coords, 80) for poly in C_COMP_L]
    # scaled_v = [scale_polygon(poly.exterior.coords, 80) for poly in V_COMP_L]
    # flipped_i = [vertically_flip_polygon(poly, 0) for poly in scaled_i]
    # flipped_c = [vertically_flip_polygon(poly, 0) for poly in scaled_c]
    # flipped_v = [vertically_flip_polygon(poly, 0) for poly in scaled_v]
    # shifted_i = [shift_polygon(poly, d_x=50, d_y=50) for poly in flipped_i]
    # shifted_c1 = [shift_polygon(poly, d_x=150, d_y=50) for poly in flipped_c]
    # shifted_c2 = [shift_polygon(poly, d_x=250, d_y=50) for poly in flipped_c]
    # shifted_v = [shift_polygon(poly, d_x=350, d_y=50) for poly in flipped_v]
    #
    # poly_body_l = pymunk.Body(body_type=pymunk.Body.STATIC)
    #
    shapes_l = []
    #
    # for vertices in shifted_i:
    #     poly = pymunk.Poly(poly_body_l, vertices)
    #     poly.sensor = True  # Keeps it as a sensor if needed
    #     shapes_l.append(poly)
    #
    # for vertices in shifted_c1:
    #     poly = pymunk.Poly(poly_body_l, vertices)
    #     poly.sensor = True  # Keeps it as a sensor if needed
    #     shapes_l.append(poly)
    #
    # for vertices in shifted_c2:
    #     poly = pymunk.Poly(poly_body_l, vertices)
    #     poly.sensor = True  # Keeps it as a sensor if needed
    #     shapes_l.append(poly)
    #
    # for vertices in shifted_v:
    #     poly = pymunk.Poly(poly_body_l, vertices)
    #     poly.sensor = True  # Keeps it as a sensor if needed
    #     shapes_l.append(poly)
    #
    # space.add(poly_body_l, *shapes_l)

    scaled_i = [scale_polygon(poly.exterior.coords, 80) for poly in C_COMP_T]
    scaled_c = [scale_polygon(poly.exterior.coords, 80) for poly in C_COMP_T]
    scaled_v = [scale_polygon(poly.exterior.coords, 80) for poly in V_COMP_T]
    flipped_i = [vertically_flip_polygon(poly, 0) for poly in scaled_i]
    flipped_c = [vertically_flip_polygon(poly, 0) for poly in scaled_c]
    flipped_v = [vertically_flip_polygon(poly, 0) for poly in scaled_v]
    shifted_i = [shift_polygon(poly, d_x=50, d_y=50) for poly in flipped_i]
    shifted_c1 = [shift_polygon(poly, d_x=150, d_y=50) for poly in flipped_c]
    shifted_c2 = [shift_polygon(poly, d_x=250, d_y=50) for poly in flipped_c]
    shifted_v = [shift_polygon(poly, d_x=350, d_y=50) for poly in flipped_v]
    visualize(space)
    poly_body_t = pymunk.Body(body_type=pymunk.Body.STATIC)

    shapes_t = []

    for vertices in shifted_i:
        poly = pymunk.Poly(poly_body_t, vertices)
        poly.sensor = True  # Keeps it as a sensor if needed
        shapes_t.append(poly)

    for vertices in shifted_c1:
        poly = pymunk.Poly(poly_body_t, vertices)
        poly.sensor = True  # Keeps it as a sensor if needed
        shapes_t.append(poly)

    for vertices in shifted_c2:
        poly = pymunk.Poly(poly_body_t, vertices)
        poly.sensor = True  # Keeps it as a sensor if needed
        shapes_t.append(poly)

    for vertices in shifted_v:
        poly = pymunk.Poly(poly_body_t, vertices)
        poly.sensor = True  # Keeps it as a sensor if needed
        shapes_t.append(poly)

    space.add(poly_body_t, *shapes_t)
    visualize(space)

    return shapes_l, shapes_t


def generate_random_circle(space, radius=10):
    x, y = random.uniform(0, 400), random.uniform(0, 100)
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = (x, y)
    shape = pymunk.Circle(body, radius)
    space.add(body, shape)
    return shape, (x, y)

def check_intersection_space(space, shape):
    for s in space.shapes:
        if s != shape and shape.shapes_collide(s).points:
            return True
    return False

def check_intersection(shape, shapes):
    for s in shapes:
        if shape.shapes_collide(s).points:
            return True
    return False

def check_intersection_robust(shape, shapes_l, shapes_t):
    for s in shapes_l:
        if shape.shapes_collide(s).points:
            if check_intersection(s, shapes_t):
                return True
    return False

def check_intersection_coords(space, point):
    for s in space.shapes:
        if s.point_query(point).distance <= 10:
            return True
    return False

def measure_intersection_time(space, shapes_l, shapes_t, num_trials=1000):
    total_time_mesh = 0
    total_time_mlp = 0
    wrong_check_count = 0
    for i in range(num_trials):
        shape, circle_coords = generate_random_circle(space)
        # circle_coords = generate_random_circle_coords()
        start_time_mesh = time.perf_counter()
        # mesh_flag = check_intersection_robust(shape, shapes_l, shapes_t)
        # mesh_flag = check_intersection(shape, shapes_t)
        mesh_flag = check_intersection_space(space, shape)
        # check_intersection_coords(space, circle_coords)
        total_time_mesh += time.perf_counter() - start_time_mesh
        start_time_mlp = time.perf_counter()
        mlp_flag = check_intersection_mlps([i_net, c_net, c_net, v_net], circle_coords=circle_coords,
                                           d_xs=(50, 150, 250, 350), d_ys=(50, 50, 50, 50))
        total_time_mlp += time.perf_counter() - start_time_mlp
        if mesh_flag != mlp_flag:
            print(mesh_flag, mlp_flag)
            wrong_check_count += 1
        if i != num_trials - 1:
            space.remove(shape.body, shape)

    return total_time_mesh / num_trials, total_time_mlp / num_trials, wrong_check_count

def visualize(space):
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
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

def generate_random_circle_coords(radius=10):
    x, y = random.uniform(0, 400), random.uniform(0, 100)
    return x, y

def check_intersection_mlp(net, circle_coords, circle_radius=10, d_x=0, d_y=0):
    x, y = circle_coords
    x, y = (x - d_x) / 80., - (y - d_y) / 80.
    distance = net(torch.tensor([x, y])).item() * 80
    # print(distance)
    return distance <= circle_radius

def check_intersection_mlps(nets, circle_coords, circle_radius=10, d_xs=(0,), d_ys=(0,)):
    for net, d_x, d_y in zip(nets, d_xs, d_ys):
        if check_intersection_mlp(net, circle_coords, circle_radius, d_x, d_y):
            return True
    return False

def measure_intersection_time_mlp(num_trials=1000):
    total_time = 0
    for i in range(num_trials):
        circle_coords = generate_random_circle_coords()
        start_time = time.perf_counter()
        intersection_flag = any([check_intersection_mlp(net=i_net, circle_coords=circle_coords, d_x=50, d_y=50),
                                 check_intersection_mlp(net=c_net, circle_coords=circle_coords, d_x=150, d_y=50),
                                 check_intersection_mlp(net=c_net, circle_coords=circle_coords, d_x=250, d_y=50),
                                 check_intersection_mlp(net=v_net, circle_coords=circle_coords, d_x=350, d_y=50)])
        total_time += time.perf_counter() - start_time

    return total_time / num_trials

# def intersection_evaluation

def main():
    global c_net
    c_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    c_net = c_net.to(device=set_t['device'])
    c_components = carve(c_net, deep=False)
    global C_COMP_L
    C_COMP_L = [shapely.geometry.Polygon(vertices) for vertices in c_components]
    c_components = carve(c_net, deep=True)
    global C_COMP_T
    C_COMP_T = [shapely.geometry.Polygon(vertices) for vertices in c_components]

    global v_net
    v_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/V_MLP.pth', 'mlp')
    v_net = v_net.to(device=set_t['device'])
    v_components = carve(v_net, deep=False)
    global V_COMP_L
    V_COMP_L = [shapely.geometry.Polygon(vertices) for vertices in v_components]
    v_components = carve(v_net, deep=True)
    global V_COMP_T
    V_COMP_T = [shapely.geometry.Polygon(vertices) for vertices in v_components]

    global i_net
    i_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/I_MLP.pth', 'mlp')
    i_net = i_net.to(device=set_t['device'])
    i_components = carve(i_net, deep=False)
    global C_COMP_L
    I_COMP_L = [shapely.geometry.Polygon(vertices) for vertices in i_components]
    i_components = carve(i_net, deep=True)
    global C_COMP_T
    I_COMP_T = [shapely.geometry.Polygon(vertices) for vertices in i_components]

    space = pymunk.Space()
    shapes_l, shapes_t = create_static_polygons(space)
    print(len(shapes_l), len(shapes_t))
    avg_time_mesh, avg_time_mlp, wrong_results = measure_intersection_time(space, shapes_l, shapes_t)
    print(f'Average intersection check time: {avg_time_mesh:.6f} seconds')
    # visualize(space)

    # avg_time_mlp = measure_intersection_time_mlp(num_trials=1000)
    print(f'Average intersection check time with MLP: {avg_time_mlp:.6f} seconds')
    print(f'Number of incorrect checks: {wrong_results}')

if __name__ == "__main__":
    main()
