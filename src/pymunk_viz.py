"""Very simple example that does not depend on any third party library such 
as pygame or pyglet like the other examples. 
"""

import random
import sys
import math

import pygame
import pymunk
import pymunk.util
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import argparse
from neural_sdf import MLP, Siren
from neural_utils import load_net_object
import crown
import mlp
import kd_tree
from shapely.ops import split, unary_union
import shapely
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import copy


try:
    plt.style.use("seaborn-white")
except OSError as e:
    plt.style.use("seaborn-v0_8-white")

set_t = {
    'dtype': torch.float32,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

to_numpy = lambda x : x.detach().cpu().numpy()

def project_line_onto_square(a1, a2, b, x1_min, x1_max, x2_min, x2_max):
    # Define the bounding box (square)
    square = shapely.geometry.box(x1_min, x2_min, x1_max, x2_max)

    # Define the line equation a1*x1 + a2*x2 + b = 0 in explicit form
    if a2 != 0:
        # Express x2 as a function of x1
        line = shapely.geometry.LineString([
            (x1_min, (-a1*x1_min - b) / a2),
            (x1_max, (-a1*x1_max - b) / a2)
        ])
    else:
        # Vertical line case: x1 = constant
        x1 = -b / a1
        line = shapely.geometry.LineString([(x1, x2_min), (x1, x2_max)])

    # Intersect the line with the square
    segment = line.intersection(square)

    return segment


def carve(net: MLP, deep=False):
    lower = torch.tensor([-0.55, -0.55])
    upper = torch.tensor([0.55, 0.55])
    func = crown.CrownImplicitFunction(mlp.func_from_spec(mode='default'), net, crown_mode='crown', input_dim=2)
    if deep:
        lowers, uppers, lAs, lbs, uAs, ubs, pos_lowers, pos_uppers, neg_lowers, neg_uppers = kd_tree.construct_hybrid_unknown_tree(
            func, net, lower, upper, base_depth=12, max_depth=15, node_dim=2, include_pos_neg=True)
    else:
        lowers, uppers, lAs, lbs, uAs, ubs, pos_lowers, pos_uppers, neg_lowers, neg_uppers = kd_tree.construct_hybrid_unknown_tree(
            func, net, lower, upper, base_depth=6, max_depth=9, node_dim=2, include_pos_neg=True)
    lowers = lowers.detach().cpu().numpy()
    uppers = uppers.detach().cpu().numpy()
    lAs = lAs.detach().cpu().numpy()
    lbs = lbs.detach().cpu().numpy()
    uAs = uAs.detach().cpu().numpy()
    ubs = ubs.detach().cpu().numpy()
    pos_lowers = pos_lowers.detach().cpu().numpy()
    pos_uppers = pos_uppers.detach().cpu().numpy()
    neg_lowers = neg_lowers.detach().cpu().numpy()
    neg_uppers = neg_uppers.detach().cpu().numpy()

    squares = []
    outer_segments = []
    outer_segments_lAs = []
    outer_segments_lbs = []
    inner_segments = []
    outer_polygons = []
    inner_polygons = []
    inner_segments_uAs = []
    inner_segments_ubs = []

    convex_poly_list = []
    for n_l, n_u in zip(neg_lowers, neg_uppers):
        box_inside = shapely.geometry.Polygon([n_l, (n_l[0], n_u[1]), n_u, (n_u[0], n_l[1])])
        convex_poly_list.append(box_inside)

    for l, u, lA, lb, uA, ub in zip(lowers, uppers, lAs, lbs, uAs, ubs):
        square = shapely.geometry.Polygon([l, (l[0], u[1]), u, (u[0], l[1])])
        squares.append(square)
        outer_line = project_line_onto_square(lA[0], lA[1], lb, -0.55, 0.55, -0.55, 0.55)
        inner_line = project_line_onto_square(uA[0], uA[1], ub, -0.55, 0.55, -0.55, 0.55)


        # For each node and its outer_line segment, get its neighbors that the outer_line segment also intersect with
        outer_segment = shapely.intersection(square, outer_line)
        inner_segment = shapely.intersection(square, inner_line)
        if len(np.array(outer_segment.coords)) == 2:
            outer_segments.append(outer_segment)
            outer_segments_lAs.append(lA)
            outer_segments_lbs.append(lb)
        if len(np.array(inner_segment.coords)) == 2:
            inner_segments.append(inner_segment)
            inner_segments_uAs.append(uA)
            inner_segments_ubs.append(ub)

        slices1 = split(square, outer_line)

        for g in slices1.geoms:
            if g.geom_type == 'Polygon':
                c = shapely.centroid(g)
                c = np.array([c.x, c.y])
                cls = np.dot(lA, c) + lb
                if cls < 0.:
                    convex_poly_list.append(g)
                    outer_polygons.append(g)

    outer_qualified_neighbors = []
    outer_contact_points = []

    for outer_segment in outer_segments:
        neighbors_buffer = []
        points_buffer = []
        for outer_polygon, lA, lb in zip(outer_polygons, lAs, lbs):
            segment_polygon_intersection = shapely.intersection(outer_polygon, outer_segment)
            if segment_polygon_intersection.geom_type == 'Point':
                p = np.array([segment_polygon_intersection.x, segment_polygon_intersection.y])
                cls = np.dot(lA, p) + lb
                if cls <= 0:
                    neighbors_buffer.append(outer_polygon)
                    points_buffer.append(segment_polygon_intersection)
        outer_qualified_neighbors.append(neighbors_buffer)
        outer_contact_points.append(points_buffer)

    for outer_segment, neighbors_buffer, points_buffer, lA, lb in zip(outer_segments, outer_qualified_neighbors,
                                                                      outer_contact_points, outer_segments_lAs,
                                                                      outer_segments_lbs):
        if len(neighbors_buffer) == 2:
            poly_A = neighbors_buffer[0]
            poly_B = neighbors_buffer[1]
            point_A = points_buffer[0]
            point_B = points_buffer[1]
            vertices_A = list(poly_A.exterior.coords)
            vertices_B = list(poly_B.exterior.coords)
            for v_A in vertices_A:
                if point_A.x == v_A[0] or point_A.y == v_A[1]:
                    if np.dot(lA, v_A) + lb > 0.:
                        point_A_new = shapely.geometry.Point(v_A)
            for v_B in vertices_B:
                if point_B.x == v_B[0] or point_B.y == v_B[1]:
                    if np.dot(lA, v_B) + lb > 0.:
                        point_B_new = shapely.geometry.Point(v_B)

            added_poly = shapely.geometry.Polygon(
                ((point_A.x, point_A.y), (point_B.x, point_B.y),
                 (point_B_new.x, point_B_new.y), (point_A_new.x, point_A_new.y))
            )
            convex_poly_list.append(added_poly)
        elif len(neighbors_buffer) == 1:
            poly_A = neighbors_buffer[0]
            point_A = points_buffer[0]
            vertices_A = list(poly_A.exterior.coords)
            for v_A in vertices_A:
                if point_A.x == v_A[0] or point_A.y == v_A[1]:
                    if np.dot(lA, v_A) + lb > 0.:
                        point_A_new = shapely.geometry.Point(v_A)
            unchanged_point = outer_segment.boundary.geoms[0] if shapely.equals(point_A,
                                                                                outer_segment.boundary.geoms[1]) else \
            outer_segment.boundary.geoms[1]
            added_poly = shapely.geometry.Polygon(
                ((unchanged_point.x, unchanged_point.y), (point_A.x, point_A.y), (point_A_new.x, point_A_new.y))
            )
            convex_poly_list.append(added_poly)

    return convex_poly_list

def begin(arbiter, space, data):
    data["log"] = {
        "begin": 1,
        "pre_solve": 0,
        "post_solve": "N/A (no post_solve for sensors)",
        "separate": 0,
    }

    return True


def pre_solve(arbiter: pymunk.Arbiter, space, data):
    data["log"]["pre_solve"] += 1

    screen = data["screen"]

    screen.blit(
        data["font"].render(
            "collision normal",
            True,
            pygame.Color("black"),
        ),
        (5, 500),
    )
    n = arbiter.normal * 30
    pygame.draw.aaline(screen, pygame.Color("red"), (50, 550), (50 + n.x, 550 + n.y))

    cps: pymunk.ContactPointSet = arbiter.contact_point_set
    for p in cps.points:
        pygame.draw.circle(screen, pygame.Color("darkblue"), p.point_a, 5, 1)
        pygame.draw.circle(screen, pygame.Color("darkred"), p.point_b, 5, 1)
        pygame.draw.aaline(screen, pygame.Color("yellow"), p.point_a, p.point_b)

        screen.blit(
            data["font"].render(
                f"distance {p.distance:.2f}",
                True,
                pygame.Color("black"),
            ),
            (p.point_a.interpolate_to(p.point_b, 0.5)),
        )

    return True


def post_solve(arbiter, space, data):
    # Will not be called, since the shapes are kinematic sensors
    pass


def separate(arbiter, space, data):
    data["log"]["separate"] += 1
    pass

def scale_polygon(vertices, scale_factor):
    """ Scale a polygon's vertices by a given factor. """
    return [(x * scale_factor, y * scale_factor) for x, y in vertices]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str,
                        help="The path to the .pth model from the root directory.")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Must specify if the model is one of the following: [mlp, siren].")
    parser.add_argument("--x_L", type=float, nargs='+', default=[-1., -1.],
                        help="Bottom left point of the input bounding box.")
    parser.add_argument("--x_U", type=float, nargs='+', default=[1., 1.],
                        help="Upper right point of the input bounding box.")
    parser.add_argument("--crown_mode", type=str, default='CROWN',
                        help="Bounding method to use on the neural SDF.")
    parser.add_argument("--deep", default=False, action='store_true')
    # Parse arguments
    args = parser.parse_args()
    net = load_net_object(args.input_file, args.model_type)
    net = net.to(device=set_t['device'])
    concex_polygons = carve(net, deep=args.deep)
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 20)

    space = pymunk.Space()
    # space.gravity = Vec2d(0.0, 900.0)
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # segment_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    # segment_body.position = 600, 200
    # segment = pymunk.Segment(segment_body, Vec2d(-100, 0), Vec2d(100, 0), 5)
    # segment.sensor = True
    # space.add(segment_body, segment)

    # circle_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    # circle_body.position = 200, 300
    # circle = pymunk.Circle(circle_body, 50)
    # circle.sensor = True
    # space.add(circle_body, circle)

    poly_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    poly_body.position = 600, 400
    # poly = pymunk.Poly.create_box(poly_body, (200, 100), 10)
    # poly.sensor = True
    convex_polygons = [
        list(polygon.exterior.coords) for polygon in concex_polygons
    ]

    scaled_polygons = [scale_polygon(convex_polygon, 100) for convex_polygon in convex_polygons]

    # Create and add each convex polygon to the space
    shapes = []
    for vertices in scaled_polygons:
        poly = pymunk.Poly(poly_body, vertices)
        poly.sensor = True  # Keeps it as a sensor if needed
        shapes.append(poly)

    # Add the body and all shapes at the same time
    space.add(poly_body, *shapes)

    # space.add(poly_body, poly)


    mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    space.add(mouse_body)

    circle = pymunk.Circle(mouse_body, 60)
    circle.sensor = True
    circle.collision_type = 1
    segment = pymunk.Segment(mouse_body, Vec2d(-1, 0), Vec2d(1, 0), 20)
    segment.sensor = True
    segment.collision_type = 1
    poly = pymunk.Poly.create_box(mouse_body, (20, 10), 10)
    poly.sensor = True
    poly.collision_type = 1

    shapes = [circle, segment, poly]
    selected_shape_idx = 0
    space.add(shapes[selected_shape_idx])

    h = space.add_collision_handler(0, 1)
    h.data["screen"] = screen
    h.data["log"] = {"begin": 0, "pre_solve": 0, "post_solve": 0, "separate": 0}
    h.data["font"] = font
    h.begin = begin
    h.pre_solve = pre_solve
    h.post_solve = post_solve
    h.separate = separate

    while True:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(screen, "collisions.png")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                space.remove(shapes[selected_shape_idx])
                selected_shape_idx = (selected_shape_idx + 1) % len(shapes)
                space.add(shapes[selected_shape_idx])
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                mouse_body.angle += math.pi / 8

        p = pygame.mouse.get_pos()
        mouse_body.position = p[0], p[1]

        screen.fill(pygame.Color("white"))
        space.debug_draw(draw_options)

        screen.blit(
            font.render(
                "Left click to switch shape type, right click to rotate. (The shape follows the mouse)",
                True,
                pygame.Color("black"),
            ),
            (5, 5),
        )

        y = 30
        for k in h.data["log"]:
            screen.blit(
                font.render(
                    f"{k}: {h.data['log'][k]}",
                    True,
                    pygame.Color("black"),
                ),
                (5, y),
            )
            y += 20

        space.step(1.0 / 60.0)

        pygame.display.flip()
        clock.tick(50)


if __name__ == "__main__":
    sys.exit(main())