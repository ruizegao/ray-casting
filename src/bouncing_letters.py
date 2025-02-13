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

global I_COMP, C_COMP, V_COMP

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
    print("Carving")
    lower = torch.tensor([-0.55, -0.55])
    upper = torch.tensor([0.55, 0.55])
    func = crown.CrownImplicitFunction(mlp.func_from_spec(mode='default'), net, crown_mode='crown', input_dim=2)
    if deep:
        # print("deep")
        lowers, uppers, lAs, lbs, uAs, ubs, pos_lowers, pos_uppers, neg_lowers, neg_uppers = kd_tree.construct_hybrid_unknown_tree(
            func, net, lower, upper, base_depth=9, max_depth=12, node_dim=2, include_pos_neg=True)
    else:
        # print("not deep")
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
    # Include the negative nodes
    for n_l, n_u in zip(neg_lowers, neg_uppers):
        box_inside = shapely.geometry.Polygon([n_l, (n_l[0], n_u[1]), n_u, (n_u[0], n_l[1])])
        convex_poly_list.append(box_inside.buffer(0))
    # merged_inner_nodes = shapely.ops.unary_union(convex_poly_list)
    # if merged_inner_nodes.geom_type == 'Polygon':
    #     print("Merged into one polygon")
    #     convex_poly_list = shapely.ops.triangulate(merged_inner_nodes)
    # elif merged_inner_nodes.geom_type == 'MultiPolygon':
    #     convex_poly_list = []
    #     print("Merged into multiple polygons")
    #     for poly in merged_inner_nodes.geoms:
    #         convex_poly_list.extend(shapely.ops.triangulate(poly))
    # else:
    #     print("Keeping the original polygons")
    # print("After re-arrangement, # of polygons: {}".format(len(convex_poly_list)))

    # Include the negative portions of unknown nodes
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
                    convex_poly_list.append(g.buffer(0))
                    outer_polygons.append(g.buffer(0))

    merged = shapely.ops.unary_union(convex_poly_list)

    # Extract the largest connected component
    if merged.geom_type == 'MultiPolygon':
        largest_component = max(merged.geoms, key=lambda p: p.area)
    else:
        largest_component = merged  # If only one component exists, return all

    convex_poly_list = [poly for poly in convex_poly_list if poly.intersects(largest_component)]
    outer_polygons = [poly for poly in outer_polygons if poly.intersects(largest_component)]
    # print("existing # of polygons", len(convex_poly_list))
    # print("existing # of outer polygons", len(outer_polygons))
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
            convex_poly_list.append(added_poly.buffer(0))
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
            convex_poly_list.append(added_poly.buffer(0))
    # print("Updated # of polygons", len(convex_poly_list))

    return convex_poly_list

def scale_polygon(vertices, scale_factor):
    """ Scale a polygon's vertices by a given factor. """
    return [(x * scale_factor, y * scale_factor) for x, y in vertices]

"""This example spawns (bouncing) balls randomly on a L-shape constructed of 
two segment shapes. Not interactive.
"""

__docformat__ = "reStructuredText"

# Python imports
import random
from typing import List

# Library imports
import pygame

# pymunk imports
import pymunk
import pymunk.pygame_util


class BouncyBalls(object):
    """
    This class implements a simple scene in which there is a static platform (made up of a couple of lines)
    that don't move. Balls appear occasionally and drop onto the platform. They bounce around.
    """

    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # Letters that exist in the world
        self._letters: List[List[pymunk.Poly]] = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 30

    def run(self) -> None:
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            self._process_events()
            self._update_letters()
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies.
        :return: None
        """
        static_body = self._space.static_body
        static_lines = [
            pymunk.Segment(static_body, (100, 400), (300, 500), 0.0),
            pymunk.Segment(static_body, (300, 500), (500, 400), 0.0),
        ]
        for line in static_lines:
            line.elasticity = 0.9
            line.friction = 0.5
        self._space.add(*static_lines)

    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")

    def _update_letters(self) -> None:
        """
        Create/remove balls as necessary. Call once per frame only.
        :return: None
        """
        self._ticks_to_next_ball -= 1
        if self._ticks_to_next_ball <= 0:
            self._create_letter()
            self._ticks_to_next_ball = 100
        # Remove balls that fall below 100 vertically
        balls_to_remove = [ball for ball in self._letters if any([component.body.position.y > 600 for component in ball])]
        for ball in balls_to_remove:
            # for component in ball:
            self._space.remove(*ball, ball[0].body)
            self._letters.remove(ball)

    def _create_letter(self) -> None:
        """
        Create a letter.
        :return:
        """
        mass = 50

        # Convert to pymunk-friendly format
        complex_polygon = random.choice([I_COMP, C_COMP, C_COMP, V_COMP])
        scaled_polygons = [scale_polygon(polygon.exterior.coords, 50) for polygon in complex_polygon]

        # Create body with appropriate moment of inertia
        inertia = sum(pymunk.moment_for_poly(mass / len(scaled_polygons), vertices) for vertices in scaled_polygons)
        body = pymunk.Body(mass, inertia)
        x = random.randint(100, 500)
        body.position = x, 50

        # Create and add each convex polygon shape to the body
        shapes = []
        for vertices in scaled_polygons:
            shape = pymunk.Poly(body, vertices)
            shape.elasticity = 0.8
            shape.friction = 0.5
            shapes.append(shape)

        self._space.add(body, *shapes)
        self._letters.append(shapes)

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)


def main():
    c_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    c_net = c_net.to(device=set_t['device'])
    c_components = carve(c_net, deep=False)
    global C_COMP
    C_COMP = [shapely.geometry.Polygon(vertices) for vertices in c_components]

    v_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/V_MLP.pth', 'mlp')
    v_net = v_net.to(device=set_t['device'])
    v_components = carve(v_net, deep=False)
    global V_COMP
    V_COMP = [shapely.geometry.Polygon(vertices) for vertices in v_components]

    i_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/I_MLP.pth', 'mlp')
    i_net = i_net.to(device=set_t['device'])
    i_components = carve(i_net, deep=False)
    global I_COMP
    I_COMP = [shapely.geometry.Polygon(vertices) for vertices in i_components]

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("input_file", type=str,
    #                     help="The path to the .pth model from the root directory.")
    # parser.add_argument("--model_type", type=str, required=True,
    #                     help="Must specify if the model is one of the following: [mlp, siren].")
    # parser.add_argument("--x_L", type=float, nargs='+', default=[-1., -1.],
    #                     help="Bottom left point of the input bounding box.")
    # parser.add_argument("--x_U", type=float, nargs='+', default=[1., 1.],
    #                     help="Upper right point of the input bounding box.")
    # parser.add_argument("--crown_mode", type=str, default='CROWN',
    #                     help="Bounding method to use on the neural SDF.")
    # parser.add_argument("--deep", default=False, action='store_true')
    # # Parse arguments
    # args = parser.parse_args()
    # net = load_net_object(args.input_file, args.model_type)
    # net = net.to(device=set_t['device'])
    # concex_polygons = carve(net, deep=args.deep)

    # poly_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    # poly_body.position = 600, 400
    # # poly = pymunk.Poly.create_box(poly_body, (200, 100), 10)
    # # poly.sensor = True
    # convex_polygons = [
    #     list(polygon.exterior.coords) for polygon in concex_polygons
    # ]
    #
    # scaled_polygons = [scale_polygon(convex_polygon, 100) for convex_polygon in convex_polygons]
    #
    # # Create and add each convex polygon to the space
    # shapes = []
    # for vertices in scaled_polygons:
    #     poly = pymunk.Poly(poly_body, vertices)
    #     poly.sensor = True  # Keeps it as a sensor if needed
    #     shapes.append(poly)
    #
    # # Add the body and all shapes at the same time
    # space.add(poly_body, *shapes)

    game = BouncyBalls()
    game.run()


if __name__ == "__main__":
    main()
