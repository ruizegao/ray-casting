import time
import shapely
import shapely.set_operations
import random

from triton.language import dtype

from neural_utils import load_net_object
from bouncing_letters import carve, scale_polygon
import torch
import torch.nn as nn
import numpy as np
import functorch
import time

import jax
import jax.numpy as jnp

@jax.jit
def detect_circle_polygon_collision(circle_centers: jnp.ndarray, circle_radius: jnp.ndarray,
                                          polygon_vertices: jnp.ndarray) -> jnp.ndarray:
    """
    Detects if a batch of circles collides with a polygon (either intersects or is contained).

    Args:
        circle_centers: Array of shape (B, 2) where B is the batch size, representing circle centers.
        circle_radius: Scalar or Array of shape (B,) representing the radii of the circles.
        polygon_vertices: Array of shape (N, 2) representing polygon vertices.

    Returns:
        collision_mask: Boolean array of shape (B,) indicating whether each circle in the batch collides with the polygon.
    """
    B = circle_centers.shape[0]  # Batch size
    N = polygon_vertices.shape[0]  # Number of vertices

    # Shift vertices to form edges
    edges_start = polygon_vertices
    edges_end = jnp.roll(polygon_vertices, shift=-1, axis=0)  # Shift vertices to form edges

    # Compute closest points on edges to the circle centers
    vw = edges_end - edges_start
    pv = circle_centers[:, None, :] - edges_start  # Shape: (B, N, 2)

    t = (pv * vw).sum(axis=-1) / (vw * vw).sum(axis=-1).clip(min=1e-6)
    t = jnp.clip(t, 0, 1)  # Clamp to segment

    closest_points = edges_start + t[:, :, None] * vw  # Closest points

    # Compute distances from closest points to the circle centers
    distances = jnp.linalg.norm(closest_points - circle_centers[:, None, :], axis=-1)

    # Check intersection: if any closest point distance < radius
    intersection = distances < circle_radius[:, None]

    # Check containment using ray-casting method
    v1, v2 = edges_start, edges_end

    condition1 = (v1[:, 1] > circle_centers[:, 1][:, None]) != (v2[:, 1] > circle_centers[:, 1][:, None])
    slope = (v2[:, 0] - v1[:, 0]) / (v2[:, 1] - v1[:, 1] + 1e-6)
    # x_intersect = v1[:, 0] + slope[:, None] * (circle_centers[:, 1][:, None] - v1[:, 1])
    x_intersect = v1[:, 0] + slope[None, :] * (circle_centers[:, 1][:, None] - v1[:, 1])

    # Ignore horizontal edges
    non_horizontal = v1[:, 1] != v2[:, 1]
    condition2 = circle_centers[:, 0][:, None] < x_intersect

    intersections = jnp.sum((condition1 & condition2) & non_horizontal, axis=1)  # Count ray crossings

    contained = intersections % 2 == 1  # Odd crossings and no intersection

    collision_mask = jnp.any(intersection, axis=1) | contained  # Union of intersection and containment

    return collision_mask

# Batch version of the function using jax.vmap
@jax.jit
def detect_circle_polygon_collision_batch(circle_centers: jnp.ndarray, circle_radius: jnp.ndarray, polygon_vertices: jnp.ndarray):
    return jax.vmap(detect_circle_polygon_collision, in_axes=(0, 0, 0))(circle_centers, circle_radius, polygon_vertices)

set_t = {
    'dtype': torch.float32,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

def generate_circles(N=10000):
    centers = np.random.uniform(-2, 2, (N, 2))
    radii = np.random.uniform(0.1, 0.8, (N,))
    return jnp.stack([jnp.array(centers)]*4), jnp.stack([jnp.array(radii)]*4)

def main():
    mlp_offsets = np.array([[-1.5, 0.], [-0.5, 0.], [0.5, 0.], [1.5, 0.]]).astype(np.float32)
    mlp_offsets = np.array([[-0.4, 0.], [-0.2, 0.], [0.2, 0.], [0.4, 0.]]).astype(np.float32)
    i_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/I_MLP.pth', 'mlp')
    i_polygon_t = carve(i_net, deep=True, smoothify=False, return_merged=True)
    i_polygon_t = shapely.affinity.translate(i_polygon_t, mlp_offsets[0][0], mlp_offsets[0][1])
    c_net_left = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    c_polygon_t_left = carve(c_net_left, deep=True, smoothify=False, return_merged=True)
    c_polygon_t_left = shapely.affinity.translate(c_polygon_t_left, mlp_offsets[1][0], mlp_offsets[1][1])
    c_net_right = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
    c_polygon_t_right = carve(c_net_right, deep=True, smoothify=False, return_merged=True)
    c_polygon_t_right = shapely.affinity.translate(c_polygon_t_right, mlp_offsets[2][0], mlp_offsets[2][1])
    v_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/V_MLP.pth', 'mlp')
    v_polygon_t = carve(v_net, deep=True, smoothify=False, return_merged=True)
    v_polygon_t = shapely.affinity.translate(v_polygon_t, mlp_offsets[3][0], mlp_offsets[3][1])

    # i_polygon_l = carve(i_net, deep=False, smoothify=True, return_merged=True)
    # i_polygon_l = shapely.affinity.translate(i_polygon_l, mlp_offsets[0][0], mlp_offsets[0][0])
    # c_polygon_l_left = carve(c_net_left, deep=False, smoothify=True, return_merged=True)
    # c_polygon_l_left = shapely.affinity.translate(c_polygon_l_left, mlp_offsets[1][0], mlp_offsets[1][1])
    # c_polygon_l_right = carve(c_net_right, deep=False, smoothify=True, return_merged=True)
    # c_polygon_l_right = shapely.affinity.translate(c_polygon_l_right, mlp_offsets[2][0], mlp_offsets[2][1])
    # v_polygon_l = carve(v_net, deep=False, smoothify=True, return_merged=True)
    # v_polygon_l = shapely.affinity.translate(v_polygon_l, mlp_offsets[3][0], mlp_offsets[3][1])

    shapely_polygons = [i_polygon_t, c_polygon_t_left, c_polygon_t_right, v_polygon_t]
    num_verts = jnp.array([len(shapely_polygon.exterior.coords) for shapely_polygon in shapely_polygons])
    max_num_verts = jnp.max(num_verts)
    polygons = [jnp.array(shapely_polygon.exterior.coords) for shapely_polygon in shapely_polygons]
    polygons = [jnp.concatenate((v, v[-1][jnp.newaxis, :].repeat(max_num_verts-num_v, 0))) for v, num_v in zip(polygons, num_verts)]
    polygons = jnp.stack(polygons)
    vert_masks = [jnp.hstack([jnp.ones(n_v, dtype=jnp.bool), jnp.zeros(max_num_verts - n_v, dtype=jnp.bool)]) for n_v in num_verts]
    vert_masks = jnp.stack(vert_masks)
    # print(vert_masks.shape)
    # print(vert_masks)
    polygons_bb = jnp.array([shapely.envelope(poly).exterior.coords for poly in shapely_polygons])
    # polygons_bb = jnp.array([i_polygon_l, c_polygon_l_left, c_polygon_l_right, v_polygon_l])
    i_net.cuda()
    c_net_left.cuda()
    c_net_right.cuda()
    v_net.cuda()
    mlps = torch.nn.ModuleList([i_net, c_net_left, c_net_right, v_net])
    test_circle_centers, test_circle_radii = generate_circles(10000)
    _ = detect_circle_polygon_collision_batch(test_circle_centers, test_circle_radii, polygons)
    _ = detect_circle_polygon_collision_batch(test_circle_centers, test_circle_radii, polygons)
    _ = detect_circle_polygon_collision_batch(test_circle_centers, test_circle_radii, polygons)
    time_0 = time.perf_counter()
    _ = detect_circle_polygon_collision_batch(test_circle_centers, test_circle_radii, polygons)
    mesh_time = time.perf_counter() - time_0
    _ = detect_circle_polygon_collision_batch(test_circle_centers, test_circle_radii, polygons_bb)
    _ = detect_circle_polygon_collision_batch(test_circle_centers, test_circle_radii, polygons_bb)
    _ = detect_circle_polygon_collision_batch(test_circle_centers, test_circle_radii, polygons_bb)

    time_0 = time.perf_counter()
    narrow_phase_mask = detect_circle_polygon_collision_batch(test_circle_centers, test_circle_radii, polygons_bb)
    # narrow_phase_indices = jnp.where(narrow_phase_mask, full_indices, -1).astype(jnp.int32)
    # print(narrow_phase_indices)
    broad_phase_time = time.perf_counter() - time_0
    centers = [test_circle_centers[i,:][narrow_phase_mask[i,:]] for i in range(4)]
    radii = [test_circle_radii[i,:][narrow_phase_mask[i,:]] for i in range(4)]
    centers = [torch.from_numpy(np.array(c)).cuda() for c in centers]
    radii = [torch.from_numpy(np.array(r)).cuda() for r in radii]
    centers = torch.cat(centers, dim=0)

    streams = [torch.cuda.Stream() for _ in range(4)]
    outputs = [None] * 4

    time_1 = time.perf_counter()
    d = mlps[0](centers)
    # for c, r, mlp in zip(centers, radii, mlps):
    #     d = mlp(c)
    # for i in range(4):
    #     with torch.cuda.stream(streams[i]):
    #         outputs[i] = mlps[i](centers[i])
    narrow_phase_time = time.perf_counter() - time_1
    print(f'Mesh time: {mesh_time * 1000:.3f} ms')
    print(f'MLP time: {(broad_phase_time + narrow_phase_time) * 1000:.3f} ms')

if __name__ == "__main__":
    main()
