import torch
import torch.nn as nn
import crown
import kd_tree
import shapely
import matplotlib
import numpy as np
from shapely.ops import split, unary_union
import matplotlib.pyplot as plt
from descartes import PolygonPatch
# torch.set_default_device(torch.device('cuda:0'))
from auto_LiRPA import BoundedModule
from shapely import wkt
import matplotlib as mpl

def sdf_circle(pts: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Signed distance to a circle.
    pts: (N, 2)
    center: (2,)
    radius: float
    Returns: (N, 1)
    """
    return (pts - center).norm(dim=1, keepdim=True) - radius

def sdf_square(pts: torch.Tensor, center: torch.Tensor, dims: float) -> torch.Tensor:
    """
    Signed distance to an axis-aligned square.
    pts: (N, 2)
    center: (2,)
    dims: float (side length)
    Returns: (N, 1)
    """
    q = torch.abs(pts - center) - dims / 2.0
    outside = torch.clamp(q, min=0)
    outside_dist = outside.norm(dim=1, keepdim=True)
    inside_dist = torch.clamp(q.max(dim=1, keepdim=True).values, max=0)
    return outside_dist + inside_dist

class CircleSDF(nn.Module):
    def __init__(self, center, radius, device='cpu'):
        super().__init__()
        self.register_buffer('center', torch.tensor(center, dtype=torch.float32, device=device))
        self.radius = radius

    def forward(self, pts):
        dist = pts - self.center
        return torch.sqrt(torch.sum(dist ** 2, dim=1, keepdim=True)) - self.radius

class SquareSDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('center', torch.tensor(-0.5))
        self.register_buffer('dims', torch.tensor(2.))

    def forward(self, pts):
        q = torch.abs(pts - self.center) - self.dims / 2.
        q0 = q[:, 0:1]
        q1 = q[:, 1:2]
        pow_outside_dist = torch.relu(q0) ** 2 + torch.relu(q1) ** 2
        outside_dist = torch.sqrt(pow_outside_dist)
        q2 = torch.nn.functional.relu(q1 - q0)
        q3 = q0 + q2
        inside_dist = -torch.nn.functional.relu(-q3)
        return outside_dist + inside_dist

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
    # print(a1, a2, b)
    # print(line)
    # print(square)
    segment = line.intersection(square)

    return segment

def carve(net, deep=True, inner=False, bbox=((-0.5, -0.5), (0.5, 0.5))):
    print("Carving")
    lower = torch.tensor(bbox[0])
    upper = torch.tensor(bbox[1])
    func = crown.CrownImplicitFunction(None, crown_func=net, crown_mode='crown', input_dim=2)
    if deep:
        # print("deep")
        lowers, uppers, lAs, lbs, uAs, ubs, pos_lowers, pos_uppers, neg_lowers, neg_uppers = kd_tree.construct_hybrid_unknown_tree(
            func, net, lower, upper, base_depth=15, max_depth=18, node_dim=2, include_pos_neg=True)
    else:
        # print("not deep")
        lowers, uppers, lAs, lbs, uAs, ubs, pos_lowers, pos_uppers, neg_lowers, neg_uppers = kd_tree.construct_hybrid_unknown_tree(
            func, net, lower, upper, base_depth=7, max_depth=8, node_dim=2, include_pos_neg=True)
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
    convex_poly_list = []
    # Include the negative nodes
    for n_l, n_u in zip(neg_lowers, neg_uppers):
        box_inside = shapely.geometry.Polygon([n_l, (n_l[0], n_u[1]), n_u, (n_u[0], n_l[1])])
        convex_poly_list.append(box_inside.buffer(0))

    # Include the negative portions of unknown nodes
    for l, u, lA, lb, uA, ub in zip(lowers, uppers, lAs, lbs, uAs, ubs):
        square = shapely.geometry.Polygon([l, (l[0], u[1]), u, (u[0], l[1])])
        if inner:
            if uA[0] == 0 and uA[1] == 0:
                continue
            inner_line = project_line_onto_square(uA[0], uA[1], ub, bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1])
            # plt.plot(inner_line.xy[0], inner_line.xy[1], c='k')
            slices1 = split(square, inner_line)
        else:
            if lA[0] == 0 and lA[1] == 0:
                continue
            outer_line = project_line_onto_square(lA[0], lA[1], lb, bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1])
            slices1 = split(square, outer_line)

        for g in slices1.geoms:
            if g.geom_type == 'Polygon':
                c = shapely.centroid(g)
                c = np.array([c.x, c.y])
                if inner:
                    cls = np.dot(uA, c) + ub
                else:
                    cls = np.dot(lA, c) + lb
                if cls < 0.:
                    convex_poly_list.append(g.buffer(0))
            else:
                print(g.geom_type)
    merged = shapely.ops.unary_union(convex_poly_list)

    # Extract the largest connected component
    if merged.geom_type == 'MultiPolygon':
        print('multipolygon')
        largest_component = max(merged.geoms, key=lambda p: p.area)
        return largest_component
    else:
        print('polygon')
        return merged


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def make_sdf_cm():
    sdf_cm_colors = [(0, '#ffffff'), (0.5 + 1e-5, '#489acc'), (0.5 + 1e-5, '#fffae3'), (1, '#ff7424')]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('SDF', sdf_cm_colors, N=256)
    return SDF_cmap(0.95, 'sdf_cmap', cmap._segmentdata)


"""
This class overrides the Colormap class and does some preprocesseing before the main call to the 
colormap to add alternating darkening to the isolines. It is only designed to work with data plotted
using mpl's countour/countourf methods. (eg. through plot_sdf below)
"""


class SDF_cmap(mpl.colors.LinearSegmentedColormap):
    def __init__(self, alph, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alph = alph

    def set_contours(self, cont):
        self.cont = cont

    def set_min_max(self, minv, maxv):
        self.minv = minv
        self.maxv = maxv

    def __call__(self, X, **kwargs):
        X = X.squeeze()
        n_c = X.shape[0]
        Xp = (1 / 2) * ((-X + 1) * (self.minv / self.maxv) + X + 1)
        colors = super().__call__(Xp, **kwargs)
        i = np.minimum(np.floor(X * (n_c)), n_c - 1);
        mask_is = np.repeat((i % 2 == 1)[:, None], 4, axis=1).squeeze()
        mult_mask = (np.ones(colors.shape)).squeeze()
        mult_mask[mask_is] = self.alph
        mult_mask[:, 3] = 1
        return mult_mask * colors


def plot_sdf(sdf, X, Y, **kwargs):
    pts = np.stack((X.flatten(), Y.flatten()), 1)
    return plot_sdf_from_vals(X, Y, sdf(pts), **kwargs)


def plot_sdf_from_vals(X, Y, Z, ax=None, colorbar=True, cmap=None, N=25, alpha=1, levels=None, return_levels=False):
    if cmap is None:
        cmap = make_sdf_cm()

    Z = np.reshape(Z, X.shape)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()

    if levels is None:
        levels = mpl.ticker.MaxNLocator(N).tick_values(np.min(Z), np.max(Z))

    try:
        # If this is an SDF_cmap, want to set its min/max
        cmap.set_min_max(levels[0], levels[-1])
    except:
        pass
    # norm = mpl.colors.TwoSlopeNorm(vmin=np.min(Z), vcenter=0.0, vmax=np.max(Z))

    cp = ax.contourf(X, Y, Z, levels, cmap=cmap, alpha=alpha)

    if colorbar:
        fig.colorbar(cp, ax=ax)

    ax.set_aspect('equal', 'box')

    if return_levels:
        return cp, levels
    return cp


def make_grid(*dims, res=100):
    return np.meshgrid(*[np.linspace(*dim, res) for dim in dims])


# def plot_sdf_from_vals(X, Y, Z, ax=None, colorbar=True, cmap=None, N=25, alpha=1, levels=None, return_levels=False):
#     if cmap is None:
#         cmap = make_sdf_cm()
#
#     Z = np.reshape(Z, X.shape)
#
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.gca()
#     else:
#         fig = plt.gcf()
#
#     if levels is None:
#         levels = mpl.ticker.MaxNLocator(N).tick_values(np.min(Z), np.max(Z))
#
#     # if levels is None:
#     #     zmin = np.min(Z)
#     #     zmax = np.max(Z)
#     #     zlim = max(abs(zmin), abs(zmax))
#     #     levels = np.linspace(-zlim, zlim, N)
#
#     try:
#         # If this is an SDF_cmap, want to set its min/max
#         cmap.set_min_max(levels[0], levels[-1])
#     except:
#         pass
#     norm = mpl.colors.TwoSlopeNorm(vmin=np.min(Z), vcenter=0.0, vmax=np.max(Z))
#
#     cp = ax.contourf(X, Y, Z, levels, cmap=cmap, norm=norm, alpha=alpha)
#
#     if colorbar:
#         fig.colorbar(cp, ax=ax)
#
#     ax.set_aspect('equal', 'box')
#
#     if return_levels:
#         return cp, levels
#     return cp

import time

# def point_to_polygon_distance(points: torch.Tensor, polygon: torch.Tensor, signed: bool = False):
#     """
#     Compute distance from 2D points to a 2D polygon.
#
#     Args:
#         points (torch.Tensor): Tensor of shape (N, 2) for N points.
#         polygon (torch.Tensor): Tensor of shape (M, 2) for M polygon vertices (assumed ordered and forming a closed polygon).
#         signed (bool): Whether to compute signed distance (negative if inside the polygon).
#
#     Returns:
#         distances (torch.Tensor): Tensor of shape (N,) with (signed) distances to the polygon.
#     """
#     N = points.shape[0]
#     M = polygon.shape[0]
#
#     # Form polygon edges: (M, 2) -> (M, 2), (M, 2)
#     v0 = polygon
#     v1 = torch.roll(polygon, shifts=-1, dims=0)  # Next vertex (wraps around)
#
#     # Reshape for broadcasting
#     p = points[:, None, :]       # (N, 1, 2)
#     a = v0[None, :, :]           # (1, M, 2)
#     b = v1[None, :, :]           # (1, M, 2)
#
#     ab = b - a                   # (1, M, 2)
#     ap = p - a                   # (N, M, 2)
#
#     t = (ap * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)  # (N, M)
#     t = t.clamp(0, 1)                                       # project to edge segment
#     projection = a + t[..., None] * ab                      # (N, M, 2)
#     dists = ((p - projection)**2).sum(-1).sqrt()            # (N, M)
#     min_dists, _ = dists.min(dim=1)                         # (N,)
#
#     if not signed:
#         return min_dists
#
#     # Ray casting for inside test
#     x, y = points[:, 0], points[:, 1]  # (N,)
#
#     x0, y0 = v0[:, 0], v0[:, 1]        # (M,)
#     x1, y1 = v1[:, 0], v1[:, 1]        # (M,)
#
#     cond1 = ((y0 <= y[:, None]) & (y1 > y[:, None])) | ((y1 <= y[:, None]) & (y0 > y[:, None]))
#     slope = (x1 - x0) / (y1 - y0 + 1e-9)
#     xinters = x0 + slope * (y[:, None] - y0)
#
#     crossings = (x[:, None] < xinters) & cond1
#     inside = crossings.sum(dim=1) % 2 == 1
#
#     signed_dists = min_dists * (~inside * 2 - 1).float()
#     return signed_dists

import warp as wp
import numpy as np


@wp.kernel
def point_to_polygon_distance_kernel(
        points: wp.array(dtype=wp.vec2),
        polygon: wp.array(dtype=wp.vec2),
        num_edges: int,
        out_distances: wp.array(dtype=wp.float32)
):
    tid = wp.tid()

    p = points[tid]
    min_dist = float(1e30)

    for i in range(num_edges):
        a = polygon[i]
        b = polygon[(i + 1) % num_edges]

        ab = b - a
        ap = p - a
        ab_len2 = wp.dot(ab, ab)

        # Projection factor t
        t = wp.dot(ap, ab) / (ab_len2 + 1e-8)
        t = wp.clamp(t, 0.0, 1.0)

        # Closest point on segment
        proj = a + t * ab
        d = wp.length(p - proj)

        if d < min_dist:
            min_dist = d

    out_distances[tid] = min_dist

def compute_point_to_polygon_distance_warp(points_np: np.ndarray, polygon_np: np.ndarray) -> tuple[np.ndarray, float]:
    assert points_np.shape[1] == 2
    assert polygon_np.shape[1] == 2

    device = wp.get_preferred_device()

    points = wp.array(points_np.astype(np.float32), dtype=wp.vec2, device=device)
    polygon = wp.array(polygon_np.astype(np.float32), dtype=wp.vec2, device=device)
    distances = wp.empty(len(points_np), dtype=wp.float32, device=device)

    # Warmup kernel (for accurate timing)
    wp.launch(
        kernel=point_to_polygon_distance_kernel,
        dim=len(points_np),
        inputs=[points, polygon, polygon.shape[0], distances],
        device=device
    )
    wp.synchronize()

    # Timed kernel
    with wp.ScopedTimer("mesh SD") as timer:
        wp.launch(
            kernel=point_to_polygon_distance_kernel,
            dim=len(points_np),
            inputs=[points, polygon, polygon.shape[0], distances],
            device=device
        )
        wp.synchronize()

    return distances.numpy()


import warp as wp

@wp.kernel
def signed_point_to_polygon_distance_kernel(
    points: wp.array(dtype=wp.vec2),
    polygon: wp.array(dtype=wp.vec2),
    num_edges: int,
    out_signed_distances: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    p = points[tid]
    px = p[0]
    py = p[1]

    min_dist = float(1e30)
    inside = bool(False)

    for i in range(num_edges):
        a = polygon[i]
        b = polygon[(i + 1) % num_edges]

        # ---- Distance part ----
        ab = b - a
        ap = p - a
        ab_len2 = wp.dot(ab, ab)
        t = wp.dot(ap, ab) / (ab_len2 + 1e-8)
        t = wp.clamp(t, 0.0, 1.0)
        proj = a + t * ab
        d = wp.length(p - proj)
        if d < min_dist:
            min_dist = d

        # ---- Ray casting part (horizontal ray to +x) ----
        ay = a[1]
        by = b[1]
        ax = a[0]
        bx = b[0]

        cond1 = (ay <= py and by > py) or (by <= py and ay > py)
        if cond1:
            slope = (bx - ax) / (by - ay + 1e-8)
            x_intersect = ax + slope * (py - ay)
            if px < x_intersect:
                inside = not inside  # flip inside flag

    if inside:
        signed_dist = -min_dist
    else:
        signed_dist = min_dist

    out_signed_distances[tid] = signed_dist


def compute_signed_distance_warp(points_np: np.ndarray, polygon_np: np.ndarray) -> tuple[np.ndarray, float]:
    assert points_np.shape[1] == 2
    assert polygon_np.shape[1] == 2

    device = wp.get_preferred_device()

    points = wp.array(points_np.astype(np.float32), dtype=wp.vec2, device=device)
    polygon = wp.array(polygon_np.astype(np.float32), dtype=wp.vec2, device=device)
    signed_distances = wp.empty(len(points_np), dtype=wp.float32, device=device)

    # Warm-up
    wp.launch(
        kernel=signed_point_to_polygon_distance_kernel,
        dim=len(points_np),
        inputs=[points, polygon, polygon.shape[0], signed_distances],
        device=device
    )
    wp.synchronize()

    # Timed run
    with wp.ScopedTimer("SDF") as timer:
        wp.launch(
            kernel=signed_point_to_polygon_distance_kernel,
            dim=len(points_np),
            inputs=[points, polygon, polygon.shape[0], signed_distances],
            device=device
        )
        wp.synchronize()

    return signed_distances.numpy()



def plot_poly_sdf(polygon, hole, sdf1, sdf2, save=False, domain=None):
    if domain is None:
        domain = ([-2, 2], [-2, 2])
    X, Y = make_grid(*domain, res=500)

    test_pts = torch.from_numpy(np.stack((X.flatten(), Y.flatten()), 1).astype('float32'))
    pred_Z = []

    _ = compute_signed_distance_warp(test_pts.numpy(), np.array(polygon.exterior.coords))
    test_pts = test_pts.cuda()
    start = time.time()
    circle_sdf = sdf_circle(test_pts, center=torch.tensor([0.5, 0.5], device='cuda'), radius=1.2)
    end = time.time()
    print(f'Time elapsed: {1000.0 * (end - start)}ms')
    start = time.time()
    square_sdf = sdf_square(test_pts, center=torch.tensor([-0.5, -0.5], device='cuda'), dims=2.0)
    end = time.time()
    print(f'Time elapsed: {1000.0 * (end - start)}ms')
    # return
    test_pts = test_pts.cpu()
    time_start = time.perf_counter()
    for point in test_pts:
        p = shapely.Point(point)
        if hole.contains(p):
            pred_Z.append(-polygon.distance(p)-0.001)
        elif polygon.contains(p):
            pred_Z.append(min(sdf1(point.unsqueeze(0))[0][0].item(), sdf2(point.unsqueeze(0))[0][0].item()))
        else:
            pred_Z.append(polygon.distance(p)+0.001)
    time_end = time.perf_counter()
    print(f'Time elapsed: {1000.0 * (time_end - time_start)}ms')
    # verts_tensor = torch.from_numpy(np.array(polygon.exterior.coords))

    print(test_pts.shape)
    # pred_Z = np.array([polygon.distance(shapely.Point(point)) for point in test_pts])
    pred_Z = np.array(pred_Z).astype(np.float32)
    baseline_Z = np.load('csg_sdf.npy').astype(np.float32).flatten()
    gt_Z = np.load('exp_results/csg/gt_csg.npz')['sdf'].astype(np.float32)
    diff_Z1 = np.abs(gt_Z - pred_Z).astype(np.float32)
    diff_Z2 = np.abs(gt_Z - baseline_Z).astype(np.float32)
    print(diff_Z1.mean(), diff_Z2.mean())

    # fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns
    #
    # # Plot the first image
    # im1 = axes[0].imshow(diff_Z1.reshape((500, 500, 1)), cmap='viridis', origin='lower')
    # axes[0].set_title(f'Shell+SDF,\n error={diff_Z1.mean():04f}')
    # axes[0].axis('off')  # Hide axes ticks and labels
    #
    # # Plot the second image
    # im2 = axes[1].imshow(diff_Z2.reshape((500, 500, 1)), cmap='viridis', origin='lower')
    # axes[1].set_title(f'Optimized SDF,\n mean error={diff_Z2.mean():04f}')
    # axes[1].axis('off')
    # fig.colorbar(im1, ax=axes.ravel().tolist(), label='L1 Error', fraction=0.046, pad=0.04)
    # plt.show()

    viz_fig, viz_ax = plt.subplots(figsize=(8, 8))
    plot_sdf_from_vals(X, Y, pred_Z, colorbar=False, ax=viz_ax, N=50, alpha=1)

    square_centers = [(-0.5, -0.5), (0.5, 0.5)]
    square_size = 0.5

    for cx, cy in square_centers:
        lower_left = (cx - square_size / 2, cy - square_size / 2)
        square = mpl.patches.Rectangle(
            lower_left, square_size, square_size,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        viz_ax.add_patch(square)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    # plt.savefig('assets/images/shell_csg.png', bbox_inches='tight', dpi=300, transparent=True)

    plt.show()


if __name__ == '__main__':
    circle_sdf = CircleSDF(torch.tensor([[0.5, 0.5]]), 1.2)
    circle_outer = carve(circle_sdf, deep=True, bbox=((-0.7, -0.7), (1.7, 1.7)))
    circle_inner = carve(circle_sdf, deep=True, inner=True, bbox=((-0.7, -0.7), (1.7, 1.7)))
    # x, y = circle_outer.exterior.xy

    # # Plot the polygon
    # plt.plot(x, y, marker='o', color='red', linestyle='-')
    # plt.fill(x, y, color='blue', alpha=0.3)  # Optional: fill the shape
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Polygon Plot")
    # plt.grid()

    # x, y = circle_inner.exterior.xy
    #
    # # Plot the polygon
    # plt.plot(x, y, marker='o', color='blue', linestyle='-')
    # plt.fill(x, y, color='blue', alpha=0.3)  # Optional: fill the shape
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Polygon Plot")
    # plt.grid()

    square_sdf = SquareSDF()

    square_outer = carve(square_sdf, deep=True, bbox=((-2.5, -2.5), (2.5, 2.5)))
    square_inner = carve(square_sdf, deep=True, inner=True, bbox=((-2.5, -2.5), (2.5, 2.5)))
    x, y = square_outer.exterior.xy

    # Plot the polygon
    # plt.plot(x, y, marker='o', color='red', linestyle='-')
    # plt.fill(x, y, color='blue', alpha=0.3)  # Optional: fill the shape
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Polygon Plot")
    # plt.grid()
    #
    # x, y = square_inner.exterior.xy
    #
    # # Plot the polygon
    # plt.plot(x, y, marker='o', color='blue', linestyle='-')
    # plt.fill(x, y, color='blue', alpha=0.3)  # Optional: fill the shape
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Polygon Plot")
    # plt.grid()
    #
    # plt.show()

    union_outer = shapely.union(circle_outer, square_outer)
    union_inner = shapely.union(circle_inner, square_inner)

    union_shell = shapely.difference(union_outer, union_inner)
    plot_poly_sdf(union_shell, union_inner, circle_sdf, square_sdf)
