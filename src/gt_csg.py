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

import math
import torch

def point_to_segment_distance(p, a, b):
    pa = p - a
    ba = b - a
    h = torch.clamp((pa @ ba) / (ba @ ba), 0.0, 1.0)
    proj = a + h.unsqueeze(-1) * ba
    return torch.norm(p - proj, dim=-1)

def point_to_arc_distance(p, center, radius, start_angle, end_angle):
    v = p - center
    d = torch.norm(v, dim=-1)
    angle = torch.atan2(v[:, 1], v[:, 0]) % (2 * math.pi)
    start_angle %= 2 * math.pi
    end_angle %= 2 * math.pi

    # Handle angle wrap-around
    if start_angle < end_angle:
        mask = (angle >= start_angle) & (angle <= end_angle)
    else:
        mask = (angle >= start_angle) | (angle <= end_angle)

    arc_dist = torch.abs(d - radius)
    arc_start = center + radius * torch.tensor([
        math.cos(start_angle), math.sin(start_angle)
    ])
    arc_end = center + radius * torch.tensor([
        math.cos(end_angle), math.sin(end_angle)
    ])
    dist_to_start = torch.norm(p - arc_start, dim=-1)
    dist_to_end = torch.norm(p - arc_end, dim=-1)

    return torch.where(mask, arc_dist, torch.minimum(dist_to_start, dist_to_end))

def sdf_union_square_circle(p):
    square_edges = [
        (torch.tensor([-1.5, -1.5]), torch.tensor([0.5, -1.5])),  # bottom
        (torch.tensor([0.5, -1.5]), torch.tensor([0.5, -0.7])),  # right (truncated)
        (torch.tensor([-0.7, 0.5]), torch.tensor([-1.5, 0.5])),  # top (truncated)
        (torch.tensor([-1.5, 0.5]), torch.tensor([-1.5, -1.5])),  # left
    ]

    d_line = torch.stack([
        point_to_segment_distance(p, a, b) for a, b in square_edges
    ], dim=1).min(dim=1).values

    # Circle centered at (0.5, 0.5) with radius 1.2
    arc_center = torch.tensor([0.5, 0.5])
    arc_radius = 1.2
    # Arc angles: roughly from -90° to 180°
    d_arc = point_to_arc_distance(
        p, arc_center, arc_radius,
        start_angle=- math.pi / 2,  # 135°
        end_angle= math.pi     # 315°
    )

    # Signed interior check: inside square or inside circle
    inside_square = (
        (p[:, 0] >= -1.5) & (p[:, 0] <= 0.5) &
        (p[:, 1] >= -1.5) & (p[:, 1] <= 0.5)
    )
    inside_circle = torch.norm(p - arc_center, dim=-1) <= arc_radius
    inside_union = inside_square | inside_circle

    return torch.minimum(d_line, d_arc) * (1 - 2 * inside_union.float())

if __name__ == '__main__':
    square_min = torch.tensor([-1.5, -1.5])
    square_max = torch.tensor([0.5, 0.5])
    circle_center = torch.tensor([0.5, 0.5])
    circle_radius = 1.2

    x, y = torch.meshgrid(torch.linspace(-2, 2, 500), torch.linspace(-2, 2, 500), indexing='ij')
    points = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)

    # Compute SDF
    import time
    time_start = time.perf_counter()
    sdf = sdf_union_square_circle(points)
    time_end = time.perf_counter()
    print(f'Time elapsed: {time_end - time_start:.3f} s')
    # sdf_image = sdf.reshape(100, 100)
    np.savez('exp_results/csg/gt_csg.npz', sdf=sdf)
    viz_fig, viz_ax = plt.subplots(figsize=(8, 8))
    plot_sdf_from_vals(x, y, sdf.detach().cpu().numpy().astype(np.float32), colorbar=False, ax=viz_ax, N=50, alpha=1)

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

    plt.savefig('assets/images/gt_csg.png', bbox_inches='tight', dpi=300, transparent=True)

    plt.show()