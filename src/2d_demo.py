import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def intersect_plane_with_cube(a, b, c, d, cube_min=0.0, cube_max=1.0):
    # Plane equation: ax + by + cz + d = 0
    # Cube has 12 edges. We test intersection of plane with each.
    cube_edges = [
        # bottom square
        [(0,0,0), (1,0,0)], [(1,0,0), (1,1,0)], [(1,1,0), (0,1,0)], [(0,1,0), (0,0,0)],
        # top square
        [(0,0,1), (1,0,1)], [(1,0,1), (1,1,1)], [(1,1,1), (0,1,1)], [(0,1,1), (0,0,1)],
        # vertical edges
        [(0,0,0), (0,0,1)], [(1,0,0), (1,0,1)],
        [(1,1,0), (1,1,1)], [(0,1,0), (0,1,1)],
    ]

    intersection_points = []
    for p1, p2 in cube_edges:
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        direction = p2 - p1
        denom = a * direction[0] + b * direction[1] + c * direction[2]
        if denom == 0:
            continue  # Parallel, no intersection
        t = -(a * p1[0] + b * p1[1] + c * p1[2] + d) / denom
        if 0 <= t <= 1:
            point = p1 + t * direction
            if np.all((point >= cube_min) & (point <= cube_max)):
                intersection_points.append(tuple(point))

    # Remove duplicates
    intersection_points = list(set(intersection_points))

    # Convex hull ordering (optional for neat rendering)
    if len(intersection_points) >= 3:
        from scipy.spatial import ConvexHull
        points_2d = np.array(intersection_points)[:, :2]  # project to xy
        try:
            hull = ConvexHull(points_2d)
            ordered_points = [intersection_points[i] for i in hull.vertices]
        except:
            ordered_points = intersection_points
    else:
        ordered_points = intersection_points

    return ordered_points

# ========== Start of Plotting ==========
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)
ax.xaxis.line.set_color((1., 1., 1., 1.))
ax.yaxis.line.set_color((1., 1., 1., 1.))
ax.zaxis.line.set_color((1., 1., 1., 1.))

# Sphere
u = np.linspace(0, 0.5 * np.pi, 100)
v = np.linspace(0, 0.5 * np.pi, 100)
u, v = np.meshgrid(u, v)
r = 0.7
x_sphere = 1. - r * np.cos(u) * np.sin(v)
y_sphere = 1. - r * np.sin(u) * np.sin(v)
z_sphere = 1. - r * np.cos(v)
mask_sphere = (x_sphere >= 0) & (x_sphere <= 1) & (y_sphere >= 0) & (y_sphere <= 1) & (z_sphere >= 0) & (z_sphere <= 1)
x_sphere_masked = np.ma.masked_where(~mask_sphere, x_sphere)
y_sphere_masked = np.ma.masked_where(~mask_sphere, y_sphere)
z_sphere_masked = np.ma.masked_where(~mask_sphere, z_sphere)
ax.plot_surface(x_sphere_masked, y_sphere_masked, z_sphere_masked, color='#b3b3b3', alpha=1.0, linewidth=0)

# Plot clipped planes as polygons
plane1_pts = intersect_plane_with_cube(1, 1, 1, -2.4)
plane2_pts = intersect_plane_with_cube(1, 1, 1, -1.5)

if len(plane1_pts) >= 3:
    ax.add_collection3d(Poly3DCollection([plane1_pts], color='#00cc00', alpha=0.5, edgecolor='k'))

if len(plane2_pts) >= 3:
    ax.add_collection3d(Poly3DCollection([plane2_pts], color='#6666ff', alpha=0.5, edgecolor='k'))

# Cube edges
cube_lines = np.array([
    [[0,0,0],[1,0,0]], [[0,0,0],[0,1,0]], [[0,0,0],[0,0,1]],
    [[1,1,1],[0,1,1]], [[1,1,1],[1,0,1]], [[1,1,1],[1,1,0]],
    [[0,1,0],[0,1,1]], [[0,1,0],[1,1,0]],
    [[1,0,0],[1,0,1]], [[1,0,0],[1,1,0]],
    [[0,0,1],[1,0,1]], [[0,0,1],[0,1,1]],
])
for line in cube_lines:
    ax.plot(*zip(*line), color="black", linewidth=1)

cube_vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

# Define cube faces (each face is a list of 4 vertices)
cube_faces = [
    [cube_vertices[i] for i in [0, 1, 2, 3]],  # bottom
    # [cube_vertices[i] for i in [4, 5, 6, 7]],  # top
    # [cube_vertices[i] for i in [0, 1, 5, 4]],  # front
    [cube_vertices[i] for i in [2, 3, 7, 6]],  # back
    # [cube_vertices[i] for i in [1, 2, 6, 5]],  # right
    [cube_vertices[i] for i in [0, 3, 7, 4]]   # left
]

# Add the colored cube
ax.add_collection3d(Poly3DCollection(
    cube_faces,
    facecolors='#f8cecc',  # Or any color like '#ffcccc'
    linewidths=0.5,
    # linestyles='dashed',
    edgecolors='none',
    alpha=1.  # Adjust transparency here
))

for face in cube_faces:
    for i in range(len(face)):
        start = face[i]
        end = face[(i + 1) % len(face)]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color='red',
            linestyle='dashed',
            linewidth=0.8
        )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

plt.tight_layout()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
