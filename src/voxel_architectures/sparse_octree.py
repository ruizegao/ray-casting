import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class OctreeNode:
    def __init__(self, boundary, points=None):
        self.boundary = boundary  # [center_x, center_y, center_z, size]
        self.points = points if points is not None else []
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def contains_point(self, point):
        cx, cy, cz, size = self.boundary
        half_size = size / 2
        return all([
            cx - half_size <= point[0] < cx + half_size,
            cy - half_size <= point[1] < cy + half_size,
            cz - half_size <= point[2] < cz + half_size
        ])

    def subdivide(self):
        cx, cy, cz, size = self.boundary
        half_size = size / 2
        child_size = size / 2
        # Create 8 children
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    new_center = torch.tensor(
                        [cx + dx * half_size / 2, cy + dy * half_size / 2, cz + dz * half_size / 2])
                    self.children.append(OctreeNode(boundary=[new_center[0], new_center[1], new_center[2], child_size]))

    def insert(self, point, max_points=4):
        if not self.contains_point(point):
            return False

        if len(self.points) < max_points or self.boundary[3] <= 1.0:
            # Insert the point if max capacity not reached or node too small to subdivide further
            self.points.append(point)
            return True

        if self.is_leaf():
            # Subdivide if it's a leaf and max capacity reached
            self.subdivide()

        # Recursively insert into children
        for child in self.children:
            if child.insert(point):
                return True

        return False


class Octree:
    def __init__(self, boundary, points, max_points=4):
        self.root = OctreeNode(boundary)
        for point in points:
            self.root.insert(point, max_points=max_points)

    def insert(self, point):
        return self.root.insert(point)


def ray_intersects_box(ray_origin, ray_direction, box_min, box_max):
    inv_dir = 1.0 / ray_direction
    tmin = (box_min - ray_origin) * inv_dir
    tmax = (box_max - ray_origin) * inv_dir

    tmin = torch.max(torch.min(tmin, tmax), dim=0).values
    tmax = torch.min(torch.max(tmin, tmax), dim=0).values

    return tmax > torch.max(tmin, torch.tensor([0.0]))


# Raycast function modified to track intersected nodes
def raycast(node, ray_origin, ray_direction, intersected_nodes):
    if node.is_leaf():
        if len(node.points) > 0:
            intersected_nodes.append(node.boundary)
        return node.points

    half_size = node.boundary[3] / 2
    min_corner = torch.tensor(
        [node.boundary[0] - half_size, node.boundary[1] - half_size, node.boundary[2] - half_size])
    max_corner = torch.tensor(
        [node.boundary[0] + half_size, node.boundary[1] + half_size, node.boundary[2] + half_size])

    if ray_intersects_box(ray_origin, ray_direction, min_corner, max_corner):
        if node.is_leaf():
            intersected_nodes.append(node.boundary)
        for child in node.children:
            raycast(child, ray_origin, ray_direction, intersected_nodes)

    return intersected_nodes

# Function to draw a cube (node boundary) in 3D
def draw_cube(ax, boundary, color='blue', alpha=0.2):
    cx, cy, cz, size = boundary
    r = size / 2
    # Define the vertices of the cube
    vertices = [[cx - r, cy - r, cz - r], [cx + r, cy - r, cz - r], [cx + r, cy + r, cz - r], [cx - r, cy + r, cz - r],
                [cx - r, cy - r, cz + r], [cx + r, cy - r, cz + r], [cx + r, cy + r, cz + r], [cx - r, cy + r, cz + r]]
    # Define the faces of the cube
    faces = [[vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]],
             [vertices[j] for j in [0, 1, 5, 4]],
             [vertices[j] for j in [2, 3, 7, 6]],
             [vertices[j] for j in [0, 3, 7, 4]],
             [vertices[j] for j in [1, 2, 6, 5]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='r', alpha=alpha))

# Visualization of octree and intersecting ray
def visualize_octree(octree, ray_origin, ray_direction, intersected_nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the root node (full octree)
    draw_cube(ax, octree.root.boundary, color='gray', alpha=0.1)

    # Plot intersected nodes in a different color
    for node_boundary in intersected_nodes:
        draw_cube(ax, node_boundary, color='red', alpha=0.5)

    # Plot the ray
    t = torch.linspace(0, 100, 100)
    ray_line = ray_origin.unsqueeze(0) + ray_direction.unsqueeze(0) * t.unsqueeze(1)
    ax.plot(ray_line[:, 0], ray_line[:, 1], ray_line[:, 2], color='green', label='Ray')

    # Set plot limits
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])

    plt.show()


if __name__ == "__main__":
    # Create random 3D points (for example)
    num_points = 1000
    points = torch.rand(num_points, 3) * 100  # Random points in a 100x100x100 space

    # Build the octree
    boundary = [50.0, 50.0, 50.0, 100.0]  # Centered at (50, 50, 50) with size 100
    octree = Octree(boundary, points)

    # Perform ray casting
    ray_origin = torch.tensor([0.0, 0.0, 0.0])
    ray_direction = torch.tensor([1.0, 1.0, 1.0])
    ray_direction /= torch.norm(ray_direction) # normalize it to be a unit vector
    # Get intersected nodes
    intersected_nodes = []
    hits = raycast(octree.root, ray_origin, ray_direction, intersected_nodes)

    # Visualize the octree and intersections
    visualize_octree(octree, ray_origin, ray_direction, intersected_nodes)
