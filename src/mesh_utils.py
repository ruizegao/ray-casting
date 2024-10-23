import numpy as np
import torch

def points_on_cube_edges(lower_corner, upper_corner, points):
    x_min, y_min, z_min = lower_corner
    x_max, y_max, z_max = upper_corner
    edge_points = []

    for point in points:
        x, y, z = point

        # Check if the point lies on any of the 12 edges
        is_on_x_edges = (x == x_min or x == x_max) and (y == y_min or y == y_max or z == z_min or z == z_max)
        is_on_y_edges = (y == y_min or y == y_max) and (x == x_min or x == x_max or z == z_min or z == z_max)
        is_on_z_edges = (z == z_min or z == z_max) and (x == x_min or x == x_max or y == y_min or y == y_max)

        # A point is on the edge if it satisfies any of the edge conditions
        if is_on_x_edges or is_on_y_edges or is_on_z_edges:
            edge_points.append(point)

    return torch.tensor(edge_points)

def sort_vertices(vertices, order='CCW'):
    """
    Sorts a list of vertices in 3D space in a specified order (CW or CCW) using Shapely.

    Args:
        vertices (ndarray): A numpy array of shape (n, 3), representing vertices in 3D space.
        order (str): 'CCW' for counterclockwise, 'CW' for clockwise order.

    Returns:
        ndarray: The sorted vertices in the specified order.
    """
    # Project 3D points to 2D (assuming the points are coplanar)
    centroid = torch.mean(vertices, dim=0)

    # Project the points onto the XY plane for sorting
    points_2d = vertices[:, :2]

    # Calculate angles with respect to the centroid
    angles = torch.arctan2(points_2d[:, 1] - centroid[1], points_2d[:, 0] - centroid[0])

    # Sort based on angles
    sorted_indices = torch.argsort(angles)

    # If CW order is requested, reverse the sorted indices
    if order == 'CW':
        sorted_indices = sorted_indices[::-1]

    return vertices[sorted_indices]

def triangulate(verts):
    triangles = []
    for i in range(1, len(verts) - 1):
        triangles.append([0, i, i+1])

    return torch.tensor(triangles)


if __name__ == '__main__':
    import trimesh
    bounds = np.array([[-1., -1., -1.], [2., 2., 2.]])
    lower = bounds[0]
    upper = bounds[1]
    cube = trimesh.creation.box(bounds=bounds)
    cube_verts = np.array(cube.vertices)
    # origin = [0.5, 0.5, 0.5]
    offset = -10
    normal = np.array([1, 1, 1])
    origin = np.array([0., 0., - offset / normal[2]])
    neg_mask = (np.matmul(cube_verts, normal) + offset) < 0
    print(neg_mask)
    print(cube_verts[neg_mask])
    # Slice the cube mesh with the plane
    # slice_mesh = cube.section(plane_origin=origin, plane_normal=normal)
    # vertices = slice_mesh.vertices
    #
    # vertices = sort_vertices(points_on_cube_edges(lower_corner=lower, upper_corner=upper, points=vertices))
    # print(len(vertices))
    # tri = triangulate(vertices)
    # mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=tri, process=False)
    # mesh.show()

    mesh_pos = cube.slice_plane(origin, normal, cap=True)
    print(mesh_pos.faces)
    # mesh_pos.show()
    # print(len(mesh_pos.vertices))

