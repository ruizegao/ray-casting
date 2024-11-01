import numpy as np
from torch import Tensor
from torch import vmap
from functools import partial
from numpy import ndarray
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from enum import Enum
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
import matplotlib.patches as mpatches
from math import ceil
import imageio, pickle, logging, torch, os

set_t = {
    "dtype": torch.float32,
    "device": torch.device("cuda"),
}
out_path = "/home/jorgejc2/Documents/Research/ray-casting/plane_training/"

delaunay_logging_file = os.getenv('DELAUNAY_LOGGING', None)
if delaunay_logging_file is not None:
    logger = logging.getLogger('delaunay_logger')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(delaunay_logging_file)
    fh.setLevel(logging.ERROR)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    print(f"Printing log handlers: {logger.handlers}")
else:
    logger = None

class VolumeCalculationMethod(Enum):
    SinglePlaneSingleVolume = 1
    SinglePlaneBatchVolume = 2
    MultiPlaneSingleVolume = 3
    MultiPlaneBatchVolume = 4
    MiscMain = 5

class OptimizerMethod(Enum):
    Adam = 1
    SGD = 2

optimizers_config = {
    OptimizerMethod.Adam: {
        "class": torch.optim.Adam,
        "params": {
            "lr": 0.1
        }
    },
    OptimizerMethod.SGD: {
        "class": torch.optim.SGD,
        "params": {
            "lr": 0.01,
            "momentum": 0.9
        }
    }
}

def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

# Function to define the unit cube vertices
def get_cube_vertices(x_L: Tensor, x_U: Tensor) -> Tensor:
    """
    Given the lower and upper bounds on the input domain, generate vertices in lexicographical order by using the cartesian product operator.
    :param x_L: Input lower bounds
    :param x_U: Input upper bounds
    :return:    Vertices in lexicographic order
    """
    x_L = x_L.reshape(1, -1)
    x_U = x_U.reshape(1, -1)
    cat_tensor = torch.cat((x_L, x_U), 0).permute(1, 0)
    points = torch.cartesian_prod(*cat_tensor)
    return points


# Function to define the faces of a unit cube
def unit_cube_faces() -> ndarray:
    # Each face consists of 4 vertices (indices refer to the vertex array)
    return np.array([[0, 1, 5, 4],  # Bottom face
            [2, 3, 7, 6],  # Top face
            [0, 1, 3, 2],  # Front face
            [4, 5, 7, 6],  # Back face
            [0, 2, 6, 4],  # Left face
            [1, 3, 7, 5]])  # Right face


def plot_plane(
        ax: Axes,
        normal: ndarray,
        D: float,
        xlim: list[Union[float, int]],
        ylim: list[Union[float, int]],
        samples: int = 1000
):
    """

    :param ax:      Graph plotting object
    :param normal:  Plane coefficients
    :param D:       Plane offset
    :param xlim:    Graph x limits
    :param ylim:    Graph y limits
    :return:
    """
    [A, B, C] = normal
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], samples), np.linspace(ylim[0], ylim[1], samples))
    zz = (-D - A * xx - B * yy) / C
    ax.plot_surface(xx, yy, zz, alpha=0.5)


def plot_cube(ax: Axes, vertices: ndarray, faces: ndarray):
    """

    :param ax:          Graph plotting object
    :param vertices:    Cube vertex coordinates
    :param faces:       Index mask into vertices describing the face in clockwise order
    :return:
    """
    # Plot the cube's faces
    for face in faces:
        verts = [vertices[face]]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.2, edgecolor='k'))


def generate_cube_edges() -> list[Tuple[int, int]]:
    """
    These are the edges for vertices represented by their lexicographic order.
    :return:
    """
    edges = []
    num_vertices = 8  # 2^3 for a cube

    for v1 in range(num_vertices):  # Vertices represented as 0 to 7 (3-bit numbers)
        for bit in range(3):  # Flip each of the 3 bits (positions 0, 1, 2)
            v2 = v1 ^ (1 << bit)  # XOR to flip the 'bit'-th bit
            if v2 > v1:  # Ensure we only add each edge once
                edges.append((v1, v2))

    return edges


def plot_intersection(ax: Axes, vertices: ndarray, normal: ndarray, D: float):
    """

    :param ax:          Graph plotting object
    :param vertices:    Cube vertex coordinates
    :param normal:      Plane normal coefficients
    :param D:           Plane offset
    :return:
    """
    [A, B, C] = normal
    # Compute distance from each vertex to the plane
    distances = A * vertices[:, 0] + B * vertices[:, 1] + C * vertices[:, 2] + D

    # Identify vertices below and above the plane
    below = vertices[distances < 0]
    above = vertices[distances >= 0]

    # Plot intersection points
    ax.scatter(below[:, 0], below[:, 1], below[:, 2], color='r', label='Below Plane')
    ax.scatter(above[:, 0], above[:, 1], above[:, 2], color='b', label='Above Plane')


def calculate_intersection(edges: Tensor, normal: Tensor, D: float, verbose=False
                           ) -> Tuple[Tensor, Tensor, Tensor]:
    """

    :param edges:   Edge vertex pairs
    :param normal:  Plane normal coefficients
    :param D:       Plane offset
    :param verbose: Prints debugging logs
    :return:
    """

    batches = edges.shape[0]

    # For 3D boxes, batch = 12
    Vis = edges[:, 0, :].reshape(batches, 3, 1)
    Vjs = edges[:, 1, :].reshape(batches, 3, 1)
    e_ij = Vjs - Vis  # (batch x 3 x 1)

    # convert to Hessian Normal Form
    mag = torch.linalg.norm(normal)
    hn_normal = normal / mag
    d = -D / mag

    normal_exp = hn_normal.reshape(1, 1, -1).expand(batches, -1, -1)
    if verbose:
        print(f"Shape Vis: {Vis.shape}")
        print(f"Shape Vjs: {Vjs.shape}")
        print(f"Shape Eij: {e_ij.shape}")
        print(f"Shape Normal Expanded: {normal_exp.shape}")

    lambdas = (d - normal_exp.bmm(Vis)) / normal_exp.bmm(e_ij)
    lambdas = lambdas.squeeze()  # squeeze out singleton dimensions

    if verbose:
        print(f"Lambdas (num {len(lambdas)}): \n{lambdas}")

    # create an intersection mask; intersection only happens when lambda is in range [0,1]
    i_mask = torch.logical_and(0. <= lambdas, lambdas <= 1.)
    num_intersections = i_mask.to(dtype=torch.int).sum()

    if verbose:
        print(f"Intersection mask ({num_intersections}): {i_mask}")

    intersection_pts = torch.full_like(Vis.squeeze(), torch.inf)  # init as all inf

    # Update entries that are valid intersection points
    intersection_pts = torch.where(i_mask.unsqueeze(1).expand(intersection_pts.shape),
                                   Vis.squeeze() + lambdas.unsqueeze(1) * e_ij.squeeze(),
                                   intersection_pts)

    if verbose:
        print(f"Intersection points (shape {intersection_pts.shape}): {intersection_pts}")

    return lambdas, i_mask, intersection_pts


def volume_of_tetrahedron(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    """
    Calculate the volume of a tetrahedron with vertices a, b, c, d.
    :param a:   Vertex 1 (numpy array)
    :param b:   Vertex 2 (numpy array)
    :param c:   Vertex 3 (numpy array)
    :param d:   Vertex 4 (numpy array)
    :return:    Volume of the tetrahedron
    """
    return torch.abs(torch.dot(a - d, torch.cross(b - d, c - d))) / 6.0

def distance_to_plane(normal: Tensor, D: float, vertex: Tensor) -> Tensor:
    """
    Calculates the perpendicular distance from a point to the plane
    :param normal:
    :param D:
    :param vertex:
    :return:
    """
    if normal.shape == vertex.shape:
        # Only a single vertex was given
        dist = (torch.dot(normal, vertex) + D) / torch.norm(normal)
    else:
        # A set of vertices were given, calculate distances for all vertices
        num_vertices = vertex.shape[0]
        normal = normal.unsqueeze(0)
        normal = normal.expand(num_vertices, 1, -1)
        vertices = vertex.reshape(num_vertices, -1, 1)
        dist = (normal.bmm(vertices) + D) / torch.norm(normal)
    return dist.abs()


def get_edges_from_hull(points: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Given an array of 3D points, calculates the convex hull of these points as well. The return contains:

    * edge_idx: The indices into the points that form the edges of the polyhedron
    * edges: The pairs of points that form the edges of the polyhedron
    * vertices: The points responsible for forming the outer convex hull
    * points: The original points

    :param points:
    :return: edge_idx, edges, vertices, points
    """
    curr_hull = ConvexHull(points)
    edge_idx = set()
    for simplex in curr_hull.simplices:
        # Each simplex has three edges, we add each edge as a sorted tuple to the set
        edge_idx.add(tuple(sorted([simplex[0], simplex[1]])))
        edge_idx.add(tuple(sorted([simplex[1], simplex[2]])))
        edge_idx.add(tuple(sorted([simplex[0], simplex[2]])))

    edge_idx = np.array(list(edge_idx))
    edges = points[edge_idx]  # get edges
    vertices = points[curr_hull.vertices]  # filter out interior points
    return edge_idx, edges, vertices, points

def calculate_volume_above_plane(
        x_L: Tensor,
        x_U: Tensor,
        vertices: Tensor,
        intersection_pts: Tensor,
        i_mask: Tensor,
        normal: Tensor,
        D: Union[Tensor, float],
        lower_bound = False,
        ax: Optional[Axes] = None,
        verbose = False):
    """
    Calculate the volume above the plane inside the cube.
    :param x_L:
    :param x_U:
    :param vertices:            Cube vertex coordinates
    :param intersection_pts:    Points of intersection between cube edges and plane
    :param i_mask:              Intersection mask. True if element in intersection_pts is not infinity.
    :param normal:              Normal vector of the plane
    :param D:                   Plane offset
    :param ax:                  Matplotlib ax subplot
    :param verbose:             Prints debugging logs
    :return:                    Volume above the plane inside the cube
    """

    [A, B, C] = normal

    # Compute distances of each vertex from the plane
    vertices_np = to_numpy(vertices)
    distances = A * vertices[:, 0] + B * vertices[:, 1] + C * vertices[:, 2] + D

    # Separate vertices above the plane
    if lower_bound:
        above_vertices = vertices[distances >= 0]
    else:
        above_vertices = vertices[distances <= 0]

    if i_mask.to(dtype=torch.int).sum() == 0 and len(above_vertices) == 0:
        # Plane is completely out the box, set the volume to be negative
        # total_volumes[i] = dm_lb[i] - threshold[i]
        return -1 * distance_to_plane(normal, D, vertices).max()

    # Combine above vertices with intersection points to form the polyhedron
    if i_mask.to(dtype=torch.int).sum() == 0:
        polyhedron_vertices = above_vertices
    else:
        polyhedron_vertices = torch.vstack([above_vertices, intersection_pts[i_mask]])
    polyhedron_vertices_np = to_numpy(polyhedron_vertices)

    # Perform Delaunay tetrahedralization
    try:
        delaunay = Delaunay(polyhedron_vertices_np)
    except Exception as e:
        # save the hyperplane if desired
        if logger is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            npz_filename = delaunay_logging_file.replace('.log', '.npz')
            normal_np = to_numpy(normal)
            if isinstance(D, float):
                d_np = np.array([D]).reshape(1)
            elif isinstance(D, Tensor):
                d_np = to_numpy(D).reshape(1)
            else:
                raise ValueError(f"Plane offset D is not a float or Tensor, D type: {type(D)}")
            full_np = np.concatenate((normal_np, d_np))
            x_L_np = to_numpy(x_L)
            x_U_np = to_numpy(x_U)
            logger.error(
                f"{timestamp}: Saving to {npz_filename} with timestamp {timestamp}\nHyperplane: {full_np}\nx_L: {x_L_np}\nx_U: {x_U_np}\n{e}")
            if os.path.isfile(npz_filename):
                data = dict(np.load(npz_filename, allow_pickle=True))
            else:
                data = {}
            data[f"hyperplane_{timestamp}"] = full_np
            data[f"x_L_{timestamp}"] = x_L_np
            data[f"x_U_{timestamp}"] = x_U_np
            np.savez(npz_filename, **data)

        # set volume to be 0 to satisfy gradient dependency
        total_volume = 0 * (normal.sum() + D)
        return total_volume

    # Extract the tetrahedrons (each row represents the indices of 4 vertices forming a tetrahedron)
    tetrahedrons = delaunay.simplices
    tetrahedron_vertices = polyhedron_vertices[tetrahedrons, :]

    # Display the tetrahedrons
    if verbose:
        print(f"Shape polyhedron vertices: {polyhedron_vertices.shape}")
        print("Tetrahedrons (vertex indices):")
        print(tetrahedrons)
        print(f"tetrahedron_vertices (shape {tetrahedron_vertices.shape}): \n{to_numpy(tetrahedron_vertices)}")

    total_volume = 0.0
    for i in range(len(tetrahedron_vertices)):
        a = tetrahedron_vertices[i, 0, :]
        b = tetrahedron_vertices[i, 1, :]
        c = tetrahedron_vertices[i, 2, :]
        ref_point = tetrahedron_vertices[i, 3, :]
        curr_volume = volume_of_tetrahedron(ref_point, a, b, c)
        total_volume += curr_volume
        if verbose:
            print(f"i {i + 1} | {curr_volume:.3f} | {total_volume:.3f}")

    if i_mask.to(dtype=torch.int).sum() == 0:
        # Hyperplane does not intersect the polyhedron. The polyhedron's volume at this point is calculated independent
        # of the plane. Include this calculation to track the gradient.
        # FIXME: Redundant calculation to include gradient. Much faster to precompute volumes instead
        total_volume += (0 * (normal.sum() + D)).squeeze()

    if ax is not None:
        # Plot each tetrahedron
        for tet in tetrahedrons:
            # Extract the tetrahedron vertices
            tet_vertices = polyhedron_vertices_np[tet]

            # List of faces, each face is defined by 3 vertices
            faces = [
                [tet_vertices[0], tet_vertices[1], tet_vertices[2]],
                [tet_vertices[0], tet_vertices[1], tet_vertices[3]],
                [tet_vertices[0], tet_vertices[2], tet_vertices[3]],
                [tet_vertices[1], tet_vertices[2], tet_vertices[3]]
            ]

            # Create the 3D polygon collection for the tetrahedron
            poly3d = Poly3DCollection(faces, alpha=0.3, edgecolor='k')

            # Add the collection to the plot
            ax.add_collection3d(poly3d)

        # Set plot limits
        ax.set_xlim([vertices_np[:, 0].min(), vertices_np[:, 0].max()])
        ax.set_ylim([vertices_np[:, 1].min(), vertices_np[:, 1].max()])
        ax.set_zlim([vertices_np[:, 2].min(), vertices_np[:, 2].max()])

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Intersection Tetrahedrons (Total Volume {total_volume:.3f})")

    return total_volume

def cube_intersection_with_plane(
        x_L: Tensor,
        x_U: Tensor,
        normal: Tensor,
        D: float,
        iter: int,
        lower_bound = False,
        show_plots=True,
        verbose=False,
        visualize=False
) -> Tensor:
    """

    :param x_L:         Input lower bounds
    :param x_U:         Input upper bounds
    :param normal:      Plane normal coefficients
    :param D:           Plane offset
    :param verbose:     Print debugging logs
    :param visualize:   Creates the Matplotlib figures
    :return:
    """
    # Create 3D plot
    ax1, ax2 = None, None
    if visualize:
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

    # Define cube vertices and faces
    vertices = get_cube_vertices(x_L, x_U)
    edge_masks = generate_cube_edges()
    edges = vertices[edge_masks, :]
    faces = unit_cube_faces()

    if verbose:
        print(f"Vertices (num {len(vertices)}): \n{to_numpy(vertices)}")
        print(f"Edges (num {len(edges)}): \n{edges}")

    lambdas, intersection_mask, intersection_pts = calculate_intersection(edges, normal, D, verbose)

    if visualize or verbose:
        # intersection occurs with the bounding box and the plane
        intersection_pts_np = to_numpy(intersection_pts[intersection_mask])
        num_i_pts = intersection_pts.to(dtype=torch.int).sum()
        if visualize and num_i_pts > 0:
            ax1.scatter(intersection_pts_np[:, 0], intersection_pts_np[:, 1], intersection_pts_np[:, 2], color='g',
                        label='Intersection Points')
        if verbose and num_i_pts > 0:
            print(f"Intersection Points: \n{intersection_pts_np}")

    intersection_volume = calculate_volume_above_plane(x_L, x_U, vertices, intersection_pts, intersection_mask,
                                                       normal, D, lower_bound, ax2, verbose)
    if verbose:
        print(f"Volume intersection: {to_numpy(intersection_volume)}")
        print(f"Total volume: {(x_U - x_L).prod()}")

    if visualize:
        # Plot the cube
        plot_cube(ax1, to_numpy(vertices), faces)

        # Plot the intersecting plane
        plot_plane(ax1, to_numpy(normal), D, [0, 1], [0, 1])

        # # Plot the intersection of the plane with the cube
        plot_intersection(ax1, to_numpy(vertices), to_numpy(normal), D)

        # Label the axes
        x_L_np = to_numpy(x_L)
        x_U_np = to_numpy(x_U)
        ax1.set_xlim([x_L_np[0], x_U_np[0]])
        ax1.set_ylim([x_L_np[1], x_U_np[1]])
        ax1.set_zlim([x_L_np[2], x_U_np[2]])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax1.set_title(f"Bounding Box with Plane Intersection (Iter. {iter})")

        ax1.legend()
        plt.savefig(out_path + f"frame_{iter}.png")
        if show_plots:
            plt.show()
        else:
            plt.close()

    return intersection_volume

def cube_intersection_with_plane_with_constraints(
        x_L: Tensor,
        x_U: Tensor,
        normal: Tensor,
        D: float,
        iter: int,
        plane_constraints: list[Tuple[Tensor, float]],
        lower_bound = True,
        show_plots=True,
        verbose=False,
        visualize=False
) -> Tensor:
    """

    :param x_L:                 Input lower bounds
    :param x_U:                 Input upper bounds
    :param normal:              Plane normal coefficients
    :param D:                   Plane offset
    :param iter:
    :param plane_constraints:
    :param lower_bound:
    :param show_plots:          Displays and saves the plots, otherwise just saves the plots.
    :param verbose:             Print debugging logs
    :param visualize:           Creates the Matplotlib figures
    :return:
    """
    # Create 3D plot
    ax1, ax2, ax3 = [None] * 3
    if visualize:
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

    # Define cube vertices and faces
    vertices = get_cube_vertices(x_L, x_U)
    edge_masks = generate_cube_edges()
    edges = vertices[edge_masks, :]
    faces = unit_cube_faces()

    # respective numpy vectors
    vertices_np = to_numpy(vertices)

    if verbose:
        print(f"Vertices (num {len(vertices)}): \n{to_numpy(vertices)}")
        print(f"Edges (num {len(edges)}): \n{edges}")

    # initialize the hull to be the bounding box
    hull_ret = get_edges_from_hull(vertices_np)
    [curr_edges_idx_np, curr_edges_np, curr_vertices_np, curr_points_np] = hull_ret
    [_, curr_edges_torch, curr_vertices_torch, _] = [torch.tensor(ret, **set_t) for ret in hull_ret]
    is_constrained = False  # initially assume that no constraint hyperplanes intersect the bbox
    for c_normal, c_D in plane_constraints:
        [A, B, C] = c_normal
        c_ret = calculate_intersection(curr_edges_torch, c_normal, c_D, verbose)
        [_, c_i_mask, c_inter_pts] = c_ret
        if c_i_mask.to(dtype=int).sum() == 0:
            continue

        is_constrained = True
        distances = A * curr_vertices_torch[:, 0] + B * curr_vertices_torch[:, 1] + C * curr_vertices_torch[:, 2] + c_D
        # Separate vertices above the plane
        if not lower_bound:
            above_vertices = curr_vertices_torch[distances >= 0]
        else:
            above_vertices = curr_vertices_torch[distances <= 0]
        total_vertices = to_numpy(torch.vstack((above_vertices, c_inter_pts[c_i_mask])))
        hull_ret = get_edges_from_hull(total_vertices)
        [curr_edges_idx_np, curr_edges_np, curr_vertices_np, curr_points_np] = hull_ret
        [_, curr_edges_torch, curr_vertices_torch, _] = [torch.tensor(ret, **set_t) for ret in hull_ret]

    # display the polyhedron (3D box with planar constraints)
    if visualize or verbose:
        print(f"curr edges ({curr_edges_np.shape}): \n{curr_edges_np}")
        print(f"curr vertices ({curr_vertices_np.shape}): \n{curr_vertices_np}")

        # Plot vertices
        ax3.scatter(curr_vertices_np[:, 0], curr_vertices_np[:, 1], curr_vertices_np[:, 2], c='r', marker='o')

        # Plot edges
        for edge in curr_edges_idx_np:
            i1, i2 = edge
            ax3.plot([curr_points_np[i1, 0], curr_points_np[i2, 0]],
                    [curr_points_np[i1, 1], curr_points_np[i2, 1]],
                    [curr_points_np[i1, 2], curr_points_np[i2, 2]], 'b-')

        # Label vertices
        for i, point in enumerate(curr_vertices_np):
            ax3.text(point[0], point[1], point[2], str(i))

        ax3.set_title(f"Input Polyhedron\nIs Constrained: {is_constrained}")

    # Now calculate the intersection between the polyhedron and the optimizable plane as well as the volume
    lambdas, intersection_mask, intersection_pts = calculate_intersection(curr_edges_torch, normal, D, verbose)
    intersection_volume = calculate_volume_above_plane(x_L, x_U, curr_vertices_torch, intersection_pts,
                                                       intersection_mask, normal, D, lower_bound, ax2, verbose)
    if verbose:
        print(f"Volume intersection: {to_numpy(intersection_volume)}")
        print(f"Total volume: {(x_U - x_L).prod()}")

    if visualize:

        # Plot vertices
        ax1.scatter(curr_vertices_np[:, 0], curr_vertices_np[:, 1], curr_vertices_np[:, 2], marker='o')

        # Plot edges
        for edge in curr_edges_idx_np:
            i1, i2 = edge
            ax1.plot([curr_points_np[i1, 0], curr_points_np[i2, 0]],
                     [curr_points_np[i1, 1], curr_points_np[i2, 1]],
                     [curr_points_np[i1, 2], curr_points_np[i2, 2]], 'b-')

        # Label vertices
        for i, point in enumerate(curr_vertices_np):
            ax1.text(point[0], point[1], point[2], str(i))

        # Plot the cube
        plot_cube(ax1, to_numpy(vertices), faces)

        # Plot the intersecting plane
        plot_plane(ax1, to_numpy(normal), D, [0, 1], [0, 1])

        # # Plot the intersection of the plane with the cube
        plot_intersection(ax1, to_numpy(vertices), to_numpy(normal), D)

        # Label the axes
        x_L_np = to_numpy(x_L)
        x_U_np = to_numpy(x_U)
        ax1.set_xlim([x_L_np[0], x_U_np[0]])
        ax1.set_ylim([x_L_np[1], x_U_np[1]])
        ax1.set_zlim([x_L_np[2], x_U_np[2]])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax1.set_title(f"Bounding Box with Plane Intersection (Iter. {iter})")

        ax1.legend()
        plt.savefig(out_path + f"frame_{iter}.png")
        if show_plots:
            plt.show()
        else:
            plt.close()

    return intersection_volume

### batch algorithms that cannot be simply vectorized using vmap

def batch_cube_intersection_with_plane(
        x_L: Tensor,
        x_U: Tensor,
        normal: Tensor,
        D: Tensor,
        dm_lb: Tensor,
        threshold: Tensor,
        debug_i: Optional[int] = None,
        lower_bound = True
):
    """
    A batch function to calculate the volume above the plane inside the cube. vmap cannot be used on the
    original definition since it has logic pertaining to verbosity and plotting. The actual volume calculation as well
    is not completely parallelizable since it requires the Delaunay algorithm which is CPU based.
    :param x_L:
    :param x_U:
    :param normal:
    :param D:
    :param dm_lb:
    :param threshold:
    :param debug_i:
    :param lower_bound:
    :return:
    """

    # Define cube vertices
    vertices = b_get_cube_vertices(x_L, x_U)
    edge_masks = generate_cube_edges()
    edges = vertices[:, edge_masks, :]

    # get the intersection points formed by the hyperplane and bounding box
    ret = b_calculate_intersection(edges, normal, D)
    [_, intersection_mask, intersection_pts] = ret

    # Using these intersection points, triangulate the volume above the hyperplane and below the bounding box
    # into tetrahedrons and calculate this volume
    total_volumes = batch_calculate_volume_above_plane(x_L, x_U, vertices, intersection_pts, intersection_mask,
                                                       normal, D, dm_lb, threshold, x_L, x_U, debug_i, lower_bound)

    return total_volumes

def batch_calculate_volume_above_plane(
        vertices: Tensor,
        intersection_pts: Tensor,
        i_mask: Tensor,
        normal: Tensor,
        D: Tensor,
        dm_lb: Tensor,
        threshold: Tensor,
        x_L: Tensor,
        x_U: Tensor,
        debug_i: Optional[int] = None,
        lower_bound = True
) -> Tensor:
    """
    Calculate the volume above the plane inside the cube. Due to the dynamic nature of how many
    vertices there may be forming the polyhedron as well as the CPU based Delaunay method, this
    function must iterate over each batch separately.
    :param vertices:            Cube vertex coordinates
    :param intersection_pts:    Points of intersection between cube edges and plane
    :param i_mask:              Intersection mask. True if element in intersection_pts is not infinity.
    :param normal:              Normal vector of the plane
    :param D:                   Plane offset
    :param dm_lb:
    :param threshold:
    :param x_L:                 Only need x_L for debugging in the logger
    :param x_U:                 Only need x_U for debugging in the logger
    :param debug_i:
    :param lower_bound:
    :return:                    Volume above the plane inside the cube
    """
    batches = normal.shape[0]
    total_volumes = torch.zeros((batches,), dtype=normal.dtype, device=normal.device)

    for i in range(batches):
        # Get current batch parameters
        [A, B, C] = normal[i]
        curr_D = D[i].squeeze()  # squeeze offset to be a float
        curr_vertices = vertices[i]
        curr_intersection_pts= intersection_pts[i]
        curr_i_mask = i_mask[i]
        num_inter_points = curr_i_mask.to(dtype=torch.int).sum()  # number of intersection points

        # Compute distances of each vertex from the plane
        distances = A * curr_vertices[:, 0] + B * curr_vertices[:, 1] + C * curr_vertices[:, 2] + curr_D

        # Separate vertices above the plane
        if lower_bound:
            above_vertices = curr_vertices[distances >= 0]
        else:
            above_vertices = curr_vertices[distances <= 0]

        if num_inter_points == 0 and len(above_vertices) == 0:
            # Plane is completely out the box, set the volume to be negative
            # total_volumes[i] = dm_lb[i] - threshold[i]
            total_volumes[i] = -1 * distance_to_plane(normal[i], curr_D, curr_vertices).max()
            continue

        # Combine above vertices with intersection points to form the polyhedron
        if num_inter_points == 0:
            polyhedron_vertices = above_vertices
        else:
            polyhedron_vertices = torch.vstack([above_vertices, curr_intersection_pts[curr_i_mask]])
        polyhedron_vertices_np = to_numpy(polyhedron_vertices)

        # Perform Delaunay tetrahedralization
        try:
            delaunay = Delaunay(polyhedron_vertices_np)
        except Exception as e:
            # set volume to be 0 to satisfy gradient dependency
            total_volumes[i] += 0 * (normal[i].sum() + curr_D)
            # save the hyperplane if desired
            if logger is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                npz_filename = delaunay_logging_file.replace('.log', '.npz')
                normal_np = to_numpy(normal[i])
                d_np = to_numpy(curr_D).reshape(1)
                full_np = np.concatenate((normal_np, d_np))
                x_L_np = to_numpy(x_L[i])
                x_U_np = to_numpy(x_U[i])
                logger.error(f"{timestamp}: Saving to {npz_filename} with timestamp {timestamp}\nHyperplane: {full_np}\nx_L: {x_L_np}\nx_U: {x_U_np}\n{e}")
                if os.path.isfile(npz_filename):
                    data = dict(np.load(npz_filename, allow_pickle=True))
                else:
                    data = {}
                data[f"hyperplane_{timestamp}"] = full_np
                data[f"x_L_{timestamp}"] = x_L_np
                data[f"x_U_{timestamp}"] = x_U_np
                np.savez(npz_filename, **data)
            continue

        # Extract the tetrahedrons (each row represents the indices of 4 vertices forming a tetrahedron)
        tetrahedrons = delaunay.simplices
        tetrahedron_vertices = polyhedron_vertices[tetrahedrons, :]

        # Calculate the total volume formed by all the tetrahredrons
        b_a = tetrahedron_vertices[:, 0, :]
        b_b = tetrahedron_vertices[:, 1, :]
        b_c = tetrahedron_vertices[:, 2, :]
        b_ref = tetrahedron_vertices[:, 3, :]
        total_volumes[i] = b_volume_of_tetrahedron(b_ref, b_a, b_b, b_c).sum(0)

        if num_inter_points == 0:
            # FIXME: Redundant calculation to include gradient. Much faster to precompute volumes instead
            total_volumes[i] += 0 * (normal[i].sum() + curr_D)

    return total_volumes

def batch_cube_intersection_with_plane_with_constraints(
        x_L: Tensor,
        x_U: Tensor,
        normal: Tensor,
        D: Tensor,
        plane_constraints: Tensor,
        lower_bound = True,
        verbose=False,
) -> Tensor:
    """
    Calculate the volume above some hyperplane and polyhedron. The polyhedron is defined by the intersection of the
    plane constraints with input bounding box (bbox). This polyhedron separates the verified and unverified region
    in the bbox. If no constraints are given, acts like `batch_cube_intersection` (which is preferred in this case).
    :param x_L:                 Input lower bounds
    :param x_U:                 Input upper bounds
    :param normal:              Plane normal coefficients
    :param D:                   Plane offset
    :param plane_constraints:   Input constraints that form a polyhedron inside the bbox
                                regions
    :param lower_bound:
    :param verbose:             Print debugging logs
    :return:
    """

    # Define cube vertices
    box_vertices = b_get_cube_vertices(x_L, x_U)
    box_edge_masks = generate_cube_edges()
    box_edges = box_vertices[:, box_edge_masks, :]

    batches = normal.shape[0]
    total_volumes = torch.zeros((batches,), dtype=normal.dtype, device=normal.device)

    for i in range(batches):
        # At this point, we lose all parallelism. Everything must be done sequentially

        # Define cube vertices and faces
        vertices = box_vertices[i]
        edges = box_edges[i]

        # Get batch parameters
        curr_plane_constraints = plane_constraints[i]
        curr_normal = normal[i]
        curr_D = D[i]

        # respective numpy vectors
        vertices_np = to_numpy(vertices)

        if verbose:
            print(f"Vertices (num {len(vertices)}): \n{to_numpy(vertices)}")
            print(f"Edges (num {len(edges)}): \n{edges}")

        # initialize the hull to be the bounding box
        hull_ret = get_edges_from_hull(vertices_np)
        [_, curr_edges_torch, curr_vertices_torch, _] = [torch.tensor(ret, **set_t) for ret in hull_ret]
        for j in range(len(curr_plane_constraints)):
            c_normal = curr_plane_constraints[j, :3].flatten()
            c_D = curr_plane_constraints[j, 3]
            [A, B, C] = c_normal
            c_ret = calculate_intersection(curr_edges_torch, c_normal, c_D, verbose)
            [_, c_i_mask, c_inter_pts] = c_ret
            if c_i_mask.to(dtype=int).sum() == 0:
                continue

            distances = A * curr_vertices_torch[:, 0] + B * curr_vertices_torch[:, 1] + C * curr_vertices_torch[:, 2] + c_D
            # Separate vertices above the plane
            if not lower_bound:
                above_vertices = curr_vertices_torch[distances >= 0]
            else:
                above_vertices = curr_vertices_torch[distances <= 0]
            total_vertices = to_numpy(torch.vstack((above_vertices, c_inter_pts[c_i_mask])))
            hull_ret = get_edges_from_hull(total_vertices)
            [_, curr_edges_torch, curr_vertices_torch, _] = [torch.tensor(ret, **set_t) for ret in hull_ret]

        # Now calculate the intersection between the polyhedron and the optimizable plane as well as the volume
        lambdas, intersection_mask, intersection_pts = calculate_intersection(curr_edges_torch, curr_normal, curr_D, verbose)
        intersection_volume = calculate_volume_above_plane(x_L, x_U, curr_vertices_torch, intersection_pts,
                                                           intersection_mask,
                                                           curr_normal, curr_D, lower_bound, None, verbose)
        total_volumes[i] = intersection_volume
        if verbose:
            print(f"Batches {i}")
            print(f"Volume intersection: {to_numpy(intersection_volume)}")
            print(f"Total volume: {(x_U - x_L).prod()}")

    return total_volumes


### vmap function signatures for simple batching

b_get_cube_vertices = vmap(get_cube_vertices)
b_calculate_intersection = vmap(partial(calculate_intersection, verbose=False))  # verbose must be false for batches
b_volume_of_tetrahedron = vmap(volume_of_tetrahedron)
b_distance_to_plane = vmap(distance_to_plane)

### for help with plotting in autoLiRPA

def get_rows_cols(total_plots, max_cols = 4, max_plots = 12):
    cols = max_cols
    plot_batches = min(total_plots, max_plots)

    # determine plot rows/columns
    if plot_batches <= cols:
        cols = plot_batches
        rows = 1
    else:
        rows = ceil(plot_batches / cols)

    return rows, cols

def fill_plots(
        axs,
        forward_fn,
        x_L: Tensor,
        x_U: Tensor,
        lA: Tensor,
        lbias: Tensor,
        volumes: Tensor,
        plot_batches: int,
        num_unverified,
        num_spec,
        i = 0
):
    faces = unit_cube_faces()

    for b in range(plot_batches):
        ax1 = axs[b]
        vertices = to_numpy(get_cube_vertices(x_L[b], x_U[b]))
        normal = to_numpy(lA[b].squeeze())
        curr_nv_x_L = x_L[b].reshape(1, -1)
        curr_nv_x_U = x_U[b].reshape(1, -1)
        x_L_np = to_numpy(x_L[b])
        x_U_np = to_numpy(x_U[b])

        # sample the region
        with torch.no_grad():
            in_samples = torch.rand(10000, 3) * (curr_nv_x_U - curr_nv_x_L) + curr_nv_x_L
            out_samples = to_numpy(forward_fn(in_samples))
            in_samples = to_numpy(in_samples)[np.logical_and(out_samples >= 0, out_samples <= 1e-4).squeeze()]
            ax1.scatter(in_samples[:, 0], in_samples[:, 1], in_samples[:, 2], label="Implicit Surface")

        D = lbias.reshape(num_unverified, num_spec)[b].item()
        plot_cube(ax1, vertices, faces)
        plot_plane(ax1, normal, D, [x_L_np[0], x_U_np[0]], [x_L_np[1], x_U_np[1]])
        plot_intersection(ax1, vertices, normal, D)
        ax1.set_xlim([x_L_np[0], x_U_np[0]])
        ax1.set_ylim([x_L_np[1], x_U_np[1]])
        ax1.set_zlim([x_L_np[2], x_U_np[2]])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f"Iter. {i}\n(Volume {volumes[b].item():.3e})\nx_L: {x_L_np}\nx_U: {x_U_np}")
        # display the legend with the hyperplane coefficients
        handles, labels = ax1.get_legend_handles_labels()
        handles.extend([mpatches.Patch() for _ in range(4)])
        coeffs = [0 for _ in range(4)]
        for c in range(3):
            curr_char = chr(ord('a') + c)
            labels.append(f"{curr_char}: {normal[c]:.2f}")
            coeffs[c] = normal[c]
        labels.append(f"d: {D:.2f}")
        coeffs[3] = D
        print(f"Coefficients: {coeffs}")
        ax1.legend(handles=handles, labels=labels)

def finalize_plots(figs, main_title: str, titles: list[str], pickle_data = False):
    """
    Finalizes the plots received by autoLiRPA for viewing
    :param figs:
    :param main_title:
    :param titles:
    :param pickle_data:
    :return:
    """
    assert len(figs) == len(titles), "Mismatch in lengths"
    with imageio.get_writer(main_title + '.mp4', fps=5) as writer:
        for fig, title in zip(figs, titles):
            fig.tight_layout()
            fig.savefig(title + ".png")
            if pickle_data:
                with open(title + '.fig.pickle', 'wb') as file:
                    pickle.dump(fig, file)
            plt.close(fig)
            image = imageio.imread(title + '.png')
            writer.append_data(image)

### main functions

def batched_main(batches: int):
    """
    Example usage of the batch_cube_intersection_with_plane method.
    This batch implementation does not support plotting or verbose logs.
    """
    # plane equation <normal, x> + D = 0
    normal = torch.ones((batches, 3), requires_grad=True, **set_t)
    D = torch.full((batches,), -1.5, **set_t)
    lower_bound = True

    # input bounding box
    x_L = torch.zeros((batches, 3), **set_t)
    x_U = torch.ones((batches, 3), **set_t)

    iters = 50
    optimizer = OptimizerMethod.Adam
    optimizer_class = optimizers_config[optimizer]["class"]
    optimizer_params = optimizers_config[optimizer]["params"]
    opt = optimizer_class([normal], maximize=True, **optimizer_params)
    losses = np.zeros((batches, iters))
    for i in range(iters):
        opt.zero_grad()
        loss = batch_cube_intersection_with_plane(x_L, x_U, normal, D, torch.zeros_like(D), torch.zeros_like(D), None, lower_bound)
        losses[:, i] = to_numpy(loss)
        loss = loss.sum()
        loss.backward()
        opt.step()

    print(f"Losses (volume): \n{losses}")

def main():
    """
    Example usage of the 'cube_interesction_with_plane' method.
    """
    # plane equation <normal, x> + D = 0
    normal = torch.tensor([1,1,1], requires_grad=True, **set_t)
    D = -1.5
    lower_bound = True

    # input bounding box
    x_L = torch.zeros(3, **set_t)
    x_U = torch.ones(3, **set_t)

    x_L = torch.tensor([-0.10644531, 0.140625, -0.04345703], **set_t)
    x_U = torch.tensor([-0.10620117, 0.14111328, -0.04296875], **set_t)
    normal = torch.tensor([0.11218592, 0.8509685, -0.03051529], requires_grad=True, **set_t)
    D = -0.10903698

    iters = 50
    optimizer = OptimizerMethod.Adam
    optimizer_class = optimizers_config[optimizer]["class"]
    optimizer_params = optimizers_config[optimizer]["params"]
    opt = optimizer_class([normal], maximize=True, **optimizer_params)
    losses = np.zeros(iters)
    normal_coeffs = np.zeros((iters, len(normal)))
    for i in range(iters):
        opt.zero_grad()
        loss = cube_intersection_with_plane(x_L, x_U, normal, D, i, lower_bound, True, True, True)
        losses[i] = loss.item()
        normal_coeffs[i] = to_numpy(normal)
        loss.backward()
        opt.step()

    print(f"Losses (volume): \n{losses}")

    ## turns plots into a video clip
    with imageio.get_writer(out_path + 'plane_training.mp4', fps=5) as writer:
        for i in range(iters):
            image = imageio.imread(out_path + f"frame_{i}.png")
            writer.append_data(image)

    plt.figure(figsize=(12, 8))
    plt.subplot(2,2,1)
    plt.plot(losses)
    plt.title("Losses Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Volume)")
    plt.grid()
    plt.subplot(2,2,2)
    plt.plot(normal_coeffs[:, 0])
    plt.title("Normal Coefficient a")
    plt.xlabel("Iteration")
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.plot(normal_coeffs[:, 1])
    plt.title("Normal Coefficient b")
    plt.xlabel("Iteration")
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.plot(normal_coeffs[:, 2])
    plt.title("Normal Coefficient c")
    plt.xlabel("Iteration")
    plt.grid()
    plt.savefig(out_path + "training_summary.png")
    plt.show()

def multiplane_main():
    """
        Example usage of the 'cube_intersection_with_plane_with_constraints' method.
    """
    # plane equation <normal, x> + D = 0

    ## single constraint
    c_normal = torch.tensor([1, 1, 1], requires_grad=False, **set_t)
    c_D = -1.5
    plane_constraints: list[tuple[Tensor, float]] = [
        (c_normal, c_D)
    ]

    ## two constraints
    # c_normal_one = torch.tensor([1, 1, 1], requires_grad=False, **set_t)
    # c_D_one = -1.5
    # c_normal_two = torch.tensor([-0.15, 0.16, 0.9], requires_grad=False, **set_t)
    # c_D_two = -0.35
    # plane_constraints: list[tuple[Tensor, float]] = [
    #     (c_normal_one, c_D_one),
    #     (c_normal_two, c_D_two)
    # ]

    # perturb the optimizable plane to prevent nan in gradient ascent
    # normal = torch.tensor([1 + 1e-6, 1 + 1e-5, 1 + 1e-4], requires_grad=True, **set_t)
    # D = -1.5 - 1e-1
    normal = torch.tensor([1, 1, 1], requires_grad=True, **set_t)
    D = -1.5

    lower_bound = True

    # input bounding box
    x_L = torch.zeros(3, **set_t)
    x_U = torch.ones(3, **set_t)

    iters = 50
    optimizer = OptimizerMethod.Adam
    optimizer_class = optimizers_config[optimizer]["class"]
    optimizer_params = optimizers_config[optimizer]["params"]
    opt = optimizer_class([normal], maximize=True, **optimizer_params)
    losses = np.zeros(iters)
    normal_coeffs = np.zeros((iters, len(normal)))
    for i in range(iters):
        opt.zero_grad()
        loss = cube_intersection_with_plane_with_constraints(
            x_L, x_U, normal, D, i, plane_constraints, lower_bound, True, True, True)
        losses[i] = loss.item()
        normal_coeffs[i] = to_numpy(normal)
        loss.backward()
        opt.step()

    print(f"Losses (volume): \n{losses}")

    ## turns plots into a video clip
    with imageio.get_writer(out_path + 'multiplane_constraint_training.mp4', fps=5) as writer:
        for i in range(iters):
            image = imageio.imread(out_path + f"frame_{i}.png")
            writer.append_data(image)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title("Losses Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Volume)")
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(normal_coeffs[:, 0])
    plt.title("Normal Coefficient a")
    plt.xlabel("Iteration")
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.plot(normal_coeffs[:, 1])
    plt.title("Normal Coefficient b")
    plt.xlabel("Iteration")
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.plot(normal_coeffs[:, 2])
    plt.title("Normal Coefficient c")
    plt.xlabel("Iteration")
    plt.grid()
    plt.savefig(out_path + "training_summary.png")
    plt.show()

def batched_multiplane_main(batches: int):
    """
    Example usage of the 'batch_cube_intersection_with_plane_with_constraints' method.
    """
    # plane equation <normal, x> + D = 0

    ## single constraint
    # c_normal = torch.tensor([1, 1, 1], requires_grad=False, **set_t)
    # c_D = -1.5
    # plane_constraints = c_normal.reshape(1, 1, -1).repeat(batches, 1, 1)
    # c_D = torch.ones(batches, 1, 1, **set_t) * c_D
    # # finalize plane constraints to be (batches, num constraints, 3 + 1 where 3 are normal params and 1 for offset)
    # plane_constraints = torch.concatenate((plane_constraints, c_D), dim=2)
    # assert plane_constraints.shape == (batches, 1, 4), "Plane constraints is not formatted correctly"

    ## two constraints
    plane_constraints = torch.tensor(
        [
            [1, 1, 1, -1.5],
            [-0.15, 0.16, 0.9, -0.35]
        ], requires_grad=False, **set_t
    ).repeat(batches, 1, 1)

    # perturb the optimizable plane to prevent nan in gradient ascent
    normal = torch.tensor([1 + 1e-6, 1 + 1e-5, 1 + 1e-4]*batches, **set_t).reshape(batches, 3)
    normal.requires_grad = True  # make optimizable
    D = torch.full((batches,), -1.5 - 1e-1, **set_t)

    lower_bound = True

    # input bounding box
    x_L = torch.zeros((batches, 3), **set_t)
    x_U = torch.ones((batches, 3), **set_t)

    iters = 50
    optimizer = OptimizerMethod.Adam
    optimizer_class = optimizers_config[optimizer]["class"]
    optimizer_params = optimizers_config[optimizer]["params"]
    opt = optimizer_class([normal], maximize=True, **optimizer_params)
    losses = np.zeros((batches, iters))
    for i in range(iters):
        opt.zero_grad()
        loss = batch_cube_intersection_with_plane_with_constraints(
            x_L, x_U, normal, D, plane_constraints, lower_bound, False)
        losses[:, i] = to_numpy(loss)
        loss = loss.sum()
        loss.backward()
        opt.step()

    print(f"Losses (volume): \n{losses}")

def misc_main():
    x_L = torch.tensor([-0.10644531, 0.140625, -0.04345703], **set_t).reshape(1, -1)
    x_U = torch.tensor([-0.10620117, 0.14111328, -0.04296875], **set_t).reshape(1, -1)
    normal = torch.tensor([[ 0.11218592, 0.8509685, -0.03051529]], **set_t).reshape(1, -1)
    D = torch.tensor([-0.10903698], **set_t)
    ###
    # x_L = torch.tensor([-0.10644531, 0.140625, -0.04345703]).reshape(1, -1)
    # x_U = torch.tensor([-0.10620117, 0.14111328, -0.04296875]).reshape(1, -1)
    # normal = torch.tensor([[0.11218592, 0.8509685, -0.03051529]]).reshape(1, -1)
    # D = torch.tensor([-0.10903698])
    ###
    # x_L = torch.tensor([-0.10644531, 0.140625, -0.04345703], device=set_t['device']).reshape(1, -1)
    # x_U = torch.tensor([-0.10620117, 0.14111328, -0.04296875], device=set_t['device']).reshape(1, -1)
    # normal = torch.tensor([[0.11218592, 0.8509685, -0.03051529]], device=set_t['device']).reshape(1, -1)
    # D = torch.tensor([-0.10903698], device=set_t['device'])
    ###
    batches = normal.shape[0]
    tensors = [x_L, x_U, normal, D]
    for i, tensor in enumerate(tensors):
        print("Initial Tensor")
        print(f"i {i} | dtype: {tensor.dtype}\n{to_numpy(tensor)}")

    x_L_64t = x_L.to(dtype=torch.float64)
    x_U_64t = x_U.to(dtype=torch.float64)

    print(f"x_L_64t: {to_numpy(x_L_64t)}")
    print(f"x_U_64t: {to_numpy(x_U_64t)}")

    vertices = b_get_cube_vertices(x_L_64t, x_U_64t)
    edge_masks = generate_cube_edges()
    edges = vertices[:, edge_masks, :]

    # convert to float64 for stable numerical precision in small bounding boxes
    vertices_64t = vertices.to(dtype=torch.float64)
    edges_64t = edges.to(dtype=torch.float64)
    normal_64t = normal.to(dtype=torch.float64)
    D_64t = D.to(dtype=torch.float64)

    # get the intersection points formed by the hyperplane and bounding box
    ret = b_calculate_intersection(edges, normal_64t, D_64t)
    [lambdas, intersection_mask, intersection_pts] = ret
    num_intersections = intersection_mask.to(dtype=int).sum()

    for i, tensor in enumerate(ret):
        print("Calculate Intersection Return")
        print(f"i {i} | dtype: {tensor.dtype}")
    print(f"Number of intersections: {num_intersections}")
    print(f"Lambdas ({lambdas.shape}): \n{lambdas}")
    print(f"Intersection Mask ({intersection_mask.shape}): \n{intersection_mask}")
    print(f"Intersection Pts ({intersection_pts.shape}): \n{intersection_pts}")
    print(
        f"Filtered Intesrctions Pts ({intersection_pts[intersection_mask].shape}): \n{intersection_pts[intersection_mask]}")
    [lambdas, intersection_mask, intersection_pts] = [r.to(dtype=set_t['dtype']) for r in ret]
    ret = [lambdas, intersection_mask, intersection_pts]
    intersection_mask = intersection_mask.to(dtype=bool)

    for i, tensor in enumerate(ret):
        print("Calculate Intersection Return")
        print(f"i {i} | dtype: {tensor.dtype}")

    num_intersections = intersection_mask.to(dtype=int).sum()
    print(f"Number of intersections: {num_intersections}")
    print(f"Lambdas ({lambdas.shape}): \n{lambdas}")
    print(f"Intersection Mask ({intersection_mask.shape}): \n{intersection_mask}")
    print(f"Intersection Pts ({intersection_pts.shape}): \n{intersection_pts}")
    print(f"Filtered Intesrctions Pts ({intersection_pts[intersection_mask].shape}): \n{intersection_pts[intersection_mask]}")


if __name__ == '__main__':
    num_batches = 3  # only used for batch methods
    method = VolumeCalculationMethod.MultiPlaneSingleVolume  # method to run

    print(f"Running program: {method.name}")

    if method == VolumeCalculationMethod.SinglePlaneSingleVolume:
        main()
    elif method == VolumeCalculationMethod.SinglePlaneBatchVolume:
        batched_main(num_batches)
    elif method == VolumeCalculationMethod.MultiPlaneSingleVolume:
        multiplane_main()
    elif method == VolumeCalculationMethod.MultiPlaneBatchVolume:
        batched_multiplane_main(num_batches)
    elif method == VolumeCalculationMethod.MiscMain:
        misc_main()
    else:
        raise Exception("Unknown volume calculation method")