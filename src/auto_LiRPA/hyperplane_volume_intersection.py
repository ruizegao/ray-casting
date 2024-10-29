import numpy as np
import torch
from torch import Tensor
from torch import vmap
from functools import partial
from numpy import ndarray
from scipy.spatial import Delaunay as SciDelaunay
from scipy.spatial import ConvexHull
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

set_t = {
    "dtype": torch.float32,
    "device": torch.device("cuda"),
}
out_path = "/home/jorgejc2/Documents/Research/ray-casting/plane_training/"

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
    normal_exp = normal.reshape(1, 1, -1).expand(batches, -1, -1)
    if verbose:
        print(f"Shape Vis: {Vis.shape}")
        print(f"Shape Vjs: {Vjs.shape}")
        print(f"Shape Eij: {e_ij.shape}")
        print(f"Shape Normal Expanded: {normal_exp.shape}")

    lambdas = -(D + normal_exp.bmm(Vis)) / normal_exp.bmm(e_ij)
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


def calculate_volume_above_plane(
        vertices: Tensor,
        intersection_pts: Tensor,
        i_mask: Tensor,
        normal: Tensor,
        D: float,
        lower_bound = False,
        ax: Optional[Axes] = None,
        verbose = False):
    """
    Calculate the volume above the plane inside the cube.
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
    delaunay = SciDelaunay(polyhedron_vertices_np)

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
        # FIXME: Redundant calculation to include gradient. Much faster to precompute volumes instead
        total_volume += 0 * (normal.sum() + D)

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

    intersection_volume = calculate_volume_above_plane(vertices, intersection_pts, intersection_mask,
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
    total_volumes = batch_calculate_volume_above_plane(vertices, intersection_pts, intersection_mask,
                                                       normal, D, dm_lb, threshold, debug_i, lower_bound)

    return total_volumes

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

def batch_calculate_volume_above_plane(
        vertices: Tensor,
        intersection_pts: Tensor,
        i_mask: Tensor,
        normal: Tensor,
        D: Tensor,
        dm_lb: Tensor,
        threshold: Tensor,
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
        delaunay = SciDelaunay(polyhedron_vertices_np)

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


### vmap function signatures for simple batching

b_get_cube_vertices = vmap(get_cube_vertices)
b_calculate_intersection = vmap(partial(calculate_intersection, verbose=False))  # verbose must be false for batches
b_volume_of_tetrahedron = vmap(volume_of_tetrahedron)
b_distance_to_plane = vmap(distance_to_plane)

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
    lr = 1e-1
    opt = torch.optim.Adam([normal], lr=lr, maximize=True)
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
    lower_bound = False

    # input bounding box
    x_L = torch.zeros(3, **set_t)
    x_U = torch.ones(3, **set_t)

    iters = 50
    lr = 1e-1
    opt = torch.optim.Adam([normal], lr=lr, maximize=True)
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


if __name__ == '__main__':
    num_batches = 1
    if num_batches == 1:
        main()
    elif num_batches > 1:
        batched_main(num_batches)
    else:
        raise Exception("Please enter a non-negative number of batches.")