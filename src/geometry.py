import numpy as np
import torch

from typing import Union, Tuple, Optional
from torch import Tensor
from numpy import ndarray
import igl

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
to_numpy = lambda x: x.detach().cpu().numpy()

def split_generator(generator, num_splits=2):
    return [torch.Generator(device=device).manual_seed(generator.initial_seed() + i) for i in range(num_splits)]

def norm(x):
    return torch.linalg.norm(x, axis=-1)

def norm2(x):
    return torch.inner(x,x)

def normalize(x):
    return x / norm(x)

def orthogonal_dir(x, remove_dir):
    # take a vector x, remove any component in the direction of vector remove_dir, and return unit x
    remove_dir = normalize(remove_dir)
    x = x - torch.dot(x, remove_dir) * remove_dir
    return normalize(x)

def dot(x,y):
    return torch.sum(x*y, axis=-1)

def normalize_positions(pos, method='bbox'):
    # center and unit-scale positions in to the [-1,1] cube

    if method == 'mean':
        # center using the average point position
        pos = pos - torch.mean(pos, axis=-2, keepdims=True)
    elif method == 'bbox': 
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, axis=-2).values
        bbox_max = torch.max(pos, axis=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center[None,:]
    else:
        raise ValueError("unrecognized method")

    scale = torch.max(norm(pos), axis=-1, keepdims=True).values[:,None]
    pos = pos / scale
    return pos

def sample_mesh_sdf(V, F, n_sample, surface_frac=0.5, surface_perturb_sigma=0.01, ambient_expand=1.25):
    import igl
    '''
    NOTE: Assumes input is scaled to lie in [-1,1] cube
    NOTE: RNG is handled internally, in part by an external library (libigl). Has none of the usual JAX RNG properties, may or may not yield same results, etc.
    '''

    n_surface = int(n_sample * surface_frac)
    n_ambient = n_sample - n_surface

    # Compute a bounding box for the mesh
    bbox_min = torch.tensor([-1,-1,-1])
    bbox_max = torch.tensor([1,1,1])
    center = 0.5*(bbox_max + bbox_min)

    # Sample ambient points
    key = torch.manual_seed(0)
    key, subkey = split_generator(key)

    Q_ambient = 2 * ambient_expand * torch.rand((n_ambient, 3), generator=torch.Generator().manual_seed(0)) - ambient_expand
    # Sample surface points
    sample_b, sample_f = igl.random_points_on_mesh(n_surface, torch.tensor(V), torch.tensor(F))
    face_verts = V[F[sample_f], :]
    raw_samples = torch.sum(sample_b[..., None] * face_verts, dim=1)
    raw_samples = torch.tensor(raw_samples)

    # add noise to surface points
    key, subkey = split_generator(key)
    offsets = torch.randn((n_surface, 3), generator=subkey) * surface_perturb_sigma
    Q_surface = raw_samples + offsets

    # Combine and shuffle
    Q = torch.vstack((Q_ambient, Q_surface))
    key, subkey = split_generator(key)
    Q = Q[torch.randperm(Q.size(0), generator=subkey)]

    # Get SDF value via distance & winding number
    sdf_vals, _, closest = igl.signed_distance(torch.tensor(Q), torch.tensor(V), torch.tensor(F))
    sdf_vals = torch.tensor(sdf_vals)

    return Q, sdf_vals


def sample_mesh_importance(V: Union[Tensor, ndarray], F: Union[Tensor, ndarray],
                           n_sample, n_sample_full_mult=10., beta=20., ambient_range=1.25,
                           sdf_max=0.4, show_surface=False
                           ) -> Tuple[ndarray, ndarray]:
    import igl

    if isinstance(V, Tensor):
        V_torch = V
        V_np = to_numpy(V_torch)
    else:
        V_np = V
        V_torch = torch.from_numpy(V_np)

    if isinstance(F, Tensor):
        F_torch = F
        F_np = to_numpy(F_torch)
    else:
        F_np = F
        F_torch = torch.from_numpy(F_np)

    n_sample_full = int(n_sample * n_sample_full_mult)

    # Sample ambient points
    # Q_ambient = torch.random.uniform(size=(n_sample_full, 3), low=-ambient_range, high=ambient_range)
    Q_ambient = torch.empty(n_sample_full, 3, dtype=V_torch.dtype).uniform_(-ambient_range, ambient_range)
    Q_ambient_np = to_numpy(Q_ambient)

    # Assign weights
    dist_sq, _, _ = igl.point_mesh_squared_distance(Q_ambient_np, V_np, F_np)
    dist_sq_torch = torch.from_numpy(dist_sq)
    weight = torch.exp(-beta * torch.sqrt(dist_sq_torch))
    weight = weight / torch.sum(weight)
    weight_np = to_numpy(weight)

    # Sample
    samp_inds = np.random.choice(n_sample_full, size=n_sample, p=weight_np)
    Q = Q_ambient[samp_inds,:]
    Q_np = to_numpy(Q)

    # Get SDF value via distance & winding number
    sdf_vals, _, closest = igl.signed_distance(Q_np, V_np, F_np)
    sdf_vals = torch.tensor(sdf_vals, dtype=V_torch.dtype)

    # convert to numpy arrays
    Q_np = to_numpy(Q)
    sdf_vals_np = to_numpy(sdf_vals)
    print(f"Smallest sdf val: {sdf_vals_np.min()}, Largest sdf val: {sdf_vals_np.max()}")
    sdf_vals_np = np.clip(sdf_vals_np, -sdf_max, sdf_max)

    if show_surface:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        filt_idx = sdf_vals_np < 0.04
        samples_np = Q_np[filt_idx]

        # Plot the points
        ax.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], c='b', marker='o')

        # Set labels for clarity
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()

    return Q_np, sdf_vals_np

def sample_point_on_triangle(v0, v1, v2):
    # Sample uniformly in the triangle using barycentric coordinates
    r1, r2 = np.random.rand(2)
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = r2 * sqrt_r1
    return (1 - sqrt_r1)*v0 + (sqrt_r1*(1-r2))*v1 + (sqrt_r1*r2)*v2
    # return (1 - u - v) * v0 + u * v1 + v * v2

def sample_points_on_mesh(V, F, num_samples, show_surface) -> ndarray:
    areas = igl.doublearea(V, F) / 2  # Half of double area gives true area for triangles
    cumulative_areas = np.cumsum(areas)
    total_area = cumulative_areas[-1]
    samples = []
    for _ in range(num_samples):
        # Select a triangle based on area
        sample_area = np.random.rand() * total_area
        triangle_idx = np.searchsorted(cumulative_areas, sample_area)

        # Get vertices of the selected triangle
        v0 = V[F[triangle_idx, 0]]
        v1 = V[F[triangle_idx, 1]]
        v2 = V[F[triangle_idx, 2]]

        # Sample a point on the triangle
        sample_point = sample_point_on_triangle(v0, v1, v2)
        samples.append(sample_point)

    samples_np = np.array(samples)

    if show_surface:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the points
        ax.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], c='b', marker='o')

        # Set labels for clarity
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()

    return samples_np

def sample_221(V: Union[Tensor, ndarray], F: Union[Tensor, ndarray], n_sample, ambient_range=1.,
               sdf_max=0.4, show_surface = False):

    if isinstance(V, Tensor):
        V_torch = V
        V_np = to_numpy(V_torch)
    else:
        V_np = V
        V_torch = torch.from_numpy(V_np)

    if isinstance(F, Tensor):
        F_torch = F
        F_np = to_numpy(F_torch)
    else:
        F_np = F
        F_torch = torch.from_numpy(F_np)

    num_uni = int(n_sample / 5)
    num_on_surf = num_near_surf = 2 * num_uni
    Q_on_surf = sample_points_on_mesh(V_np, F_np, num_on_surf, show_surface)
    Q_near_surf = np.clip(Q_on_surf + np.random.normal(0, 0.02, Q_on_surf.shape), -ambient_range, ambient_range)
    Q_uni = torch.empty(num_uni, 3, dtype=V_torch.dtype).uniform_(-ambient_range, ambient_range)

    Q = np.concatenate((Q_on_surf, Q_near_surf, Q_uni))
    # np.random.shuffle(Q)
    sdf_val, _, _ = igl.signed_distance(Q, V_np, F_np)
    print(f"Smallest sdf val: {sdf_val.min()}, Largest sdf val: {sdf_val.max()}")
    sdf_val = np.clip(sdf_val, -sdf_max, sdf_max)


    return Q, sdf_val
