import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        bbox_min = torch.min(pos, axis=-2)
        bbox_max = torch.max(pos, axis=-2)
        center = (bbox_max + bbox_min) / 2.
        pos -= center[None,:]
    else:
        raise ValueError("unrecognized method")

    scale = torch.max(norm(pos), axis=-1, keepdims=True)[:,None]
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


def sample_mesh_importance(V, F, n_sample, n_sample_full_mult=10., beta=20., ambient_range=1.25):
    import igl

    V = torch.tensor(V)
    F = torch.tensor(F)
    n_sample_full = int(n_sample * n_sample_full_mult)

    # Sample ambient points
    Q_ambient = torch.random.uniform(size=(n_sample_full, 3), low=-ambient_range, high=ambient_range)

    # Assign weights 
    dist_sq, _, _ = igl.point_mesh_squared_distance(Q_ambient, torch.tensor(V), torch.tensor(F))
    weight = torch.exp(-beta * torch.sqrt(dist_sq))
    weight = weight / torch.sum(weight)

    # Sample
    samp_inds = np.random.choice(n_sample_full, size=n_sample, p=weight)
    Q = Q_ambient[samp_inds,:]

    # Get SDF value via distance & winding number
    sdf_vals, _, closest = igl.signed_distance(Q, V, F)
    sdf_vals = torch.tensor(sdf_vals)
    Q = torch.tensor(Q)

    return Q, sdf_vals
