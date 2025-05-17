import torch
import numpy as np
from tqdm import tqdm
import argparse
import implicit_mlp_utils
from skimage import measure
import trimesh

from warp.warp import print


def sample_sdf_on_grid(sdf_model, resolution=256, bounds=[-1.0, 1.0], batch_size=65536, device='cuda'):
    """
    Samples an SDF neural network on a 3D grid and returns a (resolution x resolution x resolution) NumPy array.

    Args:
        sdf_model: PyTorch model, input: (B, 3), output: (B, 1)
        resolution: Grid resolution (int)
        bounds: [min, max] coordinates (assumed the same for x, y, z)
        batch_size: Number of points per inference batch
        device: Device to run model on

    Returns:
        sdf_grid: np.ndarray of shape (resolution, resolution, resolution)
    """
    sdf_model.eval().to(device)
    with torch.no_grad():
        # voxel_size = (bounds[1] - bounds[0]) / resolution
        lin = torch.linspace(bounds[0], bounds[1], resolution+1)
        grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)  # (res^3, 3)
        print(coords.max())
        print("actual distance between ticks", (lin[1] - lin[0]).item())
        print(lin[1])
        print(len(lin))
        sdf_values = []
        for i in tqdm(range(0, coords.shape[0], batch_size), desc="Sampling SDF"):
            batch = coords[i:i + batch_size]
            sdf_batch = sdf_model(batch).squeeze(-1)
            sdf_values.append(sdf_batch.cpu())

        sdf_tensor = torch.cat(sdf_values).reshape(resolution+1, resolution+1, resolution+1)
        mins = torch.tensor([bounds[0], bounds[0], bounds[0]], device=torch.device('cpu'))
        voxel_size = (bounds[1] - bounds[0]) / resolution
        bg_value = sdf_tensor.numpy().max().item() + 3. * voxel_size
        return sdf_tensor.numpy().astype(np.float32), mins.numpy(), voxel_size, bg_value, coords.detach().cpu().numpy()

def extract_mesh(sdf_grid, mins, voxel_size, level=0.0):
    # Perform marching cubes
    verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=level, spacing=(voxel_size,) * 3)
    # Translate verts to world space
    verts += mins
    return verts, faces


def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--res", type=int, default=256)
    # parser.add_argument("--mesh_output", type=str, default="mesh.ply")  # Save mesh as .ply

    # Parse arguments
    args = parser.parse_args()


    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode='crown', **{})
    print("function loaded")
    sdf, mins, voxel_size, bg_value, coords = sample_sdf_on_grid(implicit_func.torch_model, resolution=args.res, bounds=[-1., 1.], device='cuda')
    print("samples generated")
    print(sdf.shape)
    print(mins)
    print(voxel_size)
    print(bg_value)
    np.savez_compressed(args.output, sdf=sdf, mins=mins, voxel_size=voxel_size, bg_value=bg_value, coords=coords)

    # verts, faces = extract_mesh(sdf, mins, voxel_size)
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # # mesh.show()
    # mesh.export(args.mesh_output)
    # print(f"Mesh saved to {args.mesh_output}")

if __name__ == '__main__':
    main()