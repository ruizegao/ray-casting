import argparse
import numpy as np
import implicit_mlp_utils
import torch
from skimage import measure
import trimesh


def create_chunked_grid(grid_size, chunk_size, device='cuda'):
    # Create 1D coordinates
    coords = torch.linspace(-1, 1, grid_size, device=device)

    # Compute number of chunks per axis
    chunks_per_axis = (grid_size + chunk_size - 1) // chunk_size

    # Iterate over chunks along each axis
    for i in range(chunks_per_axis):
        for j in range(chunks_per_axis):
            for k in range(chunks_per_axis):
                # Get start and end indices for each chunk
                x_start, x_end = i * chunk_size, min((i + 1) * chunk_size, grid_size)
                y_start, y_end = j * chunk_size, min((j + 1) * chunk_size, grid_size)
                z_start, z_end = k * chunk_size, min((k + 1) * chunk_size, grid_size)
                min_corner = torch.tensor([x_start, y_start, z_start], device=device)

                # Extract chunk-specific coordinates
                x_chunk = coords[x_start:x_end]
                y_chunk = coords[y_start:y_end]
                z_chunk = coords[z_start:z_end]
                # Generate the chunk of the grid correctly
                # chunk_grid = torch.stack(torch.meshgrid(x_chunk, y_chunk, z_chunk, indexing='ij'), dim=-1)
                # chunk_grid = chunk_grid.reshape(-1, 3)  # Flatten the grid
                grid_x, grid_y, grid_z = torch.meshgrid(x_chunk, y_chunk, z_chunk, indexing='ij')
                chunk_grid = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=-1)

                yield chunk_grid, min_corner

def combine_sdf_chunks(sdf_vals, grid_size, chunk_size):
    # Initialize global SDF array
    sdf_full = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # Iterate over chunks and place them in the full grid
    chunk_idx = 0
    for i in range(0, grid_size, chunk_size):
        for j in range(0, grid_size, chunk_size):
            for k in range(0, grid_size, chunk_size):
                # Calculate the indices in the full grid
                x_start, x_end = i, min(i + chunk_size, grid_size)
                y_start, y_end = j, min(j + chunk_size, grid_size)
                z_start, z_end = k, min(k + chunk_size, grid_size)

                # Place the chunk in the full grid
                sdf_full[x_start:x_end, y_start:y_end, z_start:z_end] = sdf_vals[chunk_idx]
                chunk_idx += 1

    return sdf_full

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--grid_res", type=int, default=2**9)
    parser.add_argument("--eps_d", type=float, default=0.002)
    parser.add_argument("--eps_e", type=float, default=0.002)
    # Parse arguments
    args = parser.parse_args()

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode='crown', **{})

    grid_res = args.grid_res
    output = args.output
    eps_d = args.eps_d
    eps_e = args.eps_e
    delta = 2. / grid_res
    print(delta)
    sdf_vals = []
    chunk_size = grid_res // 16
    for chunk, bbox_min in create_chunked_grid(grid_res, chunk_size):
        sdf_val = implicit_func.torch_forward(chunk).detach().cpu().numpy()
        sdf_vals.append(sdf_val.reshape(chunk_size, chunk_size, chunk_size))
    sdf_vals = combine_sdf_chunks(sdf_vals, grid_res, chunk_size)
    bbox_min = np.array([-1., -1., -1.])
    verts_outer, faces_outer, _, _ = measure.marching_cubes(sdf_vals, level=eps_d, spacing=(delta, delta, delta))
    verts_inner, faces_inner, _, _ = measure.marching_cubes(sdf_vals, level=-eps_e, spacing=(delta, delta, delta))
    verts_outer = verts_outer + bbox_min[None,:]
    verts_inner = verts_inner + bbox_min[None,:]
    verts = np.concatenate((verts_outer, verts_inner), axis=0)
    faces = np.concatenate((faces_outer, faces_inner+len(verts_outer)), axis=0)
    mesh_dilation = trimesh.Trimesh(verts_outer, faces_outer)
    mesh_erosion = trimesh.Trimesh(verts_inner, faces_inner)
    mesh_de = trimesh.Trimesh(verts, faces)
    verts_0lvl, faces_0lvl, _, _ = measure.marching_cubes(sdf_vals, level=0., spacing=(delta, delta, delta))
    verts_0lvl = verts_0lvl + bbox_min[None, :]
    mesh_0lvl = trimesh.Trimesh(verts_0lvl, faces_0lvl)
    if output:
        # mesh_dilation.export(output[:-4]+"_dilation.obj")
        mesh_de.export(output[:-4] + "_de.obj")
        # mesh_0lvl.export(output[:-4] + "_0lvl.obj")
        # print(len(mesh_0lvl.vertices), len(mesh_0lvl.faces))
        # mesh_erosion.export(output[:-4] + "_erosion.obj")

if __name__ == '__main__':
    main()