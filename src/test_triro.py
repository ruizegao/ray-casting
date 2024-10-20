import torch
import trimesh
import matplotlib.pyplot as plt
from triro.ray.ray_optix import RayMeshIntersector

# creating mesh and intersector
mesh = trimesh.creation.icosphere()
intersector = RayMeshIntersector(mesh)

# generating rays
y, x = torch.meshgrid([torch.linspace(1, -1, 800),
                       torch.linspace(-1, 1, 800)], indexing='ij')
z = -torch.ones_like(x)
ray_directions = torch.stack([x, y, z], dim=-1).cuda()
ray_origins = torch.Tensor([0, 0, 3]).cuda().broadcast_to(ray_directions.shape)

# OptiX, Launch!
hit, front, ray_idx, tri_idx, location, uv = intersector.intersects_closest(
    ray_origins, ray_directions, stream_compaction=True
)

# drawing result
locs = torch.zeros((800, 800, 3)).cuda()
locs[hit] = location
plt.imshow(locs.cpu())
plt.show()
