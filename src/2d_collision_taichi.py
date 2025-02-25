import taichi as ti
import torch
import time

# Initialize Taichi
ti.init(arch=ti.cuda)

# Circle properties
circle_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
circle_radius = ti.field(dtype=ti.f32, shape=())

# Dummy SDF Model
class SDFModel(torch.nn.Module):
    def forward(self, x):
        return torch.norm(x, dim=-1, keepdim=True) - 0.5  # A simple sphere SDF

sdf_model = SDFModel()

def check_sdf_collision(circle_pos, radius, model):
    pos_tensor = torch.tensor(circle_pos.to_numpy(), dtype=torch.float32).unsqueeze(0)
    sdf_value = model(pos_tensor).item()
    return sdf_value <= radius

# Mesh Representation (list of line segments)
mesh_edges = ti.Vector.field(4, dtype=ti.f32, shape=5)  # Each edge is (x1, y1, x2, y2)

@ti.kernel
def init_mesh():
    for i in range(5):
        mesh_edges[i] = ti.Vector([i * 0.2, 0.2, i * 0.2 + 0.1, 0.3])

@ti.func
def point_line_distance(p, a, b):
    ap = p - a
    ab = b - a
    proj = ap.dot(ab) / ab.dot(ab)
    proj = ti.max(0.0, ti.min(1.0, proj))
    closest_point = a + proj * ab
    return (p - closest_point).norm()

@ti.kernel
def check_mesh_collision() -> ti.i32:
    p = circle_pos[None]
    for i in range(5):
        a = ti.Vector([mesh_edges[i][0], mesh_edges[i][1]])
        b = ti.Vector([mesh_edges[i][2], mesh_edges[i][3]])
        if point_line_distance(p, a, b) <= circle_radius[None]:
            return 1  # Collision detected
    return 0  # No collision

# Set up test parameters
circle_pos[None] = [0.1, 0.2]
circle_radius[None] = 0.1
init_mesh()

# Measure efficiency
start_time = time.time()
sdf_collision = check_sdf_collision(circle_pos, circle_radius[None], sdf_model)
sdf_time = time.time() - start_time

start_time = time.time()
mesh_collision = check_mesh_collision()
mesh_time = time.time() - start_time

# Print results
print(f"SDF Collision: {sdf_collision}, Time: {sdf_time:.6f}s")
print(f"Mesh Collision: {mesh_collision}, Time: {mesh_time:.6f}s")
