import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
import numpy as np

wp.init()

def generate_sphere_sdf(resolution=64, radius=0.5):
    x = np.linspace(-1, 1, resolution)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    return np.sqrt(xx**2 + yy**2 + zz**2) - radius

# Generate SDF
sdf_grid = generate_sphere_sdf()

# Create Warp volume directly
volume = wp.Volume.load_from_numpy(
    sdf_grid,
    # bounds_min=(-1, -1, -1),
    # bounds_max=(1, 1, 1),
    voxel_size=2.0/(64-1)
)

# Render
stage = wp.sim.render.Simulation()
stage.add_sdf(volume, name="sphere_sdf")
stage.run()
