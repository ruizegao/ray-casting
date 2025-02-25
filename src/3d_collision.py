import pybullet as p
import pybullet_data
import numpy as np

# Initialize PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load default textures

# Define vertices and faces for a cube
cube_size = 0.5
cube_vertices = [
    [-cube_size / 2, -cube_size / 2, -cube_size / 2],
    [cube_size / 2, -cube_size / 2, -cube_size / 2],
    [cube_size / 2, cube_size / 2, -cube_size / 2],
    [-cube_size / 2, cube_size / 2, -cube_size / 2],
    [-cube_size / 2, -cube_size / 2, cube_size / 2],
    [cube_size / 2, -cube_size / 2, cube_size / 2],
    [cube_size / 2, cube_size / 2, cube_size / 2],
    [-cube_size / 2, cube_size / 2, cube_size / 2]
]
# Define faces for the cube
cube_faces = [
    [0, 1, 2, 3],  # Bottom face
    [4, 5, 6, 7],  # Top face
    [0, 1, 5, 4],  # Front face
    [2, 3, 7, 6],  # Back face
    [0, 3, 7, 4],  # Left face
    [1, 2, 6, 5]   # Right face
]

# Create visual shape for a cube with transparency
def create_cube(position):
    return p.createVisualShape(p.GEOM_MESH, vertices=cube_vertices, indices=np.array(cube_faces).flatten(), rgbaColor=[0, 1, 0, 0.3])

# Create collision shape for a cube
def create_collision_cube():
    return p.createCollisionShape(p.GEOM_MESH, vertices=cube_vertices, indices=np.array(cube_faces).flatten())

# Create two cubes slightly overlapping
# Cube 1 at (0, 0, 0)
visual_shape1 = create_cube([0, 0, 0])
collision_shape1 = create_collision_cube()

# Cube 2 at (0.2, 0, 0) (slightly offset to guarantee collision)
visual_shape2 = create_cube([0.2, 0, 0])
collision_shape2 = create_collision_cube()

# Create rigid bodies for both cubes
body1 = p.createMultiBody(baseCollisionShapeIndex=collision_shape1, baseVisualShapeIndex=visual_shape1, basePosition=[0, 0, 0])
body2 = p.createMultiBody(baseCollisionShapeIndex=collision_shape2, baseVisualShapeIndex=visual_shape2, basePosition=[0.2, 0.2, 0])

# Set simulation parameters
p.setGravity(0, -9.81, 0)
p.stepSimulation()

# Function to compute collision points and area
def get_collision_info():
    contact_points = p.getContactPoints(body1, body2)
    collision_points = []
    total_collision_area = 0

    for point in contact_points:
        print("viewing point: ", point)
        contact_pos = np.array(point[6])  # Contact position
        collision_points.append(contact_pos)
        penetration_depth = abs(point[8])
        total_collision_area += penetration_depth  # Approximate collision area

    return np.array(collision_points), total_collision_area

# Retrieve collision points and area
collision_points, collision_area = get_collision_info()
print("collision points: ", collision_points)
print(f"Collision Area Estimate: {collision_area}")

# Visualize collision points as small red spheres
for pt in collision_points:
    p.createMultiBody(
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1]),
        basePosition=pt
    )

# Keep the simulation running until user manually disconnects
print("Press 'q' in the terminal and hit Enter to quit.")
while True:
    p.stepSimulation()
    user_input = input()
    if user_input.lower() == "q":
        break

# Disconnect PyBullet
p.disconnect()
