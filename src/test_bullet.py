import pybullet as p
import pybullet_data

# Start PyBullet in GUI mode
p.connect(p.GUI)

# Set PyBullet's resource path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create two cube collision shapes
cube_size = 0.5  # Half-extent of the cube (edge length = 1.0)
cube_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size, cube_size, cube_size])

# Define the visual shape (optional, for visualization)
cube_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size, cube_size, cube_size])

# Cube 1
cube1_position = [0, 0, 0]  # Position of cube 1
cube1_orientation = [0, 0, 0, 1]  # Orientation (quaternion)
cube1_id = p.createMultiBody(
    baseMass=1.0,  # Dynamic object (non-zero mass)
    baseCollisionShapeIndex=cube_collision_shape,
    baseVisualShapeIndex=cube_visual_shape,
    basePosition=cube1_position,
    baseOrientation=cube1_orientation
)

# Cube 2
cube2_position = [1, 0, 0]  # Slightly offset in x direction to test collision
cube2_orientation = [0, 0, 0, 1]  # Orientation (quaternion)
cube2_id = p.createMultiBody(
    baseMass=1.0,  # Dynamic object
    baseCollisionShapeIndex=cube_collision_shape,
    baseVisualShapeIndex=cube_visual_shape,
    basePosition=cube2_position,
    baseOrientation=cube2_orientation
)

# Run the simulation and check for collisions
for _ in range(100):
    # Step the simulation
    p.stepSimulation()

    # Get contact points between the two cubes
    contact_points = p.getContactPoints(bodyA=cube1_id, bodyB=cube2_id)

    if contact_points:
        print("Collision detected!")
        for contact in contact_points:
            print(f"Contact point: {contact[5]}, Contact normal: {contact[7]}")
        break
    else:
        print("No collision detected.")

# Keep the simulation running for visualization
while True:
    p.stepSimulation()
