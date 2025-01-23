import pybullet as p
import argparse
import time

def closest_point(point, mesh_path):
    # Initialize a Bullet physics server
    p.connect(p.DIRECT)  # Use GUI for visualization
    p.setGravity(0, 0, 0)  # No gravity for this scenario

    # Load the mesh
    mesh_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[1, 1, 1],  # Scale the mesh if needed
    )
    mesh_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[1, 1, 1],  # Scale the mesh if needed
    )
    mesh_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=mesh_collision_shape,
        baseVisualShapeIndex=mesh_visual_shape,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )

    # Define the query point
    query_position = [2.0, 1.0, 1.0]  # Replace with your query point
    query_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
    query_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
    query_sphere_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=query_collision_shape,
        baseVisualShapeIndex=query_visual_shape,
        basePosition=query_position,
    )

    # Perform the closest point query
    max_distance = 10.0  # Maximum search distance
    results = p.getClosestPoints(bodyA=query_sphere_id, bodyB=mesh_id, distance=max_distance)

    lineWidth = 3
    colorRGB = [1, 0, 0]
    lineId = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                lineToXYZ=[0, 0, 0],
                                lineColorRGB=colorRGB,
                                lineWidth=lineWidth,
                                lifeTime=0)

    # Extract the closest point and visualize it
    if results:
        closest_point_info = results[0]
        closest_pointA = closest_point_info[5]  # Closest point on the surface
        closest_pointB = closest_point_info[6]  # Closest point on the surface
        print("Closest point:", closest_pointA, closest_pointB)

        # Visualize the closest point
        # closest_point_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1])
        # closest_point_id = p.createMultiBody(
        #     baseMass=0,
        #     baseCollisionShapeIndex=-1,
        #     baseVisualShapeIndex=closest_point_visual_shape,
        #     basePosition=closest_point,
        # )
        # p.addUserDebugLine(lineFromXYZ=closest_pointA,
        #                    lineToXYZ=closest_point_info,
        #                    lineColorRGB=colorRGB,
        #                    lineWidth=lineWidth,
        #                    lifeTime=0,
        #                    replaceItemUniqueId=lineId)
    else:
        print("No points found within the specified distance.")

    # Run the visualization for a while
    for _ in range(10000):
        p.stepSimulation()
        time.sleep(0.01)  # Slow down the simulation to make it interactive

    p.disconnect()

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("mesh", type=str, help="Path to the mesh file")
    # Parse arguments
    args = parser.parse_args()
    closest_point(None, args.mesh)

if __name__ == "__main__":
    main()
