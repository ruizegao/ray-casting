# import igl # work around some env/packaging problems by loading this first

# import sys, os, time, math
# os.environ['OptiX_INSTALL_DIR'] = '/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'

import time
import argparse
import warnings

import torch
import os

# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
import matplotlib as plt
import imageio
import pybullet as p
import time
import trimesh

# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CROWN_MODES = ['crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def detect_collision_between(mesh_1_path, mesh_2_path):
    # Initialize the physics engine
    p.connect(p.GUI)  # Use p.DIRECT for headless execution
    p.setGravity(0, 0, 0)  # No gravity for collision detection

    # Load mesh 1
    mesh1_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_1_path,
        meshScale=[1, 1, 1],  # Scale the mesh if needed
    )
    mesh1_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_1_path,
        meshScale=[1, 1, 1],  # Scale the mesh if needed
    )
    mesh1_body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=mesh1_collision_shape, baseVisualShapeIndex=mesh1_visual_shape)
    # mesh1_body = p.createMultiBody(
    #     baseMass=1.0,  # Dynamic object
    #     baseCollisionShapeIndex=mesh1_collision_shape
    # )
    # Load mesh 2
    mesh2_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_2_path,
        meshScale=[1, 1, 1],  # Scale the mesh if needed
    )
    mesh2_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_2_path,
        meshScale=[1, 1, 1],
    )
    mesh2_body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=mesh2_collision_shape, baseVisualShapeIndex=mesh2_visual_shape)

    # Position the meshes
    p.resetBasePositionAndOrientation(mesh1_body, [1, 0, 0], [0, 0, 0, 1])  # [x, y, z] and [qx, qy, qz, qw]
    p.resetBasePositionAndOrientation(mesh2_body, [1, 0, 0], [0, 0, 0, 1])

    # Step simulation to detect collisions
    collision_detected = False
    for i in range(100):  # Perform multiple simulation steps
        p.stepSimulation()

        # Check for collisions
        contact_points = p.getContactPoints(mesh1_body, mesh2_body)
        print(contact_points)
        if contact_points:
            collision_detected = True
            print(f"Collision detected at step {i}!")
            for contact in contact_points:
                print(f"Contact point: {contact[5]}")  # Prints the position of the contact
            break

        time.sleep(0.01)  # Optional: Slow down to observe in GUI

    if not collision_detected:
        print("No collision detected.")

    # Clean up
    p.disconnect()

def viz_mesh(mesh_path):
    # Initialize the physics engine
    p.connect(p.GUI)  # Use p.DIRECT for headless execution
    p.setGravity(0, 0, 0)  # No gravity for collision detection

    # Load mesh 1
    mesh_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[1, 1, 1],  # Scale the mesh if needed
    )
    mesh_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[1, 1, 1],  # Scale the mesh if needed
    )
    mesh_body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=mesh_collision_shape, baseVisualShapeIndex=mesh_visual_shape)
    p.resetBasePositionAndOrientation(mesh_body, [1, 0, 0], [0, 0, 0, 1])  # [x, y, z] and [qx, qy, qz, qw]
    p.resetDebugVisualizerCamera(
        cameraDistance=2,  # Distance from the object
        cameraYaw=45,  # Yaw angle
        cameraPitch=-30,  # Pitch angle
        cameraTargetPosition=[0, 0, 0]  # Target position
    )
    while True:
        p.stepSimulation()

def intersection_visualization(mesh_1_path, mesh_2_path):

    # Load two meshes
    mesh1 = trimesh.load(mesh_1_path)
    print(mesh.mass)
    print(mesh1.is_watertight)
    mesh2 = trimesh.load(mesh_2_path)
    mesh1.process()
    mesh2.process()
    # Compute intersection
    intersection = mesh1.intersection(mesh2)

    # Visualize the intersection
    intersection.show()

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("mesh_1", type=str)
    parser.add_argument("mesh_2", type=str)
    # Parse arguments
    args = parser.parse_args()
    viz_mesh(args.mesh_1)
    # intersection_visualization(args.mesh_1, args.mesh_2)
    # detect_collision_between(args.mesh_1, args.mesh_2)

if __name__ == '__main__':
    main()
