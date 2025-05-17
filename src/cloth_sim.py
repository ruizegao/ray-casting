# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Sim Granular Collision SDF
#
# Shows how to set up a particle-based granular material model using the
# wp.sim.ModelBuilder(). This version shows how to create collision geometry
# objects from SDFs.
#
# Note: requires a CUDA-capable device
###########################################################################

import os

import numpy as np
from pxr import Usd, UsdGeom
import trimesh
import math

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
global detection_time

def print_custom_report(results, indent=""):
    prefilter_time = 0
    soft_contact_time = 0

    for r in results:
        if "prefilter_particles_pim" in r.name and r.name.startswith("forward kernel"):
            prefilter_time += r.elapsed
        if "create_soft_contacts" in r.name and r.name.startswith("forward kernel"):
            soft_contact_time += r.elapsed

    print(f"{indent}prefilter kernels: {prefilter_time:.6f} ms")
    print(f"{indent}create soft contacts kernels: {soft_contact_time:.6f} ms")
    print(f"{indent}total: {prefilter_time+soft_contact_time:.6f} ms")
    detection_time.append(prefilter_time+soft_contact_time)

class Example:
    def __init__(self, stage_path="example_cloth_sim_sdf.usd"):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.radius = 0.01

        builder = wp.sim.ModelBuilder()
        # builder.default_particle_radius = self.radius
        # self.particle_grid_dim_x = 64
        # self.particle_grid_dim_y = 16
        # self.particle_grid_dim_z = 64
        # self.particle_grid_height = 2.5
        # self.particle_grid_pos = wp.vec3(-self.particle_grid_dim_x * self.radius, self.particle_grid_height, -self.particle_grid_dim_z * self.radius)
        self.sim_width = 32
        self.sim_height = 16
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 4.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.sim_width,
            dim_y=self.sim_height,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            fix_left=True,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e1,
        )

        vdb, sdf_np, mins, voxel_size, bg_value, coords = self.load_np_sdf('grid_sdf/tree.npz')

        mesh = trimesh.load("meshes/mesh_tree_collision_outer.obj", force='mesh')
        mesh_bb = trimesh.creation.box(bounds=mesh.bounds)
        # mesh = trimesh.load("meshes/mesh_tree_27_outer.obj", force='mesh')
        if not mesh.is_volume:
            print("Warning: Mesh is not watertight. Physics collisions may behave oddly.")
            components = mesh.split(only_watertight=True)  # Set to True if you only want watertight components
            mesh = max(components, key=lambda m: len(m.faces))
        vertices_np = np.asarray(mesh.vertices, dtype=np.float32)
        indices_np = np.asarray(mesh.faces, dtype=np.int32).flatten()
        print(vertices_np.min())
        self.vertices = vertices_np
        self.indices = indices_np

        sdf = wp.sim.SDF(volume=vdb)#, shell_vertices=self.vertices, shell_indices=self.indices)
        # sdf = wp.sim.SDF(volume=vdb, shell_vertices=mesh_bb.vertices, shell_indices=mesh_bb.faces)

        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=sdf,
            body=-1,
            pos=wp.vec3(0., 0.82, 0.),
            rot=wp.quat(0.0, 0.0, 0.0, 1.0),
            scale=wp.vec3(1.0, 1.0, 1.0),
        )

        # mesh = self.load_obj_mesh("meshes/mesh_tree_0lvl.obj")

        self.model = builder.finalize()

        print(self.model.shape_geo)

        self.model.particle_kf = 25.0

        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf *= 2.0
        self.model.soft_contact_ke = 1.0e4
        # self.model.soft_contact_kd = 1.0e2
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)

        self.integrator = wp.sim.SemiImplicitIntegrator()
        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
        else:
            self.renderer = None

        # self.use_cuda_graph = wp.get_device().is_cuda
        self.use_cuda_graph = False
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def load_mesh(self, filename, path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()
        return wp.sim.Mesh(points, indices)

    def load_obj_mesh(self, filename):
        mesh = trimesh.load(filename, force='mesh')
        # mesh.simplify_quadric_decimation(face_count=500000)
        if not mesh.is_volume:
            print("Warning: Mesh is not watertight. Physics collisions may behave oddly.")
            # components = mesh.split(only_watertight=True)  # Set to True if you only want watertight components
            # mesh = max(components, key=lambda m: len(m.faces))
        # trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.51, iterations=10)
        points = wp.array(np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3)
        indices = wp.array(np.asarray(mesh.faces, dtype=np.int32).flatten(), dtype=int)
        # print(points.shape, indices.shape)
        mesh = wp.Mesh(points, indices)
        return mesh

    def load_np_sdf(self, filename):
        data = np.load(filename)
        sdf_np = data['sdf']
        mins = (data['mins']).tolist()
        voxel_size = float(data['voxel_size'])
        bg_value = float(data['bg_value'])
        coords = data['coords']

        vdb = wp.Volume.load_from_numpy(sdf_np, min_world=mins, voxel_size=voxel_size, bg_value=bg_value)

        return vdb, sdf_np, mins, voxel_size, bg_value, coords

    def simulate(self):
        wp.sim.prefilter_particles(self.model, self.state_0)
        wp.sim.allocate_compact_soft_contacts(self.model)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0, apply_particle_filter=True)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step", cuda_filter=wp.TIMING_ALL, report_func=print_custom_report):
        # with wp.ScopedTimer("step", cuda_filter=wp.TIMING_GRAPH):
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            correction = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.radians(-90.0))

            self.renderer.render_ref(
                name="collision",
                path='meshes/tree.usd',
                pos=wp.vec3(0., 0.82, 0.),
                rot=correction,
                scale=wp.vec3(1., 1., 1.),
                # color=(0.35, 0.55, 0.9),
            )

            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cloth_sim_sdf.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=100, help="Total number of frames.")
    args = parser.parse_known_args()[0]
    detection_time = []
    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()

    detection_time = np.array(detection_time)
    np.savez_compressed('assets/example_cloth_sim_sdf_time.npz', time=detection_time)