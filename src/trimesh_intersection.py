import trimesh
import time
import numpy as np


mesh1 = trimesh.load_mesh('meshes/mesh_fox_adaptive_mid_old.obj')
# mesh1 = trimesh.load_mesh('/home/ruize/Downloads/ribcage.obj')
if not mesh1.is_volume:
    print("mesh 1 is not volume")
    components = mesh1.split(only_watertight=True)  # Set to True if you only want watertight components

    mesh1 = max(components, key=lambda m: len(m.faces))
    print(mesh1.vertices[:, 1].min())
    mesh1.show()
# mesh2 = trimesh.load_mesh('meshes/mesh_tree_27_inner.obj')
# if not mesh2.is_volume:
#     print("mesh 2 is not volume")
#     components = mesh2.split(only_watertight=True)  # Set to True if you only want watertight components
#
#     mesh2 = max(components, key=lambda m: len(m.faces))
#     mesh2.show()
# mesh1.show()
# mesh2.show()

# mesh1.show()
mesh1.export("meshes/mesh_fox_adaptive_mid_wt.obj")
# print(len(mesh1.vertices))
# print(len(mesh1.faces))

# both_shell_verts = np.concatenate((np.array(mesh1.vertices), np.array(mesh2.vertices)), axis=0)
# both_shell_faces = np.concatenate((np.array(mesh1.faces), np.array(mesh2.faces) + len(mesh1.vertices)), axis=0)
# both_shells = trimesh.Trimesh(both_shell_verts, both_shell_faces)
# both_shells.export("meshes/mesh_tree_27_both_wt.obj")
