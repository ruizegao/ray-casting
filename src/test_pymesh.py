import pymesh
import trimesh

mesh_A = pymesh.load_mesh("../meshes/mesh_fox_adaptive.npz")
mesh_B = pymesh.load_mesh("../meshes/mesh_hammer_adaptive.npz")
intersection = pymesh.boolean(mesh_A, mesh_B, "intersection")
mesh = trimesh.Trimesh(vertices=intersection.vertices, faces=intersection.faces, process=False)
mesh.show()
