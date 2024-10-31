#python src/compute_bounds.py sample_inputs/fox.npz trees/tree_fox_30.npz --split_depth 30
python src/compute_bounds.py sample_inputs/fox.npz trees/tree_fox_adaptive.npz
python src/compute_meshes.py trees/tree_fox_adaptive.npz meshes/mesh_fox_adaptive.npz
python src/mesh_raycasting.py sample_inputs/fox.npz meshes/mesh_fox_adaptive.npz --image_write_path images/img_fox_adaptive.png
