python src/compute_bounds.py sample_inputs/fox.npz trees/tree_fox_18_27_opt.npz --split_depth 18 --max_split_depth 27
python src/compute_meshes.py trees/tree_fox_18_27_opt.npz meshes/mesh_fox_18_27_opt_outer.npz
python src/mesh_raycasting.py sample_inputs/fox.npz meshes/mesh_fox_18_27_opt_outer.npz --image_write_path images/img_fox_18_27_opt_outer.png
