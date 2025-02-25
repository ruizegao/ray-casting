python src/compute_bounds.py sample_inputs/fox.npz trees/tree_fox_18_27_opt.npz --split_depth 18 --max_split_depth 27
python src/compute_meshes.py trees/tree_fox_18_27_opt.npz meshes/mesh_fox_18_27_opt_outer.npz
python src/mesh_raycasting.py sample_inputs/fox.npz meshes/mesh_fox_18_27_opt_outer.npz --image_write_path images/img_fox_18_27_opt_outer.png

python src/compute_bounds.py sample_inputs/skull.pth trees/tree_skull_18_24.npz --start_depth 15 --split_depth 18 --max_split_depth 24 --batch_size 32
python src/compute_meshes.py trees/tree_skull_16_18.npz meshes/mesh_fox_16_18.npz
