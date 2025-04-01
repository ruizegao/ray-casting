python src/compute_bounds.py sample_inputs/fox.npz trees/tree_fox_24_30.npz --split_depth 24 --max_split_depth 30
python src/compute_meshes.py trees/tree_fox_24_30.npz meshes/mesh_fox_24_30_outer.npz
python src/mesh_raycasting.py sample_inputs/fox.npz meshes/mesh_fox_18_27_opt_outer.npz --image_write_path images/img_fox_18_27_opt_outer.png

python src/compute_bounds.py sample_inputs/skull.pth trees/tree_skull_18_24.npz --start_depth 15 --split_depth 18 --max_split_depth 24 --batch_size 32
python src/compute_meshes.py trees/tree_skull_16_18.npz meshes/mesh_fox_16_18.npz

python src/mesh_raycasting.py sample_inputs/fox.npz meshes/mesh_fox_adaptive.npz --image_write_path images/img_test.png

python src/compute_bounds.py sample_inputs/hammer.npz trees/tree_hammer_30.npz --split_depth 24 --max_split_depth 30
python src/compute_meshes.py trees/tree_hammer_30.npz meshes/mesh_hammer_30.obj
