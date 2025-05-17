#python src/compute_bounds.py sample_inputs/cat.npz trees/tree_cat_27.npz --split_depth 24 --max_split_depth 27
#python src/compute_meshes.py trees/tree_cat_27.npz meshes/mesh_cat_27.obj
#
#python src/compute_bounds.py sample_inputs/koala.npz trees/tree_koala_27.npz --split_depth 24 --max_split_depth 27
#python src/compute_meshes.py trees/tree_koala_27.npz meshes/mesh_koala_27.obj
#
#python src/compute_bounds.py sample_inputs/tree.npz trees/tree_tree_27.npz --split_depth 24 --max_split_depth 27
#python src/compute_meshes.py trees/tree_tree_27.npz meshes/mesh_tree_27.obj

#python src/mesh_raycasting.py sample_inputs/cat.npz meshes/mesh_cat_adaptive_mid.obj --option approx --grid_cam --output rendering/cat_grid_cam_mid_shell.npz --log_output exp_results/render/cat/mid_shell_time.npz
#python src/mesh_raycasting.py sample_inputs/koala.npz meshes/mesh_koala_adaptive_mid.obj --option approx --grid_cam --output rendering/koala_grid_cam_mid_shell.npz --log_output exp_results/render/koala/mid_shell_time.npz
#python src/mesh_raycasting.py sample_inputs/tree.npz meshes/mesh_tree_adaptive_mid.obj --option approx --grid_cam --output rendering/tree_grid_cam_mid_shell.npz --log_output exp_results/render/tree/mid_shell_time.npz
#python src/mesh_raycasting.py sample_inputs/fox.npz meshes/mesh_fox_adaptive_mid.obj --option approx --grid_cam --output rendering/fox_grid_cam_mid_shell.npz --log_output exp_results/render/fox/mid_shell_time.npz

#python src/compute_bounds.py sample_inputs/tree.npz trees/tree_tree_collision.npz --split_depth 3 --max_split_depth 15
#python src/compute_meshes.py trees/tree_tree_collision.npz meshes/mesh_tree_collision.obj

#python src/neural_train.py --input_file ./2d_train_data/cat_2d.png --output_file sample_inputs/cat_2d.npz --model_type mlp --input_dim 2 --activation relu --n_layers 5 --layer_width 64 --lr 1e-3 --n_samples 20000 --batch_size 2048 --n_epoch 10000 --sample_ambient_range 1. --fit_mode sdf --clip_gradient_norm 1.0


#python src/compute_bounds.py sample_inputs/koala.npz trees/tree_koala_crown.npz --split_depth 3 --max_split_depth 18
#python src/compute_meshes.py trees/tree_koala_crown.npz meshes/mesh_koala_crown.obj

python src/compute_bounds.py sample_inputs/fox.npz trees/tree_fox_adaptive.npz --split_depth 18 --max_split_depth 33
python src/compute_meshes.py trees/tree_fox_adaptive.npz meshes/mesh_fox_adaptive.obj