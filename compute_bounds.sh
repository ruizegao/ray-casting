python src/compute_bounds.py sample_inputs/cat.npz trees/tree_cat_27.npz --split_depth 24 --max_split_depth 27
python src/compute_meshes.py trees/tree_cat_27.npz meshes/mesh_cat_27.obj

python src/compute_bounds.py sample_inputs/koala.npz trees/tree_koala_27.npz --split_depth 24 --max_split_depth 27
python src/compute_meshes.py trees/tree_koala_27.npz meshes/mesh_koala_27.obj

python src/compute_bounds.py sample_inputs/tree.npz trees/tree_tree_27.npz --split_depth 24 --max_split_depth 27
python src/compute_meshes.py trees/tree_tree_27.npz meshes/mesh_tree_27.obj