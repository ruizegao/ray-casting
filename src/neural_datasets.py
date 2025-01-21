"""
Dataset loaders to aid in training the SDF/occupancy based neural networks.
"""
import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Union, Tuple, Optional
import numpy as np
from warnings import warn

from neural_rendering import ImageSDF
import igl, geometry

class SampleDataset(Dataset):
    def __init__(
            self,
            input_file: str,
            fit_mode: str,
            n_samples: int,
            sdf_max: float,
            sample_weight_beta: float,
            sample_ambient_range: float,
            sample_221: bool = False,
            show_sample_221: bool = False,
            shape_dict: Optional[dict] = None,
            verbose=False
    ):
        """
        Creates a dataset of samples given a mesh input file of type .obj. Target label varies based on what type of
        implicit surface fitting is being done.
        :param input_file:         Mesh input to fit
        :param fit_mode:                Fit mode for the mesh with options ['occupancy', 'sdf', 'tanh']
        :param n_samples:               Number of samples to create
        :param sample_weight_beta:      The sample weight beta factor
        :param sample_ambient_range:
        :param sample_221:              Use 2-2-1 sampling
        :param verbose:                 If true, prints additional info during dataset creation
        """

        if verbose:
            print(f"Loading mesh {input_file}")
        if shape_dict is None: shape_dict = {}
            
        if input_file.endswith(".obj"):
            V, F = igl.read_triangle_mesh(input_file)
            V = torch.from_numpy(V)
            F = torch.from_numpy(F)
            V = geometry.normalize_positions(V, method='bbox')
            if verbose:
                print(f"Collecting geometry samples. Is using sample_221? {sample_221}")
            if sample_221:
                samp, samp_SDF = geometry.sample_221(V, F, n_samples, sample_ambient_range, sdf_max, show_sample_221)
            else:
                samp, samp_SDF = geometry.sample_mesh_importance(V, F, n_samples, beta=sample_weight_beta,
                                                                 ambient_range=sample_ambient_range, sdf_max=sdf_max,
                                                                 show_surface=show_sample_221)
        elif input_file.endswith(".png"):
            poly_indices = shape_dict.get('poly_indices', None)
            png_sdf = ImageSDF(input_file, poly_indices=poly_indices)
            self._exact_sdf = png_sdf
            samp = torch.from_numpy(png_sdf.coords)
            on_surface_points = samp.shape[0]
            warn(f"'on_surface_points' has been updated to be {on_surface_points}. "
                 f"This is determined by the gpytoolbox.")

        if fit_mode == 'occupancy':
            # apply label and calculate sample weight to correct class imbalance
            samp_target = (samp_SDF > 0) * 1.0
            n_pos = np.sum(samp_target > 0)
            n_neg = samp_target.shape[0] - n_pos
            w_pos = n_neg / (n_pos + n_neg)
            w_neg = n_pos / (n_pos + n_neg)
            samp_weight = np.where(samp_target > 0, w_pos, w_neg)
        elif fit_mode in 'sdf':
            # apply label and give all weights equal importance
            # since this is regression not classification based
            samp_target = samp_SDF
            samp_weight = np.ones_like(samp_target)
        else:
            raise ValueError(f"Fit mode {fit_mode} not recognized. Please select from ['occupancy', 'sdf'].")
        # in the original implicit file, the samp_weight array is not used at all so disregard it here as well

        # save inputs and labels
        if verbose:
            print(f"Saving samples and labels to the dataset")
        self.x = torch.from_numpy(samp)  # shape (n_samples, 3)
        self.y = torch.from_numpy(samp_target).reshape(n_samples, 1)
        self.weights = torch.from_numpy(samp_weight).reshape(n_samples, 1)

    def __len__(self) -> int:
            return len(self.x)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        return self.x[idx], self.y[idx], self.weights[idx]

class PointCloud(Dataset):
    def __init__(self, pointcloud_path: str, on_surface_points: int, keep_aspect_ratio: bool=True,
                 shape_dict: Optional[dict] = None):
        """
        
        :param pointcloud_path:
        :param on_surface_points:
        :param keep_aspect_ratio:
        :param shape_dict:
        """
        super().__init__()

        if shape_dict is None: shape_dict = {}

        if pointcloud_path.endswith(".npy"):
            print("Loading point cloud")
            point_cloud = np.genfromtxt(pointcloud_path)
            print("Finished loading point cloud")

            coords = point_cloud[:, :3]
            self.normals = point_cloud[:, 3:]
        elif pointcloud_path.endswith(".png"):
            poly_indices = shape_dict.get('poly_indices')
            png_sdf = ImageSDF(pointcloud_path, poly_indices=poly_indices)
            self._exact_sdf = png_sdf
            coords = torch.from_numpy(png_sdf.coords)
            on_surface_points = coords.shape[0]
            warn(f"'on_surface_points' has been updated to be {on_surface_points}. "
                 f"This is determined by the gpytoolbox.")

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        surface_mask = np.zeros((total_samples, 1), dtype=bool)
        surface_mask[0:self.on_surface_points, :] = True
        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float(),
                                                              'surface_mask': torch.from_numpy(surface_mask)}