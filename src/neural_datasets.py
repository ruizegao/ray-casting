"""
Dataset loaders to aid in training the SDF/occupancy based neural networks.
"""
import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Union, Tuple, Optional
import numpy as np
from numpy import ndarray
from warnings import warn
from PIL import Image
from gpytoolbox import png2poly, edge_indices, normalize_points, signed_distance

import igl, geometry

to_numpy = lambda x : x.detach().cpu().numpy()

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
            init_scale_factor: int = 2,
            device = torch.device('cuda'),
            verbose=False
    ):
        """
        Creates a dataset of samples given a mesh input file of type .obj. Target label varies based on what type of
        implicit surface fitting is being done.
        :param input_file:              Mesh input to fit
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

        # initialize the dataset based on what type of data is being loaded in
        if input_file.endswith(".obj"):
            obj_args = {
                'mesh_input_file': input_file,
                'fit_mode': fit_mode,
                'n_samples': n_samples,
                'sdf_max': sdf_max,
                'sample_weight_beta': sample_weight_beta,
                'sample_ambient_range': sample_ambient_range,
                'sample_221': sample_221,
                'show_sample_221': show_sample_221,
                'verbose': verbose,
            }
            self._init_from_obj(**obj_args)
        elif input_file.endswith(".png"):
            png_args = {
                'sdf_png_path': input_file,
                'shape_dict': shape_dict,
                'fit_mode': fit_mode,
                'sdf_max': sdf_max,
                'minimum_points': n_samples,
                'init_scale_factor': init_scale_factor,
                'device': device
            }
            self._init_from_png(**png_args)
        else:
            raise ValueError(f"Input file {input_file} is not a .obj or .png file.")

    def _init_from_obj(
            self,
            mesh_input_file: str,
            fit_mode: str,
            n_samples: int,
            sdf_max: float,
            sample_weight_beta: float,
            sample_ambient_range: float,
            sample_221: bool = False,
            show_sample_221: bool = False,
            verbose=False):
        """
        Creates a dataset of samples given a mesh input file of type .obj. Target label varies based on what type of
        implicit surface fitting is being done.
        :param mesh_input_file:         Mesh input to fit
        :param fit_mode:                Fit mode for the mesh with options ['occupancy', 'sdf', 'tanh']
        :param n_samples:               Number of samples to create
        :param sample_weight_beta:      The sample weight beta factor
        :param sample_ambient_range:
        :param sample_221:              Use 2-2-1 sampling
        :param verbose:                 If true, prints additional info during dataset creation
        """

        if verbose:
            print(f"Loading mesh {mesh_input_file}")
        V, F = igl.read_triangle_mesh(mesh_input_file)
        V = torch.from_numpy(V)
        F = torch.from_numpy(F)
        # preprocess (center and scale)
        if verbose:
            print("Normalizing position array")
        V = geometry.normalize_positions(V, method='bbox')

        if verbose:
            print(f"Collecting geometry samples. Is using sample_221? {sample_221}")
        if sample_221:
            samp, samp_SDF = geometry.sample_221(V, F, n_samples, sample_ambient_range, sdf_max, show_sample_221)
        else:
            samp, samp_SDF = geometry.sample_mesh_importance(V, F, n_samples, beta=sample_weight_beta,
                                                             ambient_range=sample_ambient_range, sdf_max=sdf_max,
                                                             show_surface=show_sample_221)

        if verbose:
            print(f"Formatting labels")
        if fit_mode == 'occupancy':
            # apply label and calculate sample weight to correct class imbalance
            samp_target = (samp_SDF > 0) * 1.0
            n_pos = np.sum(samp_target > 0)
            n_neg = samp_target.shape[0] - n_pos
            w_pos = n_neg / (n_pos + n_neg)
            w_neg = n_pos / (n_pos + n_neg)
            samp_weight = np.where(samp_target > 0, w_pos, w_neg)
        elif fit_mode in ['sdf', 'tanh']:
            # apply label and give all weights equal importance
            # since this is regression not classification based
            samp_target = samp_SDF
            samp_weight = np.ones_like(samp_target)
        else:
            raise ValueError(f"Fit mode {fit_mode} not recognized. Please select from ['occupancy', 'sdf', 'tanh'].")
        # in the original implicit file, the samp_weight array is not used at all so disregard it here as well

        # save inputs and labels
        if verbose:
            print(f"Saving samples and labels to the dataset")
        self.x = torch.from_numpy(samp)  # shape (n_samples, 3)
        self.y = torch.from_numpy(samp_target).reshape(n_samples, 1)
        self.weights = torch.from_numpy(samp_weight).reshape(n_samples, 1)

    def _init_from_png(self, sdf_png_path: str, shape_dict: dict, fit_mode: str, minimum_points: int,
                       init_scale_factor: int, sdf_max: float, device: torch.device):
        """
        Initializes a dataset of SDF/occupancy based samples from a black and white png image. This function
        relies on the gpytoolbox to collect these samples.
        :param sdf_png_path:
        :param shape_dict:
        :param fit_mode:
        :param minimum_points:
        :param init_scale_factor:
        :param sdf_max:
        :param device:
        :return:
        """
        poly_indices = shape_dict.get('poly_indices', None)
        png_sdf = ImageSDF(sdf_png_path, poly_indices=poly_indices, min_points=minimum_points,
                           init_scale_factor=init_scale_factor)
        self._exact_sdf = png_sdf
        coords = torch.from_numpy(png_sdf.coords)
        on_surface_points = coords.shape[0]
        warn(f"'on_surface_points' has been updated to be {on_surface_points}. "
             f"This is determined by the gpytoolbox.")

        # normalize the coordinates and send them to the CPU
        # self._coords = self._coords_normalization(coords, keep_aspect_ratio).to(device='cpu')
        coords = coords.cpu()  # shape (on_surface_samples, 2)
        self._coords = coords

        self._on_surface_points = on_surface_points

        ### for mlp
        off_surface_points = 3 * on_surface_points if fit_mode == 'sdf' else on_surface_points
        # n_samples = on_surface_points + off_surface_points
        off_surface_coords = torch.from_numpy(np.random.uniform(-0.55, 0.55, size=(off_surface_points, 2))).to(
            device=device
        )
        samp_SDF, _ = self._exact_sdf(off_surface_coords, device=device)
        samp_SDF = samp_SDF.cpu()

        if fit_mode == 'occupancy':
            # apply label and calculate sample weight to correct class imbalance
            samp_target = (samp_SDF > 0) * 1.0
            n_pos = torch.sum(samp_target > 0)
            n_neg = samp_target.shape[0] - n_pos
            w_pos = n_neg / (n_pos + n_neg)
            w_neg = n_pos / (n_pos + n_neg)
            samp_weight = torch.where(samp_target > 0, w_pos, w_neg)

            coords = torch.concatenate((coords, off_surface_coords.cpu()), dim=0)
            samp_target = samp_target.cpu().reshape(off_surface_points, 1).repeat(2, 1)
            samp_target[:on_surface_points, :] = 0.
            samp_weight = samp_weight.cpu().reshape(off_surface_points, 1).repeat(2, 1)
            samp_weight[:on_surface_points, :] = 1.
        elif fit_mode == 'sdf':
            # apply label and give all weights equal importance
            # since this is regression not classification based
            samp_target = samp_SDF

            close_mask = (samp_target.abs() <= sdf_max)
            num_close = close_mask.to(dtype=int).sum().item()
            close_points = samp_target[close_mask]
            num_left = on_surface_points - num_close
            if num_left <= 0:
                off_surface_coords = off_surface_coords[close_mask][:on_surface_points, :].reshape(on_surface_points, 2)
                samp_target = torch.concatenate((torch.zeros(on_surface_points, 1),
                                                 close_points[:on_surface_points].reshape(on_surface_points, 1)), dim=0)
                samp_weight = torch.ones(on_surface_points*2, 1)
            else:
                off_surface_coords = torch.concatenate(
                    (off_surface_coords[close_mask][:num_close, :].reshape(num_close, 2),
                     off_surface_coords[torch.logical_not(close_mask)][:num_left, :].reshape(num_left, 2)), dim=0)
                samp_target = torch.concatenate((torch.zeros((on_surface_points, 1)),
                                                 samp_target[close_mask].reshape(num_close, 1),
                                                 sdf_max * torch.sign(
                                                     samp_target[torch.logical_not(close_mask)][:num_left]).reshape(
                                                     num_left, 1)), dim=0)
                samp_weight = torch.concatenate((9 / 20 * torch.ones((on_surface_points + num_close, 1)),
                                                 1 / 10 * torch.ones((num_left, 1))), dim=0)
            # save inputs and labels
            coords = torch.concatenate((coords, off_surface_coords.cpu()), dim=0)
            # samp_target = samp_target.cpu().reshape(off_surface_points, 1).repeat(2, 1)
            # samp_target[:on_surface_points, :] = 0.
            # samp_weight = samp_weight.cpu().reshape(off_surface_points, 1).repeat(2, 1)
            # samp_weight[:on_surface_points, :] = 9/20
            # close_mask = (samp_target[on_surface_points:, :].abs() <= 0.05)
            # num_close = close_mask.to(dtype=int).sum()
            print(f"{num_close} samples are close to the surface.")
            # off_weights = torch.where(close_mask, 9/20, 1/10)
            # off_targets = torch.where(close_mask, samp_target[on_surface_points:, :], 0.05*torch.sign(samp_target[on_surface_points:, :]))
            # samp_target[on_surface_points:, :] = off_targets
            # samp_weight[on_surface_points:, :] = off_weights
            # samp_weight[on_surface_points:, :] = 1/3
        else:
            raise ValueError(f"Fit mode {fit_mode} not recognized. Please select from ['occupancy', 'sdf'].")

        self.x = coords  # shape (n_samples, 2)
        self.y = samp_target
        self.weights = samp_weight

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

class ImageSDF:
    """
    This class uses the gpytoolbox in order to load in 2D images and create an SDF function. It works best with
    black+white png images and shapes that are relatively simple.
    """
    def __init__(self, path: str, poly_indices: Optional[Union[ndarray[int], list[int]]]=None,
                 min_points: Optional[int] = None, init_scale_factor: int = 2):

        # generate vertices of the original image
        vertices = self._generate_vertices(path, poly_indices)

        if min_points is not None and len(vertices) < min_points:
            # not enough samples of the surface has been created as requested by the min_points parameter,
            # prepare to upscale the image
            image = Image.open(path)
            scale_factor = init_scale_factor
            temp_path = path.split('.png')[0] + "_temp_upscaled.png"
            while len(vertices) < min_points:
                print(f"{len(vertices)} < {min_points}, rescaling image by a factor of {scale_factor}")
                new_width = int(image.width * scale_factor)
                new_height = int(image.height * scale_factor)
                new_size = (new_width, new_height)
                print(f"New shape of upscaled image is {new_size}")
                upscaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
                upscaled_image.save(temp_path)

                vertices = self._generate_vertices(temp_path, poly_indices)

                scale_factor += 1
            # os.remove(temp_path)

        # save the contour edges and normalized vertices
        self._edges = edge_indices(vertices.shape[0], closed=True)
        self._vertices = normalize_points(vertices)
        print(f"PNG Image '{path}' generated {len(vertices)} vertices.")

    def _generate_vertices(self, img_path: str, poly_indices: Optional[Union[ndarray[int], list[int]]]=None):
        """
        Using the png2poly function from gpytoolbox, generates the samples of the zero-levelset surface in the image.
        If the image has a very awkward shape and multiple contours are detected, the poly_indices parameter may be
        used to select specific contours. In the latter scenario, it is recommended to experiment with png2poly
        in a separate script/notebook to get the correct parameters for extracting the vertices.
        :param img_path:        Path to 2D black+white png image
        :param poly_indices:    Indices to select contours and vertices from the extracted png2poly results
        :return:
        """
        poly = png2poly(img_path)
        if poly_indices is None:
            vertices = poly[0]
        else:
            vertices = np.concatenate([poly[i] for i in poly_indices])
        return vertices
    def calculate_signed_distance(self, pts: Tensor, device: torch.device) -> Tuple[Tensor, None]:
        """
        For a set of 2D points, calculates its distances from the contour in the original image.
        :param pts:
        :param device:
        :return:
        """
        np_pts = to_numpy(pts)
        sdist, ind, t = signed_distance(np_pts, self._vertices, F=self._edges)
        dist = torch.from_numpy(sdist).to(device)
        return dist, None

    def __call__(self, x: Tensor, device: torch.device) -> Tuple[Tensor, None]:
        return self.calculate_signed_distance(x, device)

    @property
    def coords(self) -> ndarray:
        return self._vertices