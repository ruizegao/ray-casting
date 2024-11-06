import os
import sys
import gc
import functorch
import torch
import scipy
import numpy as np
from functools import partial
from functorch import vmap
from crown import CrownImplicitFunction
import imageio
from PIL import Image
import geometry
import queries
from utils import *
import affine
# import trimesh
import matplotlib.pyplot as plt
import sys, os, time, math
# os.environ['OptiX_INSTALL_DIR'] = '/home/ruize/Documents/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64'

# from triro.ray.ray_optix import RayMeshIntersector  # FIXME: Should be uncommented when rendering meshes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# theta_x/y should be
def camera_ray(look_dir, up_dir, left_dir, fov_deg_x, fov_deg_y, theta_x, theta_y):
    ray_image_plane_pos = look_dir \
                          + left_dir * (theta_x * torch.tan(
        torch.deg2rad(torch.tensor(fov_deg_x, device=look_dir.device)) / 2)) \
                          + up_dir * (theta_y * torch.tan(
        torch.deg2rad(torch.tensor(fov_deg_y, device=look_dir.device)) / 2))

    ray_dir = geometry.normalize(ray_image_plane_pos)

    return ray_dir


def generate_camera_rays(eye_pos, look_dir, up_dir, res=1024, fov_deg=30.):
    D = res  # image dimension
    R = res * res  # number of rays

    ## Generate rays according to a pinhole camera

    # Image coords on [-1,1] for each output pixel
    cam_ax_x = torch.linspace(-1., 1., res)
    cam_ax_y = torch.linspace(-1., 1., res)
    cam_y, cam_x = torch.meshgrid(cam_ax_x, cam_ax_y, indexing='ij')
    cam_x = cam_x.flatten()
    cam_y = cam_y.flatten()

    # Orthornormal camera frame
    up_dir = up_dir - torch.dot(look_dir, up_dir) * look_dir
    up_dir = geometry.normalize(up_dir)
    left_dir = torch.cross(look_dir, up_dir)

    ray_dirs = vmap(partial(camera_ray, look_dir, up_dir, left_dir, fov_deg, fov_deg))(cam_x, cam_y)
    ray_roots = torch.tile(eye_pos, (ray_dirs.shape[0], 1))
    return ray_roots, ray_dirs


def outward_normal(funcs_tuple, params_tuple, hit_pos, hit_id, eps, method='autodiff'):
    grad_out = torch.zeros(3)
    i_func = 1
    for func, params in zip(funcs_tuple, params_tuple):
        if isinstance(func, CrownImplicitFunction):
            f = partial(func.call_implicit_func, params)
            # f = func.torch_forward
        else:
            f = partial(func, params)

        if method == 'autodiff':
            grad_f = functorch.jacfwd(f)
            grad = grad_f(hit_pos)

        elif method == 'finite_differences':
            # 'tetrahedron' central differences approximation
            # see e.g. https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            offsets = torch.tensor((
                (+eps, -eps, -eps),
                (-eps, -eps, +eps),
                (-eps, +eps, -eps),
                (+eps, +eps, +eps),
            ))
            x_pts = hit_pos[None, :] + offsets
            samples = vmap(f)(x_pts)
            grad = torch.sum(offsets * samples[:, None], dim=0)

        else:
            raise ValueError("unrecognized method")

        grad = geometry.normalize(grad)
        grad_out = torch.where(hit_id == i_func, grad, grad_out)
        i_func += 1

    return grad_out


def outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, eps, method='autodiff'):
    this_normal_one = lambda p, id: outward_normal(funcs_tuple, params_tuple, p, id, eps, method=method)
    if method == 'autodiff':
        N = int(hit_pos.shape[0]/3)
        M = int(2*hit_pos.shape[0]/3)
        ret1 = vmap(this_normal_one)(hit_pos[:N], hit_ids[:N])
        ret2 = vmap(this_normal_one)(hit_pos[N:M], hit_ids[N:M])
        ret3 = vmap(this_normal_one)(hit_pos[M:], hit_ids[M:])
        return torch.cat((ret1, ret2, ret3), dim=0)
    return vmap(this_normal_one)(hit_pos, hit_ids)


def render_image(funcs_tuple, params_tuple, eye_pos, look_dir, up_dir, left_dir, res, fov_deg, frustum, branching_method, opts,
                 shading="normal", shading_color_tuple=((0.157, 0.613, 1.000)), matcaps=None, tonemap=False,
                 shading_color_func=None, tree_based=False, load_from=None, save_to=None):
    # make sure inputs are tuples not lists (can't has lists)
    if isinstance(funcs_tuple, list): funcs_tuple = tuple(funcs_tuple)
    if isinstance(params_tuple, list): params_tuple = tuple(params_tuple)
    if isinstance(shading_color_tuple, list): shading_color_tuple = tuple(shading_color_tuple)

    # wrap in tuples if single was passed
    if not isinstance(funcs_tuple, tuple):
        funcs_tuple = (funcs_tuple,)
    if not isinstance(params_tuple, tuple):
        params_tuple = (params_tuple,)
    if not isinstance(shading_color_tuple[0], tuple):
        shading_color_tuple = (shading_color_tuple,)

    L = len(funcs_tuple)
    if (len(params_tuple) != L) or (len(shading_color_tuple) != L):
        raise ValueError("render_image tuple arguments should all be same length")

    ray_roots, ray_dirs = generate_camera_rays(eye_pos, look_dir, up_dir, res=res, fov_deg=fov_deg)
    if frustum:
        # == Frustum raycasting

        cam_params = eye_pos, look_dir, up_dir, left_dir, fov_deg, fov_deg, res, res

        with Timer("frustum raycast"):
            t_raycast, hit_ids, counts, n_eval = queries.cast_rays_frustum(funcs_tuple, params_tuple, cam_params, opts)
            # t_raycast.block_until_ready()
            torch.cuda.synchronize()

        # TODO transposes here due to image layout conventions. can we get rid of them?
        t_raycast = t_raycast.transpose().flatten()
        hit_ids = hit_ids.transpose().flatten()
        counts = counts.transpose().flatten()

    elif tree_based:
        with Timer("opt_based raycast"):
            # t_raycast, hit_ids, counts, n_eval = queries.cast_rays_cw(funcs_tuple, params_tuple, ray_roots, ray_dirs)
            t_raycast, hit_ids, counts, n_eval = queries.cast_rays_tree_based(funcs_tuple, params_tuple, ray_roots,
                                                                              ray_dirs, load_from=load_from, save_to=save_to)
            # t_raycast, hit_ids, counts, n_eval = queries.cast_rays_parameterized(funcs_tuple, params_tuple, ray_roots, ray_dirs, opts)
            torch.cuda.synchronize()
    else:
        # == Standard raycasting
        with Timer("raycast"):
            t_raycast, hit_ids, counts, n_eval = queries.cast_rays(funcs_tuple, params_tuple, ray_roots, ray_dirs, opts)
            # t_raycast.block_until_ready()
            # print("t_raycast", t_raycast)
            torch.cuda.synchronize()

    hit_pos = ray_roots + t_raycast[:, None] * ray_dirs

    torch.cuda.empty_cache()

    hit_normals = outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, opts['hit_eps'])
    hit_color = shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple,
                            shading_color_func=shading_color_func)
    # print(hit_pos, hit_normals, hit_color)
    img = torch.where(hit_ids[:, None].bool(), hit_color, torch.ones((res * res, 3)))

    if tonemap:
        # We intentionally tonemap before compositing in the shadow. Otherwise the white level clips the shadow and gives it a hard edge.
        img = tonemap_image(img)

    img = img.reshape(res, res, 3)
    depth = t_raycast.reshape(res, res)
    counts = counts.reshape(res, res)
    hit_ids = hit_ids.reshape(res, res)

    return img, depth, counts, hit_ids, n_eval, -1


def render_image_naive(funcs_tuple, params_tuple, eye_pos, look_dir, up_dir, left_dir, res, fov_deg, frustum, opts,
                       shading="normal", shading_color_tuple=((0.157, 0.613, 1.000)), matcaps=None, tonemap=False,
                       shading_color_func=None, tree_based=False, shell_based=False, batch_size=None, enable_clipping=False, load_from=None, save_to=None):
    # make sure inputs are tuples not lists (can't has lists)
    if isinstance(funcs_tuple, list): funcs_tuple = tuple(funcs_tuple)
    if isinstance(params_tuple, list): params_tuple = tuple(params_tuple)
    if isinstance(shading_color_tuple, list): shading_color_tuple = tuple(shading_color_tuple)

    # wrap in tuples if single was passed
    if not isinstance(funcs_tuple, tuple):
        funcs_tuple = (funcs_tuple,)
    if not isinstance(params_tuple, tuple):
        params_tuple = (params_tuple,)
    if not isinstance(shading_color_tuple[0], tuple):
        shading_color_tuple = (shading_color_tuple,)

    L = len(funcs_tuple)
    if (len(params_tuple) != L) or (len(shading_color_tuple) != L):
        raise ValueError("render_image tuple arguments should all be same length")

    ray_roots, ray_dirs = generate_camera_rays(eye_pos, look_dir, up_dir, res=res, fov_deg=fov_deg)
    if frustum:
        # == Frustum raycasting

        cam_params = eye_pos, look_dir, up_dir, left_dir, fov_deg, fov_deg, res, res

        t_raycast, hit_ids, counts, n_eval = queries.cast_rays_frustum(funcs_tuple, params_tuple, cam_params, opts)
        # t_raycast.block_until_ready()
        torch.cuda.synchronize()

        # TODO transposes here due to image layout conventions. can we get rid of them?
        t_raycast = t_raycast.transpose().flatten()
        hit_ids = hit_ids.transpose().flatten()
        counts = counts.transpose().flatten()

    elif tree_based:
        t_raycast, hit_ids, counts, n_eval = queries.cast_rays_tree_based(funcs_tuple, params_tuple, ray_roots,
                                                                          ray_dirs, batch_size=batch_size,
                                                                          enable_clipping=enable_clipping, load_from=load_from, save_to=save_to)
        torch.cuda.synchronize()
    elif shell_based:
        t_raycast, hit_ids, counts, n_eval = queries.cast_rays_shell_based(funcs_tuple, params_tuple, ray_roots,
                                                                          ray_dirs, batch_size=batch_size,
                                                                          load_from=load_from)
        torch.cuda.synchronize()
    else:
        # == Standard raycasting
        t_raycast, hit_ids, counts, n_eval = queries.cast_rays(funcs_tuple, params_tuple, ray_roots, ray_dirs, opts)
        torch.cuda.synchronize()

    hit_pos = ray_roots + t_raycast[:, None] * ray_dirs

    torch.cuda.empty_cache()

    hit_normals = outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, opts['hit_eps'])
    hit_color = shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple,
                            shading_color_func=shading_color_func)

    img = torch.where(hit_ids[:, None].bool(), hit_color, torch.ones((res * res, 3)))

    if tonemap:
        # We intentionally tonemap before compositing in the shadow. Otherwise the white level clips the shadow and gives it a hard edge.
        img = tonemap_image(img)

    img = img.reshape(res, res, 3)
    depth = t_raycast.reshape(res, res)
    counts = counts.reshape(res, res)
    hit_ids = hit_ids.reshape(res, res)

    return img, depth, counts, hit_ids, n_eval, -1

def render_image_mesh(funcs_tuple, params_tuple, load_from, eye_pos, look_dir, up_dir, left_dir, res, fov_deg, opts,
                      shading="normal", shading_color_tuple=((0.157, 0.613, 1.000)), matcaps=None, tonemap=False,
                      shading_color_func=None):
    if isinstance(funcs_tuple, list): funcs_tuple = tuple(funcs_tuple)
    if isinstance(params_tuple, list): params_tuple = tuple(params_tuple)
    if isinstance(shading_color_tuple, list): shading_color_tuple = tuple(shading_color_tuple)

    # wrap in tuples if single was passed
    if not isinstance(funcs_tuple, tuple):
        funcs_tuple = (funcs_tuple,)
    if not isinstance(params_tuple, tuple):
        params_tuple = (params_tuple,)
    if not isinstance(shading_color_tuple[0], tuple):
        shading_color_tuple = (shading_color_tuple,)

    L = len(funcs_tuple)
    if (len(params_tuple) != L) or (len(shading_color_tuple) != L):
        raise ValueError("render_image tuple arguments should all be same length")

    ray_roots, ray_dirs = generate_camera_rays(eye_pos, look_dir, up_dir, res=res, fov_deg=fov_deg)

    mesh_npz = np.load(load_from)
    vertices = mesh_npz['vertices'].astype(np.float32)
    faces = mesh_npz['faces'].astype(np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # mesh.show()
    intersector = RayMeshIntersector(mesh)
    # compiled_cast_rays_shell_based = torch.compile(queries.cast_rays_shell_based)
    compiled_cast_rays_shell_based = queries.cast_rays_shell_based

    # hit_pos, hit_ids, _, _ = queries.cast_rays_shell_based(funcs_tuple, params_tuple, ray_roots, ray_dirs, intersector)
    hit_pos, hit_ids, _, _ = compiled_cast_rays_shell_based(funcs_tuple, params_tuple, ray_roots, ray_dirs, intersector)
    ray_roots, ray_dirs = generate_camera_rays(eye_pos, look_dir, up_dir, res=res, fov_deg=fov_deg)
    hit_pos, hit_ids, _, _ = compiled_cast_rays_shell_based(funcs_tuple, params_tuple, ray_roots, ray_dirs, intersector)

    hit_normals = outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, opts['hit_eps'], method='autodiff')
    hit_color = shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple,
                            shading_color_func=shading_color_func)
    img = torch.where(hit_ids[:, None].bool(), hit_color, torch.ones((res * res, 3)))

    if tonemap:
        # We intentionally tonemap before compositing in the shadow. Otherwise the white level clips the shadow and gives it a hard edge.
        img = tonemap_image(img)

    img = img.reshape(res, res, 3)

    return img


def tonemap_image(img, gamma=2.2, white_level=.75, exposure=1.):
    img = img * exposure
    num = img * (1.0 + (img / (white_level * white_level)))
    den = (1.0 + img)
    img = num / den
    img = torch.pow(img, 1.0 / gamma)
    return img


def shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple,
                shading_color_func=None):
    # Simple shading
    if shading == "normal":
        hit_color = (hit_normals + 1.) / 2.  # map normals to [0,1]

    elif shading == "matcap_color":

        # compute matcap coordinates
        ray_up = vmap(partial(geometry.orthogonal_dir, up_dir))(ray_dirs)
        ray_left = vmap(torch.cross)(ray_dirs, ray_up)
        matcap_u = vmap(torch.dot)(-ray_left, hit_normals)
        matcap_v = vmap(torch.dot)(ray_up, hit_normals)

        # pull inward slightly to avoid indexing off the matcap image
        matcap_u *= .98
        matcap_v *= .98

        # remap to image indices 
        matcap_x = (matcap_u + 1.) / 2. * matcaps[0].shape[0]
        matcap_y = (-matcap_v + 1.) / 2. * matcaps[0].shape[1]
        matcap_coords = torch.stack((matcap_x, matcap_y), dim=0)

        def sample_matcap(matcap, coords):
            import torch.nn.functional as F
            def map_coordinates(input, coordinates, order=1, mode='nearest', cval=0.0):
                assert order == 1, "Only order=1 (linear interpolation) is supported."
                assert mode in ['nearest', 'constant'], "Only 'nearest' and 'constant' modes are supported."

                def get_pixel_value(img, x, y, mode, cval):
                    if mode == 'nearest':
                        x = torch.clamp(x, 0, img.shape[0] - 1)
                        y = torch.clamp(y, 0, img.shape[1] - 1)
                        return img[x.long(), y.long()]
                    elif mode == 'constant':
                        mask = (x >= 0) & (x < img.shape[0]) & (y >= 0) & (y < img.shape[1])
                        x = torch.clamp(x, 0, img.shape[0] - 1)
                        y = torch.clamp(y, 0, img.shape[1] - 1)
                        return torch.where(mask, img[x.long(), y.long()], torch.tensor(cval, dtype=img.dtype))

                x_coords = coordinates[0]
                y_coords = coordinates[1]

                x0 = torch.floor(x_coords).long()
                x1 = x0 + 1
                y0 = torch.floor(y_coords).long()
                y1 = y0 + 1

                x0 = x0.float()
                x1 = x1.float()
                y0 = y0.float()
                y1 = y1.float()

                Ia = get_pixel_value(input, x0, y0, mode, cval)
                Ib = get_pixel_value(input, x0, y1, mode, cval)
                Ic = get_pixel_value(input, x1, y0, mode, cval)
                Id = get_pixel_value(input, x1, y1, mode, cval)

                wa = (x1 - x_coords) * (y1 - y_coords)
                wb = (x1 - x_coords) * (y_coords - y0)
                wc = (x_coords - x0) * (y1 - y_coords)
                wd = (x_coords - x0) * (y_coords - y0)

                output = wa * Ia + wb * Ib + wc * Ic + wd * Id
                return output

            # m = lambda X : scipy.ndimage.map_coordinates(X, coords.cpu(), order=1, mode='nearest')
            m = lambda X: map_coordinates(X, coords, order=1, mode='nearest')
            return vmap(m, in_dims=-1, out_dims=-1)(matcap)

        # fetch values
        mat_r = sample_matcap(matcaps[0], matcap_coords)
        mat_g = sample_matcap(matcaps[1], matcap_coords)
        mat_b = sample_matcap(matcaps[2], matcap_coords)
        mat_k = sample_matcap(matcaps[3], matcap_coords)

        # find the appropriate shading color
        def get_shade_color(hit_pos, hit_id):
            shading_color = torch.ones(3)

            if shading_color_func is None:
                # use the tuple of constant colors
                i_func = 1
                for c in shading_color_tuple:
                    shading_color = torch.where(hit_id == i_func, torch.tensor(c), shading_color)
                    i_func += 1
            else:
                # look up varying color
                shading_color = shading_color_func(hit_pos)

            return shading_color

        shading_color = vmap(get_shade_color)(hit_pos, hit_ids)

        c_r, c_g, c_b = shading_color[:, 0], shading_color[:, 1], shading_color[:, 2]
        c_k = 1. - (c_r + c_b + c_g)

        c_r = c_r[:, None]
        c_g = c_g[:, None]
        c_b = c_b[:, None]
        c_k = c_k[:, None]

        hit_color = c_r * mat_r + c_b * mat_b + c_g * mat_g + c_k * mat_k

    else:
        raise RuntimeError("Unrecognized shading parameter")

    return hit_color


# create camera parameters looking in a direction
def look_at(eye_pos, target=None, up_dir='y'):
    if target == None:
        target = torch.tensor((0., 0., 0.,))
    if up_dir == 'y':
        up_dir = torch.tensor((0., 1., 0.,))
    elif up_dir == 'z':
        up_dir = torch.tensor((0., 0., 1.,))

    look_dir = geometry.normalize(target - eye_pos)
    up_dir = geometry.orthogonal_dir(up_dir, look_dir)
    left_dir = torch.cross(look_dir, up_dir)

    return look_dir, up_dir, left_dir


def load_matcap(fname_pattern):
    imgs = []
    for c in ['r', 'g', 'b', 'k']:
        im = imageio.imread(fname_pattern.format(c))
        im = torch.tensor(im) / 256.
        imgs.append(im)

    return tuple(imgs)
