# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# object_name = "tree"
#
# def load_from_npz(filename):
#     loaded = np.load(filename)
#     return loaded['arr_0']
#
# # Load arrays
# sdf_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_sdf.npz')
# spelunking_images = load_from_npz(f'rendering/{object_name}_grid_cam_baseline.npz')
# gios_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_27_both_shells.npz')
# de_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_de.npz')
# gzl_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_mid_shell.npz')
# _0lvl_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_0lvl.npz')
#
# # Compute PSNR, track the most different pair
# spelunking_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, spelunking_images)])
# gios_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, gios_imgs)])
# de_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, de_imgs)])
# gzl_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, gzl_imgs)])
# _0lvl_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, _0lvl_imgs)])
#
# spelunking_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, spelunking_images)]) ** 0.5
# gios_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, gios_imgs)]) ** 0.5
# de_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, de_imgs)]) ** 0.5
# gzl_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, gzl_imgs)]) ** 0.5
# _0lvl_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, _0lvl_imgs)]) ** 0.5
#
# sdf_time = load_from_npz(f'exp_results/render/{object_name}/sdf_time.npz')
# spelunking_time = load_from_npz(f'exp_results/render/{object_name}/baseline_time.npz')
# print("spelunking time", spelunking_time.shape)
# both_shells_time = load_from_npz(f'exp_results/render/{object_name}/both_shells_time.npz')
# de_time = load_from_npz(f'exp_results/render/{object_name}/de_time.npz')
# mid_shell_time = load_from_npz(f'exp_results/render/{object_name}/mid_shell_time.npz')
# _0lvl_time = load_from_npz(f'exp_results/render/{object_name}/0lvl_time.npz')
#
# spelunking_ssim = np.array([ssim(img1, img2, data_range=1.0) for img1, img2 in zip(sdf_imgs, spelunking_images)])
# gios_ssim = np.array([ssim(img1, img2, data_range=1.0) for img1, img2 in zip(sdf_imgs, gios_imgs)])
# de_ssim = np.array([ssim(img1, img2, data_range=1.0) for img1, img2 in zip(sdf_imgs, de_imgs)])
# gzl_ssim = np.array([ssim(img1, img2, data_range=1.0) for img1, img2 in zip(sdf_imgs, gzl_imgs)])
# _0lvl_ssim = np.array([ssim(img1, img2, data_range=1.0) for img1, img2 in zip(sdf_imgs, _0lvl_imgs)])
#
# spelunking_fps = 1. / spelunking_time
# both_shells_fps = 1. / both_shells_time
# de_fps = 1. / de_time
# gzl_fps = 1. / mid_shell_time
# _0lvl_fps = 1. / _0lvl_time
#
# def mean_2sigma(x):
#     mean = np.mean(x)
#     std = np.std(x)
#     return mean, 2 * std
#
# # Compute values
# psnr_stats = {
#     'dilation+erosion': mean_2sigma(de_psnr),
#     'GIOS': mean_2sigma(gios_psnr),
#     '0 level set': mean_2sigma(_0lvl_psnr),
#     'GZL': mean_2sigma(gzl_psnr),
# }
#
# fps_stats = {
#     'dilation+erosion': mean_2sigma(1. / de_time),
#     'GIOS': mean_2sigma(1. / both_shells_time),
#     '0 level set': mean_2sigma(1. / _0lvl_time),
#     'GZL': mean_2sigma(1. / mid_shell_time),
# }
#
# ssim_stats = {
#     'dilation+erosion': mean_2sigma(de_ssim),
#     'GIOS': mean_2sigma(gios_ssim),
#     '0 level set': mean_2sigma(_0lvl_ssim),
#     'GZL': mean_2sigma(gzl_ssim),
# }
#
# rmse_stats = {
#     'dilation+erosion': mean_2sigma(de_rmse),
#     'GIOS': mean_2sigma(gios_rmse),
#     '0 level set': mean_2sigma(_0lvl_rmse),
#     'GZL': mean_2sigma(gzl_rmse),
# }
#
#
# print("\nQuantitative Results Table\n")
# print(f"{'Metric':<12} & {'Dilation+Erosion':^20} & {'Both Shells':^20} & {'0 Level Set':^20} & {'Middle Shell':^20}")
# print("-" * 102)
#
# print(f"{'RMSE':<12} & " +
#       " & ".join([f"{rmse_stats[k][0]:.2f} ± {rmse_stats[k][1]:.2f}" for k in rmse_stats]))
# print(f"{'PSNR':<12} & " +
#       " & ".join([f"{psnr_stats[k][0]:.2f} ± {psnr_stats[k][1]:.2f}" for k in psnr_stats]))
# print(f"{'SSIM':<12} & " +
#       " & ".join([f"{ssim_stats[k][0]:.3f} ± {ssim_stats[k][1]:.3f}" for k in ssim_stats]))
# print(f"{'FPS':<12} & " +
#       " & ".join([f"{fps_stats[k][0]:.2f} ± {fps_stats[k][1]:.2f}" for k in fps_stats]))


import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse

def load_from_npz(filename):
    return np.load(filename)['arr_0']

def mean_2sigma(x):
    mean = np.mean(x)
    std = np.std(x)
    return mean, 2 * std

def evaluate_object(object_name):
    # Load image arrays
    sdf_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_sdf.npz')
    spelunking_images = load_from_npz(f'rendering/{object_name}_grid_cam_baseline.npz')
    gios_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_27_both_shells.npz')
    de_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_de.npz')
    gzl_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_mid_shell.npz')
    _0lvl_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_0lvl.npz')

    # Compute metrics
    metrics = {
        'SDF': {'fps': []}
    }
    for name, imgs in zip(['spelunking', 'GIOS', 'dilation+erosion', 'GZL', '0 level set'],
                          [spelunking_images, gios_imgs, de_imgs, gzl_imgs, _0lvl_imgs]):
        metrics[name] = {
            'psnr': np.array([psnr(ref, img) for ref, img in zip(sdf_imgs, imgs)]),
            'ssim': np.array([ssim(ref, img, data_range=1.0, channel_axis=-1) for ref, img in zip(sdf_imgs, imgs)]),
            'rmse': np.array([mse(ref.flatten(), img.flatten()) for ref, img in zip(sdf_imgs, imgs)]) ** 0.5
        }

    # Load timing
    times = {
        'spelunking': load_from_npz(f'exp_results/render/{object_name}/baseline_time.npz'),
        'GIOS': load_from_npz(f'exp_results/render/{object_name}/both_shells_time.npz'),
        'dilation+erosion': load_from_npz(f'exp_results/render/{object_name}/de_time.npz'),
        'GZL': load_from_npz(f'exp_results/render/{object_name}/mid_shell_time.npz'),
        '0 level set': load_from_npz(f'exp_results/render/{object_name}/0lvl_time.npz'),
        'SDF': load_from_npz(f'exp_results/render/{object_name}/sdf_time.npz')
    }

    for name in times:
        if name in metrics:
            metrics[name]['fps'] = 1.0 / times[name]

    return metrics

# === Main block ===
object_names = ["tree", "koala", "fox", "cat"]

methods = ['spelunking', 'dilation+erosion', '0 level set', 'GIOS', 'GZL', 'SDF']
aggregated = {method: {'psnr': [], 'ssim': [], 'rmse': [], 'fps': []} for method in methods}

for obj_name in object_names:
    metrics = evaluate_object(obj_name)
    for method in methods:
        for metric in aggregated[method]:
            if metric in metrics[method]:
                aggregated[method][metric].extend(metrics[method][metric])

# === Print Table ===
print("\nOverall Quantitative Results (All Objects)\n")
print(f"{'Metric':<12} & {'Spelunking':^20} & {'Dilation+Erosion':^20} & {'0 Level Set':^20} & {'GIOS':^20} & {'GZL':^20} & {'SDF':^20}")
print("-" * 147)

for metric in ['rmse', 'psnr', 'ssim', 'fps']:
    row = f"{metric.upper():<12} & "
    for method in methods:
        values = np.array(aggregated[method][metric])
        if values.size == 0 and method == 'SDF' and metric != 'fps':
            row += " ".ljust(20) + " & "
        else:
            m, s = mean_2sigma(values)
            if metric == 'ssim':
                row += f"{m:.3f} ± {s:.3f}".ljust(20) + " & "
            else:
                row += f"{m:.2f} ± {s:.2f}".ljust(20) + " & "
    print(row[:-3])  # Remove trailing ampersand
