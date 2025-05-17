import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'font.weight': 'regular',
    'axes.labelweight': 'regular'
})

from matplotlib.patches import Ellipse

object_name = "tree"

def load_from_npz(filename):
    loaded = np.load(filename)
    return loaded['arr_0']

# Load arrays
def get_res(object_name):
    sdf_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_sdf.npz')
    spelunking_images = load_from_npz(f'rendering/{object_name}_grid_cam_baseline.npz')
    gios_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_27_both_shells.npz')
    de_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_de.npz')
    gzl_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_mid_shell.npz')
    _0lvl_imgs = load_from_npz(f'rendering/{object_name}_grid_cam_0lvl.npz')

    # # Compute PSNR, track the most different pair
    # spelunking_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, spelunking_images)])
    # gios_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, gios_imgs)])
    # de_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, de_imgs)])
    # gzl_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, gzl_imgs)])
    # _0lvl_psnr = np.array([psnr(img1, img2) for img1, img2 in zip(sdf_imgs, _0lvl_imgs)])

    spelunking_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, spelunking_images)]) ** 0.5
    gios_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, gios_imgs)]) ** 0.5
    de_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, de_imgs)]) ** 0.5
    gzl_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, gzl_imgs)]) ** 0.5
    _0lvl_rmse = np.array([mse(img1.flatten(), img2.flatten()) for img1, img2 in zip(sdf_imgs, _0lvl_imgs)]) ** 0.5
    sdf_rmse = np.zeros(50)

    sdf_time = load_from_npz(f'exp_results/render/{object_name}/sdf_time.npz')
    spelunking_time = load_from_npz(f'exp_results/render/{object_name}/baseline_time.npz')
    print("spelunking time", spelunking_time.shape)
    both_shells_time = load_from_npz(f'exp_results/render/{object_name}/both_shells_time.npz')
    de_time = load_from_npz(f'exp_results/render/{object_name}/de_time.npz')
    mid_shell_time = load_from_npz(f'exp_results/render/{object_name}/mid_shell_time.npz')
    _0lvl_time = load_from_npz(f'exp_results/render/{object_name}/0lvl_time.npz')

    spelunking_results = np.vstack([1. / spelunking_time, spelunking_rmse, ]).transpose()
    both_shells_results = np.vstack([1. / both_shells_time, gios_rmse, ]).transpose()
    de_results = np.vstack([1. / de_time, de_rmse]).transpose()
    mid_shell_results = np.vstack([1. / mid_shell_time, gzl_rmse]).transpose()
    _0lvl_results = np.vstack([1. / _0lvl_time, _0lvl_rmse]).transpose()
    sdf_results = np.vstack([1. / sdf_time, sdf_rmse]).transpose()
    return sdf_results, spelunking_results, de_results, both_shells_results, _0lvl_results, mid_shell_results,


def plot_multiple_objects_row(objects_results_dict):
    n_objects = len(objects_results_dict)
    fig, axes = plt.subplots(1, n_objects, figsize=(10 * n_objects, 10), sharey=True)

    if n_objects == 1:
        axes = [axes]

    method_labels = [
        'Sphere Tracing', 'Spelunking the Deep', 'Adaptive Shells', 'GIOM (Ours)',
        '0 Level MC', 'GZL (Ours)',
    ]

    colors = [
        '#1f77b4',  # Sphere Tracing
        '#ff7f0e',  # Spelunking the Deep
        '#d62728',  # Adaptive Shells
        '#2ca02c',  # GIOS (Ours)
        '#8c564b',  # 0 Level MC
        '#9467bd',  # GZL (Ours)
    ]

    markers = [
        'o',  # Sphere Tracing
        'o',  # Spelunking
        'o',  # Adaptive Shells
        'D',  # GIOS (Ours)
        'o',  # 0 Level MC
        's',  # GZL (Ours)
    ]

    legend_handles = []

    for ax, (object_name, method_results) in zip(axes, objects_results_dict.items()):
        for i, samples in enumerate(method_results):
            color = colors[i]
            marker = markers[i]
            mean = np.mean(samples, axis=0)
            handle = ax.scatter([mean[0]], [mean[1]], s=1600, alpha=.8,
                                color=color, marker=marker, edgecolor='black',
                                linewidths=1.5, label=method_labels[i])
            if len(legend_handles) < len(method_labels):
                legend_handles.append(handle)

        ax.axvline(x=20., color='r', linestyle='--', linewidth=2)
        ax.set_title(object_name.capitalize(), fontsize=40)
        ax.set_xscale('log')
        ax.set_xlim(0.5e-1, 3.5e2)
        ax.set_ylim(-0.003, 0.033)
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.grid(True, linestyle='--')
        ax.set_xlabel('FPS', fontsize=40)

    axes[0].set_ylabel('RMSE', fontsize=40)

    fig.legend(legend_handles, method_labels, loc='upper center', ncol=len(method_labels),
               fontsize=40, frameon=False, columnspacing=1.2, bbox_to_anchor=(0.5, 1.07))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('assets/images/rmse_fps/multi_object_row.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# spelunking_results = np.vstack([1. / spelunking_time, spelunking_psnr, ]).transpose()
# both_shells_results = np.vstack([1. / both_shells_time, gios_psnr, ]).transpose()
# de_results = np.vstack([1. / de_time, de_psnr]).transpose()
# mid_shell_results = np.vstack([1. / mid_shell_time, gzl_psnr]).transpose()
# _0lvl_results = np.vstack([1. / _0lvl_time, _0lvl_psnr]).transpose()
#
# spelunking_results = np.vstack([1. / spelunking_time, spelunking_rmse, ]).transpose()
# both_shells_results = np.vstack([1. / both_shells_time, gios_rmse, ]).transpose()
# de_results = np.vstack([1. / de_time, de_rmse]).transpose()
# mid_shell_results = np.vstack([1. / mid_shell_time, gzl_rmse]).transpose()
# _0lvl_results = np.vstack([1. / _0lvl_time, _0lvl_rmse]).transpose()
# sdf_results = np.vstack([1. / sdf_time, sdf_rmse]).transpose()

object_results_dict = {
    "fox": [
        *get_res("fox")
    ],
    "cat": [
        *get_res("cat")
    ],
    "koala": [
        *get_res("koala")
    ],
    "tree": [
        *get_res("tree")
    ],
}

plot_multiple_objects_row(object_results_dict)