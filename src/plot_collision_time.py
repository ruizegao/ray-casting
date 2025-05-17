import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.rcParams['font.family'] = 'serif'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'font.weight': 'regular',
    'axes.labelweight': 'regular'
})


# without_shell = np.load('exp_results/collision/time_tree.npz')['time']
# with_shell = np.load('exp_results/collision/time_with_shell_tree.npz')['time']
# with_bbox = np.load('exp_results/collision/time_with_bbox_tree.npz')['time']

without_shell = np.load('assets/collision_detection_time.npz')['time']
with_shell = np.load('assets/collision_detection_time_with_shell.npz')['time']
with_bbox = np.load('assets/collision_detection_time_with_bbox.npz')['time']

plt.figure(figsize=(10, 4))
plt.plot(np.arange(400), without_shell, label='Without Shell', color='tab:red', linewidth=2)
plt.plot(np.arange(400), with_bbox, label='With BBox', color='tab:green', linewidth=2)  # New line
plt.plot(np.arange(400), with_shell, label='With Shell', color='tab:blue', linewidth=2)

# plt.axvline(x=35)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlim(left=0, right=400)
plt.ylim(bottom=0, top=20)
plt.legend(loc='right', fontsize=12, frameon=False)
plt.xlabel('Frame', fontsize=14)
plt.ylabel('Collision Detection\nTime (ms)', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.yticks(np.linspace(0, 20, 5))
plt.tight_layout()
plt.savefig('/home/ruize/3d_vnn_ref/collision_detection_time.pdf', dpi=600)
plt.show()
