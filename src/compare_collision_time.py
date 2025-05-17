import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Font config
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'font.weight': 'regular',
    'axes.labelweight': 'regular'
})

object_names = ["fox", "cat", "tree", "koala"]

methods = {
    "SDF only": [],
    "With shell": [],
    "With bbox": []
}

# Collect per-object times
for object_name in object_names:
    sdf = np.load(f'exp_results/collision/time_{object_name}.npz')['time']
    shell = np.load(f'exp_results/collision/time_with_shell_{object_name}.npz')['time']
    bbox = np.load(f'exp_results/collision/time_with_bbox_{object_name}.npz')['time']

    methods["SDF only"].append(sdf)
    methods["With shell"].append(shell)
    methods["With bbox"].append(bbox)

# Print table header
header = f"{'Method':<15}" + "".join([f"{name:^20}" for name in object_names])
print(header)
print('-' * len(header))

# Print each method row with mean ± 2*std
for method, times in methods.items():
    row = f"{method:<15}"
    for t in times:
        mean = t.mean()
        std = 2 * t.std()
        row += f"{mean:.2f} ± {std:.2f}".center(20)
    print(row)
