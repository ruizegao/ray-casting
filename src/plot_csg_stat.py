import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Font configuration
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'font.weight': 'regular',
    'axes.labelweight': 'regular'
})

# Data
methods = ['Ours', 'CSG-nSDF']
metrics = ['Reconstr. time (min)  \u2193', 'Reconstr. error  \u2193', 'Query time (ms)  \u2193']
raw_data = [
    [1.0, 40.5],       # Reconstruction time
    [0.002, 0.030],    # Reconstruction error
    [54.6, 48.2]       # Query time
]
formats = ["{:.1f}", "{:.3f}", "{:.1f}"]

# Normalize each group individually so taller bar is height 1
normalized_data = []
for group in raw_data:
    max_val = max(group)
    normalized_group = [val / max_val for val in group]
    normalized_data.append(normalized_group)

# Plotting
def plot_grouped_bars(normalized_data, raw_data, labels, formats, filename):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    colors = ['#d62728', '#7f7f7f']  # red and gray
    n_groups = len(normalized_data)
    bar_width = 0.35
    x = np.arange(n_groups)

    # Offsets for side-by-side bars
    offsets = [-bar_width/2, bar_width/2]

    # Plot bars per method
    for i, method in enumerate(methods):
        heights = [normalized_data[j][i] for j in range(n_groups)]
        raw_vals = [raw_data[j][i] for j in range(n_groups)]
        fmt = [formats[j] for j in range(n_groups)]
        bars = ax.bar(x + offsets[i], heights, width=bar_width, color=colors[i], label=method)
        for xi, h, val, f, method in zip(x + offsets[i], heights, raw_vals, fmt, [method] * n_groups):
            ax.text(xi, h + 0.05,
                    f"{method}\n{f.format(val)}",
                    ha='center', va='bottom', fontsize=12)

    # Style
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticks([])  # Hide y ticks
    ax.set_yticklabels([])
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # ax.legend(fontsize=12, loc='upper center', ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close()


plot_grouped_bars(normalized_data, raw_data, metrics, formats,
    "/home/ruize/3d_vnn_ref/csg_grouped_normalized.pdf")
