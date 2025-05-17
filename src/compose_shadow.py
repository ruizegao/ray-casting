import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load images and convert to RGBA
img1 = Image.open("tree_front.png").convert("RGBA")
img2 = Image.open("tree_inner_shadow.png").convert("RGBA")

# Convert to float arrays
arr1 = np.array(img1).astype(np.float32) / 255.0
arr2 = np.array(img2).astype(np.float32) / 255.0

# Separate channels
rgb1, alpha1 = arr1[..., :3], arr1[..., 3:]
rgb2, alpha2 = arr2[..., :3], arr2[..., 3:]

# Create mask where foreground alpha > 0
mask = alpha1[..., 0] > 0  # Shape: (H, W)

# Prepare output arrays
out_rgb = rgb2.copy()
out_alpha = alpha2.copy()

# Get only masked values
a1 = alpha1[mask].reshape(-1, 1)  # (N, 1)
a2 = alpha2[mask].reshape(-1, 1)
c1 = rgb1[mask]  # (N, 3)
c2 = rgb2[mask]

# Alpha blending
blended_alpha = a1 + a2 * (1 - a1)
blended_rgb = (c1 * a1 + c2 * a2 * (1 - a1)) / np.clip(blended_alpha, 1e-6, 1)

# Apply blended values back
out_rgb[mask] = blended_rgb
out_alpha[mask] = blended_alpha

# Stack and convert back to uint8
output = np.dstack((out_rgb, out_alpha)) * 255
output = np.clip(output, 0, 255).astype(np.uint8)

# Plot and save using matplotlib
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(output)
ax.axis("off")
plt.tight_layout()
fig.savefig("/home/ruize/3d_vnn_ref/tree_w_shadow.pdf", dpi=300)
plt.close(fig)
