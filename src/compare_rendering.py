import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt

# Load arrays
loaded1 = np.load('rendering/hammer_grid_cam_baseline.npz')
loaded2 = np.load('rendering/hammer_grid_cam_both_shells.npz')
loaded3 = np.load('rendering/fox_grid_cam_de.npz')
arrays1 = [loaded1[f'arr_{i}'] for i in range(len(loaded1.files))]
arrays2 = [loaded2[f'arr_{i}'] for i in range(len(loaded2.files))]
arrays3 = [loaded3[f'arr_{i}'] for i in range(len(loaded3.files))]
arrays1 = arrays1[0]
arrays2 = arrays2[0]
arrays3 = arrays3[0]

# Compute PSNR, track the most different pair
ssim_values_12 = []
ssim_values_13 = []
psnr_values_12 = []
psnr_values_13 = []
max_diff_index = -1
max_psnr_diff = - float('inf')
min_psnr = float('inf')

for i, (img1, img2, img3) in enumerate(zip(arrays1, arrays2, arrays3)):
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch at index {i}: {img1.shape} vs {img2.shape}")

    # ssim_value_12 = ssim(img1, img2, channel_axis=-1, data_range=1.0)
    # ssim_value_13 = ssim(img1, img3, channel_axis=-1, data_range=1.0)
    # ssim_values_12.append(ssim_value_12)
    # ssim_values_13.append(ssim_value_13)

    psnr_value_12 = psnr(img1, img2, data_range=img1.max() - img1.min())
    psnr_value_13 = psnr(img1, img3, data_range=img1.max() - img1.min())
    psnr_values_12.append(psnr_value_12)
    psnr_values_13.append(psnr_value_13)
    psnr_diff = psnr_value_12 - psnr_value_13
    if psnr_diff > max_psnr_diff:
        max_psnr_diff = psnr_diff
        max_diff_index = i

# Show most different pair
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(arrays1[max_diff_index], cmap='gray')
plt.title(f'Baseline')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(arrays2[max_diff_index], cmap='gray')
plt.title(f'Ours (exact), PSNR = {psnr_values_12[max_diff_index]}')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(arrays3[max_diff_index], cmap='gray')
plt.title(f'Dilation-erosion, PSNR = {psnr_values_13[max_diff_index]}')
plt.axis('off')

plt.show()

# Results
print(f'Average PSNR 12: {np.mean(psnr_values_12):.2f}')
print(f'Average PSNR 13: {np.mean(psnr_values_13):.2f}')
# print(f'Average SSIM 12: {np.mean(ssim_values_12):.2f}')
# print(f'Average SSIM 13: {np.mean(ssim_values_13):.2f}')
# print(f'Most Different Pair Index: {max_diff_index}')
# print(f'Minimum PSNR: {min_psnr:.2f}')
