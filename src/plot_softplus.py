import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import softplus


def plot_softplus(start=-10, end=10, beta=1, threshold=20, num_points=1000):
    """
    Plots the PyTorch softplus function over a user-defined interval.

    Parameters:
        start (float): Start of the interval.
        end (float): End of the interval.
        beta (float): Beta parameter for the softplus function.
        threshold (float): Threshold parameter for PyTorch's softplus.
        num_points (int): Number of points to sample in the interval.
    """
    # Generate the x values
    x = torch.linspace(start, end, steps=num_points)

    # Compute the softplus values using PyTorch
    # y = softplus(x, beta=beta, threshold=threshold)
    y = softplus(x * beta, beta=1, threshold=threshold) / beta

    # Convert to NumPy for plotting
    x_np = x.numpy()
    y_np = y.numpy()

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x_np, y_np, label=f'Softplus (beta={beta}, threshold={threshold})', color='blue')
    plt.title('Softplus Function (PyTorch)', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('Softplus(x)', fontsize=14)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.show()


# Example usage: Adjust the interval, beta, and threshold as needed
plot_softplus(start=-50, end=50, beta=100, threshold=20, num_points=1000)
