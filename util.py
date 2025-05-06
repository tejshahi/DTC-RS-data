import numpy as np
import matplotlib.pyplot as plt

# Plotting function
def plot_time_series_samples(data, n=5):
    timesteps = data.shape[1]
    input_dim = data.shape[2]
    indices = np.random.choice(data.shape[0], n, replace=False)

    plt.figure(figsize=(15, n * 2))

    for i, idx in enumerate(indices):
        for j in range(input_dim):
            plt.subplot(n, input_dim, i * input_dim + j + 1)
            plt.plot(data[idx, :, j])
            if i == 0:
                plt.title(f'Feature {j+1}')
            if j == 0:
                plt.ylabel(f'Sample {idx}')
            plt.grid(True)

    plt.tight_layout()
    plt.show()

#plot original and reconstructed time sereis from TAE
def plot_reconstruction(original, reconstructed, sample_index=0, feature_names=None):
    """
    Plots the original and reconstructed time series for one sample.

    Parameters:
        original (ndarray): Original data, shape (n_samples, timesteps, input_dim)
        reconstructed (ndarray): Reconstructed data, same shape as original
        sample_index (int): Index of the sample to plot
        feature_names (list[str] or None): Optional list of feature names
    """
    timesteps = original.shape[1]
    input_dim = original.shape[2]

    plt.figure(figsize=(15, 4))
    for i in range(input_dim):
        plt.subplot(1, input_dim, i + 1)
        plt.plot(original[sample_index, :, i], label='Original')
        plt.plot(reconstructed[sample_index, :, i], label='Reconstructed', linestyle='--')
        title = feature_names[i] if feature_names else f'Feature {i+1}'
        plt.title(title)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
