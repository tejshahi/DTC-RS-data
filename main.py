import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam


# Include the model function here or import if in another file
from tae import temporal_autoencoder
from util import plot_time_series_samples, plot_reconstruction

# Parameters
n_samples = 200
timesteps = 16
input_dim = 3

# Create time vector
x = np.linspace(0, 2 * np.pi, timesteps)

# Generate synthetic time series for each sample
data = np.array([
    np.stack([
        np.sin(x + np.random.rand()),
        np.cos(x + np.random.rand()),
        np.sin(2 * x + np.random.rand())
    ], axis=1)
    for _ in range(n_samples)
])

print("Data shape:", data.shape)  # (200, 100, 3)


#plot_time_series_samples(data, 5)

#compile the model
# Model parameters
n_filters = 32
kernel_size = 5
pool_size = 2
n_units = [32, 2]

# Create model
autoencoder, encoder, decoder = temporal_autoencoder(
    input_dim=input_dim,
    timesteps=timesteps,
    n_filters=n_filters,
    kernel_size=kernel_size,
    pool_size=pool_size,
    n_units=n_units
)

# Compile
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

# Model summary (optional)
print(autoencoder.summary())

history = autoencoder.fit(
    data, data,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

#plot the original and reconstruced time series
original=data[0:1]
reconstructed=autoencoder.predict(original)
plot_reconstruction(data,reconstructed)