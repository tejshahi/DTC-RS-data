import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import KLDivergence



# Include the model function here or import if in another file
from tae import temporal_autoencoder
from util import plot_time_series_samples, plot_reconstruction
from TSClusteringLayer import TSClusteringLayer

# Parameters
n_samples = 200
timesteps = 16
input_dim = 3

# Create time vector
x = np.linspace(0, 2 * np.pi, timesteps)

# Generate synthetic time series for each sample
import numpy as np

# Parameters
n_samples = 300
timesteps = 16
input_dim = 3
x = np.linspace(0, 2 * np.pi, timesteps)

# Define 3 cluster-generating functions
def cluster_1_pattern(x):  # e.g. Sine wave-based
    return np.stack([
        np.sin(x),
        np.sin(2 * x),
        np.sin(0.5 * x)
    ], axis=1)

def cluster_2_pattern(x):  # e.g. Cosine wave-based with phase shift
    return np.stack([
        np.cos(x + np.pi / 4),
        np.cos(2 * x + np.pi / 4),
        np.cos(0.5 * x + np.pi / 4)
    ], axis=1)

def cluster_3_pattern(x):  # e.g. Mixed sine-cosine with noise
    return np.stack([
        np.sin(x) + 0.1 * np.random.randn(*x.shape),
        np.cos(2 * x) + 0.1 * np.random.randn(*x.shape),
        np.sin(0.5 * x + 1)
    ], axis=1)

# Assign samples to clusters
data = []
labels = []

for i in range(n_samples):
    cluster_id = np.random.choice([0, 1, 2])
    if cluster_id == 0:
        sample = cluster_1_pattern(x)
    elif cluster_id == 1:
        sample = cluster_2_pattern(x)
    else:
        sample = cluster_3_pattern(x)
    data.append(sample)
    labels.append(cluster_id)

data = np.array(data)       # shape: (n_samples, timesteps, input_dim)
labels = np.array(labels)   # shape: (n_samples,)

print("Data shape:", data.shape)
print("Label distribution:", np.bincount(labels))


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

# Number of clusters
n_clusters = 3

# Input to clustering model is the same as encoder input
clustering_input = Input(shape=(16, 3))  # (timesteps, input_dim)

# Encode the input
latent_output = encoder(clustering_input)

# Cluster assignments
clustering_layer = TSClusteringLayer(n_clusters=n_clusters, name='clustering')(latent_output)

# Define the clustering model
full_model = Model(inputs=clustering_input, outputs=[autoencoder(clustering_input), clustering_layer])
full_model.compile(optimizer=Adam(0.001), loss=['mse', 'kld'], loss_weights=[1.0, 0.1])


#train the model with cust0m loop
# 1. Get soft assignments for initialization (optional)
q = full_model.predict(data)[1]

# 2. Initialize target distribution (used for KLD loss)
def target_distribution(q):
    weight = q ** 2 / np.sum(q, axis=0)
    return (weight.T / np.sum(weight, axis=1)).T

# 3. Training loop
epochs = 50
batch_size = 32

for epoch in range(epochs):
    # Step 1: Predict soft assignments
    q = full_model.predict(data)[1]

    # Step 2: Compute target distribution
    p = target_distribution(q)

    # Step 3: Train model to minimize reconstruction loss + clustering KL divergence
    full_model.fit(data, [data, p],
                   batch_size=batch_size,
                   epochs=1,
                   verbose=1)

    print(f"Epoch {epoch + 1}/{epochs} completed.")

# Predict the soft assignments (cluster probabilities)
q_new = full_model.predict(data)[1]

# Cluster assignments for new data (index of the maximum probability in each row)
predicted_clusters = np.argmax(q_new, axis=1)

# Print the predicted clusters for new data
print("Predicted clusters for new data:", predicted_clusters)

#print the cluster centroids
c1_indices=np.where(predicted_clusters==0)[0]
c2_indices=np.where(predicted_clusters==1)[0]
c3_indices=np.where(predicted_clusters==2)[0]

plot_time_series_samples(data[c1_indices])
plot_time_series_samples(data[c2_indices])
plot_time_series_samples(data[c3_indices])