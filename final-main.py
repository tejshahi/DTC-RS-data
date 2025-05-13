import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from tae import temporal_autoencoder  # Import your TAE model builder
from util import plot_time_series_samples  # Utility to plot samples
from TSClusteringLayer import TSClusteringLayer  # Custom clustering layer

def load_time_series_data(csv_paths, skip_columns=3, label_column='label'):
    dataframes = [pd.read_csv(path, header=0) for path in csv_paths]
    labels = dataframes[0][label_column].values if label_column in dataframes[0] else None
    features = [df.iloc[:, skip_columns:].values for df in dataframes]

    if not all(f.shape == features[0].shape for f in features):
        raise ValueError("All feature CSVs must have the same shape after skipping columns.")

    data = np.stack(features, axis=-1)
    return data, labels

def target_distribution(q):
    weight = q ** 2 / np.sum(q, axis=0)
    return (weight.T / np.sum(weight, axis=1)).T

def main(args):
    data, labels = load_time_series_data(csv_paths=args.csv_paths)
    labels = np.round(labels).astype(int)
    #sample_limit = args.sample_limit
    #data = data[:sample_limit]
    print(f"Loaded data shape: {data.shape}, labels shape: {labels.shape}")
    labels = labels[:sample_limit]
    n_samples, timesteps, input_dim = data.shape
    numeric_labels = LabelEncoder().fit_transform(labels)

    # === Build the autoencoder model ===
    autoencoder, encoder, decoder = temporal_autoencoder(
        input_dim=input_dim,
        timesteps=timesteps,
        n_filters=args.n_filters,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        n_units=args.n_units
    )

    # === Load or Train Autoencoder ===
    if args.load_ae and os.path.exists(args.ae_weights_path):
        autoencoder.load_weights(args.ae_weights_path)
        print(f"Loaded autoencoder weights from: {args.ae_weights_path}")
    else:
        autoencoder.compile(optimizer=Adam(args.lr), loss='mse')
        print("Training autoencoder...")
        print(autoencoder.summary())
        autoencoder.fit(data, data, epochs=args.pretrain_epochs, batch_size=args.batch_size, verbose=1)
        autoencoder.save_weights(args.ae_weights_path)
        print(f"Saved autoencoder weights to: {args.ae_weights_path}")

    # === Load models if available ===
    if args.load_models and os.path.exists(args.encoder_path) and os.path.exists(args.full_model_path):
        encoder = tf.keras.models.load_model(args.encoder_path, compile=False)
        full_model = tf.keras.models.load_model(args.full_model_path, compile=False)
        print(f"Loaded encoder from {args.encoder_path}")
        print(f"Loaded full clustering model from {args.full_model_path}")
    else:
        # === Step 1: Encode data ===
        latent_data = encoder.predict(data)
        print("Shape of latent data", latent_data.shape) # (n_samples, latent_dim))
        latent_shape = latent_data.shape[1:]
        latent_data_flat = latent_data.reshape((latent_data.shape[0], -1))
        print("Shape of latent data flat", latent_data_flat.shape) # (n_samples, latent_dim * timesteps)
        # === Step 2: Run KMeans ===
        kmeans = KMeans(n_clusters=args.n_clusters, n_init='auto', random_state=42)
        kmeans.fit(latent_data_flat)
        cluster_centers = kmeans.cluster_centers_
       
        # Reshape to match (n_clusters, 8, 2)
        cluster_centers = cluster_centers.reshape((args.n_clusters, *latent_shape))

        # === Step 3: Build clustering model ===
        clustering_layer = TSClusteringLayer(n_clusters=args.n_clusters, weights=[cluster_centers], name='clustering')
        clustering_input = Input(shape=(timesteps, input_dim))
        latent_output = encoder(clustering_input)
        clustering_output = clustering_layer(latent_output)

        full_model = Model(inputs=clustering_input, outputs=[autoencoder(clustering_input), clustering_output])
        full_model.compile(optimizer=Adam(args.lr), loss=['mse', 'kld'], loss_weights=[1.0, 0.1])

        # === Step 4: Train with clustering loss ===
        for epoch in range(args.finetune_epochs):
            q = full_model.predict(data, batch_size=args.batch_size)[1]
            p = target_distribution(q)
            full_model.fit(data, [data, p], batch_size=args.batch_size, epochs=1, verbose=1)
            print(f"Epoch {epoch+1}/{args.finetune_epochs} completed.")

        # === Save models ===
        encoder.save(args.encoder_path)
        full_model.save(args.full_model_path)
        print(f"Encoder saved to: {args.encoder_path}")
        print(f"Full model saved to: {args.full_model_path}")

    # === Predict & visualize clusters ===
    q_final = full_model.predict(data)[1]
    final_clusters = np.argmax(q_final, axis=1)
    print("Final cluster assignments:", final_clusters)
    for i in range(args.n_clusters):
        print(f"\nCluster {i} samples:")
        cluster_indices = np.where(final_clusters == i)[0]
        plot_time_series_samples(data[cluster_indices])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Temporal Clustering Pipeline")

    # === Data & Model I/O ===
    parser.add_argument('--csv_paths', nargs='+', default=['green_data.csv', 'bare_data.csv','non-green_data.csv'], help='List of CSV file paths')
    parser.add_argument('--ae_weights_path', type=str, default='autoencoder.weights.h5', help='Path to save/load autoencoder weights')
    parser.add_argument('--encoder_path', type=str, default='encoder_model.h5', help='Path to save/load encoder model')
    parser.add_argument('--full_model_path', type=str, default='clustering_model.h5', help='Path to save/load full clustering model')

    # === Flags ===
    parser.add_argument('--load_ae', action='store_true', help='Load existing autoencoder weights')
    parser.add_argument('--load_models', action='store_true', help='Load existing encoder and full clustering model')

    # === Model Architecture ===
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--pool_size', type=int, default=2)
    parser.add_argument('--n_units', type=int, nargs='+', default=[50, 1], help='List of units for LSTM layers')

    # === Clustering & Training ===
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sample_limit', type=int, default=1000, help='Limit the number of samples for quick prototyping')

    args = parser.parse_args()
    main(args)
