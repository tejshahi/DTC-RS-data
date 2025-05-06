import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, LeakyReLU, MaxPool1D, LSTM, Bidirectional, TimeDistributed, Dense, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam
import matplotlib.pyplot as plt



def temporal_autoencoder(input_dim, timesteps, n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1]):
    """
    Temporal Autoencoder (TAE) model with Convolutional and BiLSTM layers.

    # Arguments
        input_dim: input dimension
        timesteps: number of timesteps (can be None for variable length sequences)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide time series length
        n_units: numbers of units in the two BiLSTM layers

    # Return
        (ae_model, encoder_model, decoder_model):  Full autoencoder and its components
    """
    assert(timesteps % pool_size == 0) # timesteps must be divisible by pool_size

    # Input
    x_input = Input(shape=(timesteps, input_dim), name='input_seq')

    # Encoder
    x = Conv1D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same', activation='linear')(x_input)
    x = LeakyReLU()(x)
    x = MaxPool1D(pool_size=pool_size)(x)
    x = Bidirectional(LSTM(n_units[0], return_sequences=True), merge_mode='concat')(x)
    x = LeakyReLU()(x)
    x = Bidirectional(LSTM(n_units[1], return_sequences=True), merge_mode='concat')(x)
    x = LeakyReLU(name='latent')(x)

    # Decoder
    x = TimeDistributed(Dense(n_filters))(x)
    x=LeakyReLU()(x)
    x = Reshape((-1, 1, n_filters))(x)
    x = UpSampling2D(size=(pool_size, 1))(x)  
    x = Conv2DTranspose(filters=input_dim, kernel_size=(kernel_size, 1), padding='same')(x)
    output = Reshape((-1, input_dim), name='output_seq')(x)

    # AE model
    autoencoder = Model(inputs=x_input, outputs=output, name='AE')

     # Encoder Model
    encoder = Model(inputs=x_input, outputs=autoencoder.get_layer('latent').output, name='encoder')

    # Decoder Model
    encoded_input = Input(shape=(timesteps // pool_size, 2 * n_units[1]), name='decoder_input')
    x = autoencoder.get_layer(index=8)(encoded_input)  # TimeDistributed(Dense)
    x = autoencoder.get_layer(index=9)(x)              # LeakyReLU
    x = autoencoder.get_layer(index=10)(x)             # Reshape
    x = autoencoder.get_layer(index=11)(x)             # UpSampling2D
    x = autoencoder.get_layer(index=12)(x)             # Conv2DTranspose
    decoder_output = autoencoder.get_layer('output_seq')(x)

    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_output, name='decoder')

    return autoencoder, encoder, decoder