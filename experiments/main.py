import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data.data_loader import DUKE_DATA_NAME, load_dataset
from models.Sampling_Layer import Sampling
from models.VAE import VAE


def get_encoder():
    latent_dim = 2
    encoder_inputs = tf.keras.Input(shape=(512, 100))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def get_decoder():
    latent_dim = 2
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    width = 512
    height = 100
    scale = 1.0
    figure = np.zeros((width * n, height * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(width, height)
            figure[
                i * width : (i + 1) * width,
                j * height : (j + 1) * height,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range_w = width // 2
    end_range_w = n * width + start_range_w
    pixel_range_w = np.arange(start_range_w, end_range_w, width)
    start_range_h = width // 2
    end_range_h = n * width + start_range_h
    pixel_range_h = np.arange(start_range_h, end_range_h, width)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range_w, sample_range_x)
    plt.yticks(pixel_range_h, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


def get_data():
    dataset = load_dataset(DUKE_DATA_NAME)
    return dataset.train.features

def main():
    train_data = get_data()
    print(train_data.shape)
    vae = VAE(get_encoder(), get_decoder())
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    vae.fit(train_data, epochs=30, batch_size=128, verbose=1)


if __name__ == '__main__':
    main()
