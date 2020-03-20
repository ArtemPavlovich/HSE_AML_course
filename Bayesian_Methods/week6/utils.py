import subprocess
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Lambda, InputLayer, concatenate, Activation, Flatten, Reshape
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential


def create_encoder(input_dims, base_filters=64, layers=2, latent=512):
    w = input_dims[0] // 2**layers
    h = input_dims[1] // 2**layers
    c = base_filters * 2**(layers - 1)
    encoder = Sequential()
    encoder.add(InputLayer(input_dims))
    for i in range(layers):
        encoder.add(Conv2D(filters=base_filters * 2**i, kernel_size=(5, 5),
                           strides=(2, 2), padding='same', use_bias=False, activation='relu'))
#         encoder.add(BatchNormalization(axis=3))
    encoder.add(Reshape([w * h * c]))
    encoder.add(Dense(latent * 2))
    return encoder


def create_decoder(output_dims, base_filters=64, layers=2, latent=512):
    w = output_dims[0] // 2**layers
    h = output_dims[1] // 2**layers
    c = base_filters * 2**(layers - 1)
    decoder = Sequential()
    decoder.add(InputLayer((latent)))
    decoder.add(Dense(w * h * c))
    decoder.add(Reshape([w, h, c]))
    for i in range(layers - 1, 0, -1):
        decoder.add(Conv2DTranspose(filters=base_filters * 2**i, kernel_size=(5, 5),
                                    strides=(2, 2), padding='same', use_bias=False, activation='relu'))
#         decoder.add(BatchNormalization(axis=3))
    decoder.add(Conv2DTranspose(filters=3, kernel_size=(5, 5),
                                strides=(2, 2), padding='same', activation='sigmoid'))
    return decoder


def sample(mean_log_var):
    mean, log_var = mean_log_var
    eps_shape = mean.shape
    epsilon = tf.random.normal(shape=eps_shape)
    z = epsilon * tf.math.exp(log_var / 2) + mean
    return z


def create_vae(batch_size, base_filters=128, latent=8,
               image_size=64, layers=4, reconstruction_weight=1, learning_rate=0.0001):
    """
    Constructs VAE model with given parameters.
    :param batch_size: size of a batch (used for placeholder)
    :param base_filters: number of filters after first layer.
        Other layers will double this number
    :param latent: latent space dimension
    :param image_size: size of input image
    Returns compiled Keras model along with encoder and decoder
    """


    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    x = Input(shape=(image_size[0], image_size[1], 3), batch_size=batch_size)
    encoder = create_encoder([image_size[0], image_size[1], 3],
                             base_filters=base_filters,
                             latent=latent,
                             layers=layers)
    decoder = create_decoder([image_size[0], image_size[1], 3],
                             base_filters=base_filters,
                             latent=latent,
                             layers=layers)
    mean_log_var = encoder(x)
    mean, log_var = tf.split(mean_log_var, num_or_size_splits=2, axis=1)
    z = Lambda(sample)([mean, log_var])
    reconstruction = decoder(z)

    vae = Model(x, reconstruction)

    class vlb_mse_kl(tf.keras.losses.Loss):

        def call(self, y_true, y_pred):
            t_mean, t_log_var = tf.split(
                encoder(y_true), num_or_size_splits=2, axis=1)
            log_likelihood = -0.5 * reconstruction_weight * \
                tf.reduce_sum((y_true - y_pred)**2, axis=[1, 2, 3])
            KL = 0.5 * tf.reduce_sum((-t_log_var + tf.math.square(t_mean) +
                                      tf.math.exp(t_log_var) - 1), axis=1)
            return -tf.reduce_mean((log_likelihood - KL))

    loss = vlb_mse_kl()

    vae.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                loss=loss)

    return vae, encoder, decoder


weights_options = {1:
    ['https://github.com/ArtemPavlovich/HSE_AML_course/raw/master/Bayesian_Methods/week6/VAE_celeba_img64_filt128_4layers_32lsize_3.h5',
    'VAE_celeba_img64_filt128_4layers_32lsize_3.h5'],
    2: ['https://github.com/ArtemPavlovich/HSE_AML_course/raw/master/Bayesian_Methods/week6/VAE_celeba_img64_filt256_4layers_16lsize_2.h5',
    'VAE_celeba_img64_filt256_4layers_16lsize_2.h5'],
    3: ['https://github.com/ArtemPavlovich/HSE_AML_course/raw/master/Bayesian_Methods/week6/VAE_celeba_img64_filt128_4layers_8lsize.h5',
        'VAE_celeba_img64_filt128_4layers_8lsize.h5']}


def parse_name(weight_name):
    split = weight_name.split('_')
    image_size = int(split[2][3:])
    base_filters = int(split[3][4:])
    layers = int(split[4][:-6])
    latent_size = int(split[5][:-5])
    return image_size, base_filters, layers, latent_size


def download_and_get_params(option=1):
    url = weights_options[option][0]
    filename = weights_options[option][1]

    params = parse_name(filename)

    subprocess.run(['wget', url, '-O', filename])

    return params, filename
