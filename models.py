from tensorflow.keras import layers, Model, Input
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras import Input

def build_generator(latent_dim=100):
    input_latent = Input(shape=(latent_dim,), name="latent_input")

    x = layers.Dense(10*40*256)(input_latent)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    # Reshape to a small spatial dimension (10, 40, 1)
    x = layers.Reshape((10, 40, 256))(x)

    x = layers.Conv2DTranspose(128, kernel_size=4, strides = (1,2) ,padding='same')(x) 
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(64, kernel_size=4, strides=(1,2),padding='same')(x) 
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32, kernel_size=4,strides=(1,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(1 , kernel_size=4, strides=(1,2), padding='same', activation='tanh')(x)
    x = layers.BatchNormalization()(x)

    generator = Model(inputs=input_latent, outputs=x, name="Generator")
    return generator

def build_critic():
    input_data = Input(shape=(10, 640, 1), name="generated_or_real_input")

    # Reduce the number of filters to save memory
    x = layers.Conv2D(4, kernel_size=4, padding='same')(input_data)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(16, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(512, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    # Use Global Average Pooling to reduce memory usage
    x = layers.GlobalAveragePooling2D()(x)

    # Reduce Dense layer size
    x = layers.Dense(256, activation='relu')(x)

    # Output layer
    output = layers.Dense(2, activation='softmax')(x)

    critic = Model(inputs=input_data, outputs=output, name="Critic")
    return critic


def build_classifier():
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(filters=4, kernel_size=(8, 4), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=8, kernel_size=(16, 8), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=16, kernel_size=(32, 16), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=32, kernel_size=(64, 32), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.DepthwiseConv2D(kernel_size=(1, 64), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    
    x = layers.MaxPooling2D(pool_size=(4, 1))(x)

    x = layers.Dropout(0.5)(x)

    x = layers.SeparableConv2D(filters=16, kernel_size=(4, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=(8, 1))(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(24, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model