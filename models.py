from tensorflow.keras import layers, Model, Input
import tensorflow as tf


# models taken from paper
def build_generator(latent_dim=100):
    input_latent = Input(shape=(latent_dim,), name="latent_input")

    x = layers.Dense(256)(input_latent)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Reshape((64, 4))(x)  

    x = layers.Conv2D(10*4, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(10*16, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(10*64, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(10*256, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)



    x = layers.Conv2D(10*512, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)


    x = layers.Conv2D(640, kernel_size=4, padding='same', activation='tanh')(x)
    x = layers.BatchNormalization()(x)

    generator = Model(inputs=input_latent, outputs=x, name="Generator")
    return generator


def build_critic():
    input_data = Input(shape=(10, 640), name="generated_or_real_input")

    x = layers.Conv2D(10*4, kernel_size=4, padding='same')(input_data)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(10*16, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(10*64, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(10*256, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(10*512, kernel_size=4, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)  # Should be 64*64 = 4096
    x = layers.Dense(640, activation='relu')(x)
    output = layers.Dense(2, activation='softmax')(x)  # For classification if needed

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