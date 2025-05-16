"""
This file classiffies the original data and the data generated 
by GAN and plots the performance on the data 

"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import regularizers

from models import build_classifier

import pandas as pd
import os
import argparse 
import time


MODELS_PATH = os.path.join(".", "models")
os.makedirs(MODELS_PATH, exist_ok=True)
IMAGE_PATH = os.path.join(".", "images")
os.makedirs(IMAGE_PATH, exist_ok = True)



# def build_classifier():
#     input_shape = (10, 640, 1)
#     inputs = tf.keras.Input(shape=input_shape)

#     x = layers.Conv2D(filters=4, kernel_size=(8, 4), activation='relu', padding='same')(inputs)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv2D(filters=8, kernel_size=(16, 8), activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv2D(filters=16, kernel_size=(32, 16), activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.DepthwiseConv2D(kernel_size=(1, 32), activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)

    
#     x = layers.MaxPooling2D(pool_size=(4, 1), padding='same')(x)

#     x = layers.Dropout(0.5)(x)

#     x = layers.SeparableConv2D(filters=16, kernel_size=(4, 1), activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.MaxPooling2D(pool_size=(8, 1), padding='same')(x)

#     x = layers.Dropout(0.5)(x)

#     x = layers.Flatten()(x)

#     x = layers.Dense(24, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)

#     x = layers.Dropout(0.5)(x)

#     outputs = layers.Dense(2, activation='softmax')(x)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

def run_experimenet(train_data, x_valid, y_valid, gan, epochs):
    model = build_classifier()
    model.compile(loss = "sparse_categorical_crossentropy", optimizer='adam', metrics = ["accuracy"] )
    #Setting up the callback
    checkpoint_filepath = os.path.join(MODELS_PATH, "checkpoint_" + gan + "_" +str(epochs))
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True, 
        save_format="keras"  
    )
    #training the model
    history = model.fit(
        train_data,
        epochs = epochs,
        validation_data=(x_valid, y_valid), 
        callbacks = [model_checkpoint_callback]
    )
    #saving the model
    path = os.path.join(MODELS_PATH, "final_" + gan + "_" + str(epochs)+ ".keras")
    model.save(path)
    
    #saving models history
    model_histories = pd.concat([
        pd.Series(history.history["loss"], name = "Training Loss"),
        pd.Series(history.history["val_loss"], name = "Validation Loss"),
        pd.Series(history.history["accuracy"], name = "Training Accuracy"),
        pd.Series(history.history["val_accuracy"], name = "Validation Accuracy") ], axis = 1)
    
    path = os.path.join(MODELS_PATH, "histories" + gan + str(epochs)+ ".csv")
    model_histories.to_csv(path, index = True)

def plot_loss(name, title):
    path = os.path.join(MODELS_PATH, name)
    df = pd.read_csv(path)
    #Selecting colums of interest
    col_of_interest = ['Validation Loss']
    #adding the data
    plt.figure(figsize=(10, 6))
    for col in col_of_interest: 
        plt.plot(df[col], label= col)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(loc='upper center', ncol=3, fancybox=True)
    path = os.path.join(IMAGE_PATH, title + ".svg")
    plt.savefig(path, format = "svg")
    plt.clf()


def main():
    #setting the path for the data
    path_real = "./DATA/features/all_data.csv"
    path_gan_p = "./DATA/generated_data/real_syn_data.csv"
    path_gan_n = "./DATA/generated_data/neg_syn_data.csv"
    ##checking what we are running
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--experiment_type")
    parser.add_argument("-g", "--gan", action="store_true")
    parser.add_argument("-e", "--epochs", default=1)
    args = parser.parse_args()
    epochs = int(args.epochs)

    if args.experiment_type == "train":
        model_histories = pd.DataFrame()
        #reading actual data and preprocessing it
        data = np.loadtxt(path_real, delimiter=',')
        np.random.seed(42)
        np.random.shuffle(data)
        size_train = int(data.shape[0] * 0.7)        
        size_v = int(data.shape[0]*0.8) 
        x_train_real = data[:size_train, 3:].reshape(-1, 10, 640, 1)
        y_train_real = data[:size_train, 0]

        x_valid = data[size_train:size_v, 3:].reshape(-1, 10, 640, 1)
        y_valid = data[size_train:size_v, 0]

        # Build tf.data.Dataset from real data
        ds_real = tf.data.Dataset.from_tensor_slices((x_train_real, y_train_real))
                
        train_dataset = ds_real
        if args.gan:
            #load GAN data
            x_gan_pos = np.loadtxt(path_gan_p, delimiter=',').reshape(-1, 10, 640, 1)
            y_gan_pos = np.ones(x_gan_pos.shape[0])

            x_gan_neg = np.loadtxt(path_gan_n, delimiter=',').reshape(-1, 10, 640, 1)
            y_gan_neg = np.zeros(x_gan_neg.shape[0])

            #convert GAN data to tf.data.Datasets
            ds_gan_pos = tf.data.Dataset.from_tensor_slices((x_gan_pos, y_gan_pos))
            ds_gan_neg = tf.data.Dataset.from_tensor_slices((x_gan_neg, y_gan_neg))

            #merge all datasets
            train_dataset = ds_real.concatenate(ds_gan_pos).concatenate(ds_gan_neg)
            gan_label = "gan"
        else:
            train_dataset = ds_real
            gan_label = "real"
        
        batch_size = 64

        # After merging datasets
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        run_experimenet(train_dataset, x_valid, y_valid, gan_label, epochs)
    else: 
        name = "historiesreal51.csv"
        plot_loss(name, "Real Data CNN Training")
   

if __name__ =="__main__":
    main()


