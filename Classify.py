"""
This file classiffies the original data and the data generated 
by GAN and plots the performance on the data 

Author: Dan Shudreno
"""

from tensorflow.keras import layers, Model, Input
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras import Input
import argparse
import numpy as np
import models


def run_experimenet():
    pass

def main():
    ##checking what we are running
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--experiment_type")
    args = parser.parse_args()
    # Since we only need images from the dataset to encode and decode, we
    # won't use the labels.
    train_data, test_data = 

    # Normalize and reshape the data
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    #initializing the list of noises
    noises = [0.1, 0.28, 0.46, 0.64, 0.82, 1]

    if args.experiment_type == "train":
        #intializing df to store the histories 
        model_histories = pd.DataFrame()
        #training different models
        for noise_f in noises: 
            model_histories = model_train(train_data, test_data, noise_f, model_histories)
        #saving the history
        model_histories.to_csv("ahistories.csv", index = False)
    else: 
        if args.experiment_type == "standard":
            df = pd.read_csv("histories.csv")
            plot_loss(df, "Training")
            plot_loss(df, "Validation")
        elif args.experiment_type == "fashion":
            (train_data, _), (test_data, _) = fashion_mnist.load_data()
        #for each type of noise dispaying the image
        for noise_f in noises:
            #getting the model
            name = "model_" + str(noise_f)
            path = os.path.join("./models", name + ".keras")
            print(path)
            autoencoder = load_model(path)
            #making predictions
            noisy_test_data = noise(test_data, noise_f)
            predictions = autoencoder.predict(noisy_test_data)
            display(noisy_test_data, predictions, args.experiment_type + " " + name)


   

if __name__ =="__main__":
    main()


