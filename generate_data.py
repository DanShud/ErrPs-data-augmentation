import argparse
import models
import tensorflow as tf 
from tensorflow.keras import layers, Model, Input
import os
import pandas as pd 

import time
import numpy as np
print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def generate_synthetic_data(num_samples, path_to_generator, noise_dim = 100):
    generator = tf.keras.models.load_model(path_to_generator)

    # generate array of noice 
    noise = tf.random.normal([num_samples, noise_dim])
    synthetic_data = generator.predict(noise)

    print(synthetic_data.shape)
    synthetic_data = tf.squeeze(synthetic_data,axis = -1) # remove channel 
    print(type(synthetic_data ))
    return synthetic_data


def main():
    
    
    parser = argparse.ArgumentParser(
                    prog='ErrP Gan',
                    description='Generate and train positive or negative gan models')
    # hey pretty, add argument for data, and wether it is gan for pos or neg data add output
    parser.add_argument('-p', '--positive_gen',)      # option that takes a value
    parser.add_argument('-n', '--negative_gen')
    parser.add_argument('-o', '--output')

    args = parser.parse_args()
    pos_syn_data = generate_synthetic_data(10000,args.positive_gen)
    pos_syn_data = tf.reshape(pos_syn_data, (10000, 6400)).numpy()
    df = pd.DataFrame(pos_syn_data)  # Create a DataFrame
    df.to_csv(f'{args.output}/generated_data_pos.csv', index=False)  # Save as CSV
    neg_syn_data = generate_synthetic_data(30000,args.negative_gen)
    neg_syn_data = tf.reshape(neg_syn_data, (30000, 6400)).numpy()
    df = pd.DataFrame(neg_syn_data)  # Create a DataFrame
    df.to_csv(f'{args.output}/generated_data_neg.csv', index=False)  # Save as CSV




if __name__ == "__main__":
    main()