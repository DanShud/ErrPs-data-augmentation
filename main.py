import argparse
import models
import tensorflow as tf 
from tensorflow.keras import layers, Model, Input
import os
import time
import numpy as np

BUFFER_SIZE = 60000
BATCH_SIZE = 32
EPOCHS = 300
noise_dim = 100

DATA_DIR = './'
POS = False
OUTPUT = './'

generator = models.build_generator()
critic = models.build_critic()
# ### followed : https://www.tensorflow.org/tutorials/generative/dcgan


def main():
    global DATA_DIR
    global POS
    global OUTPUT

    
    parser = argparse.ArgumentParser(
                    prog='ErrP Gan',
                    description='Generate and train positive or negative gan models')
    # hey pretty, add argument for data, and wether it is gan for pos or neg data add output
    parser.add_argument('-d', '--data',)      # option that takes a value
    parser.add_argument('-o', '--output')
    parser.add_argument('-p', '--pos', action='store_true')      # option that takes a value
    
    arguments = parser.parse_args()
    DATA_DIR = arguments.data
    POS = arguments.pos
    OUTPUT = arguments.output
    
    print(OUTPUT, DATA_DIR)
    os.makedirs(OUTPUT,exist_ok= True)
    print(OUTPUT, "exisits")

    # change train set 

    # main functions : 
    build_tfdataset()    
    neg_train_dataset, pos_train_dataset = extract_datasets()

    for i,l in neg_train_dataset.take(1):
        print(i)

    if POS:
        train(pos_train_dataset, EPOCHS)
        
    else:
        train(neg_train_dataset,EPOCHS)





# ## Data preprocessing

def serialize_example(signal, label):
    # Convert to tf.train.Example
    feature = {
        'signal': tf.train.Feature(float_list=tf.train.FloatList(value=signal.flatten())),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
def build_tfdataset():
    if ("tfrecords" not in os.listdir(DATA_DIR)): # no tf records
        os.mkdir(os.path.join(DATA_DIR,"tfrecords")) 

        tfrecords = os.path.join(DATA_DIR,"tfrecords")
        
        for idx, subject in enumerate(os.listdir(os.path.join(DATA_DIR,"data"))): # iterate over subjects
            print(f"{idx} of 91 ({subject})")
            with tf.io.TFRecordWriter(f'{tfrecords}/pos_{subject}.tfrecord') as pos_writer: # add tf_record for subject
                with tf.io.TFRecordWriter(f'{tfrecords}/neg_{subject}.tfrecord') as neg_writer: # add tf_record for subject

                    with open(tfrecords+"/../"+subject) as data_file:
                        for event in data_file.readlines():
                                event_data = np.array(event.split(","),dtype=float)
                                signal = np.array(event_data[3:])
                                label= int(event[0])
                                serialized = serialize_example(signal, label)
                                if label == 0:
                                    neg_writer.write(serialized)
                                else:
                                    pos_writer.write(serialized)


                        
                    
def parse_example(example_proto):
    # Define the structure of the data (must match how it was written!)
    feature_description = {
        'signal': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    
    # Parse the serialized data into a dictionary of tensors
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Return a tuple (signal, label)
    return parsed['signal'], parsed['label']
def extract_datasets():
    print("path", os.path.join(DATA_DIR,'tfrecords/pos_*.tfrecord'))
    raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(os.path.join(DATA_DIR,'tfrecords')+"/pos_*.tfrecord"))
    print("SIZE OF DATASET",len(list(raw_dataset)))
    # Parse before shuffling or batching
    parsed_dataset = raw_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    pos_train_dataset = parsed_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(os.path.join(DATA_DIR,'tfrecords/pos_*.tfrecord')))

    # Parse before shuffling or batching
    parsed_dataset = raw_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    neg_train_dataset = parsed_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return pos_train_dataset, neg_train_dataset





# ### Loss and optimizers


def critic_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output) 
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) 
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return cross_entropy(tf.ones_like(fake_output), fake_output) 



generator_optimizer = tf.keras.optimizers.Adam(1e-4)
critic_optimizer = tf.keras.optimizers.Adam(1e-4)

# ### Train utils




@tf.function
def train_step(images):
    global generator_optimizer 
    global critic_optimizer
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    images = tf.reshape(images, [-1, 10, 640,1])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      print(generated_images.shape)
      real_output = critic(images, training=True)

      fake_output = critic(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = critic_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, critic.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    critic_optimizer.apply_gradients(zip(gradients_of_discriminator, critic.trainable_variables))


# ## Training loop




def train(dataset, epochs):

    checkpoint_dir = os.path.join(OUTPUT,"training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=critic_optimizer,
                                    generator=generator,
                                    discriminator=critic)
    print("TRAINING HAS BEGUN")
    for epoch in range(epochs):
        start = time.time()


        for image_batch in dataset:
            train_step(image_batch[0]) # only need signal for gan traing

        # Produce images for the GIF as you go

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix =  str(epoch) +"_"+ checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        with open(os.path.join(OUTPUT,"gan_log.txt"),'a+') as f:
            f.write('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start) + "\n")

        generator.save(os.path.join(OUTPUT, "generator.keras"))
        critic.save(os.path.join(OUTPUT, "critic.keras"))



  




if __name__ == "__main__":
    main()