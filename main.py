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
    parser.add_argument('-n', '--normalize', action='store_true')      # option that takes a value

    arguments = parser.parse_args()
    DATA_DIR = arguments.data
    POS = arguments.pos
    OUTPUT = arguments.output
    NORMALIZE = arguments.normalize
    
    print(OUTPUT, DATA_DIR)
    os.makedirs(OUTPUT,exist_ok= True)
    print(OUTPUT, "exisits")

    # change train set 

    # main functions : 
    build_tfdataset()    
    neg_train_dataset, pos_train_dataset = extract_datasets()
    master = neg_train_dataset.concatenate(pos_train_dataset)
    mins = None
    maxes = None

    if NORMALIZE:
        mins, maxes = max_min_normalize(master)
        with open("mins_maxes.csv", "w") as f: 
            f.write(str(mins)+"\n"+str(maxes))
    # for i,l in neg_train_dataset.take(1):
    #     print(i)

    if POS:
        train(pos_train_dataset, EPOCHS, mins, maxes)
        
    else:
        train(neg_train_dataset,EPOCHS, mins, maxes)





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
                                label= int(event_data[0])
                                serialized = serialize_example(signal, label)
                                if label == 0:
                                    neg_writer.write(serialized)
                                else:   
                                    pos_writer.write(serialized)


def max_min_normalize(dataset):
    maxes = np.full(10, -np.inf)
    mins = np.full(10, np.inf)
    count = 0
    for batch in dataset:
        count += 1
        # print(count, batch[0].shape)
        for i in batch[0]:
            
            reshaped = i.numpy().reshape((-1,10,640))
            for sample in reshaped:
                for sensor in range(sample.shape[0]):
                    maxes[sensor] = max(maxes[sensor],np.max(sample[sensor]))
                    mins[sensor] = min(mins[sensor],np.min(sample[sensor]))

    return (mins, maxes) 

                    
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

    raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(os.path.join(DATA_DIR,'tfrecords/neg_*.tfrecord')))

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


# wasserstien loss
def w_critic_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def w_generator_loss(fake_output):
    # Generator wants to fool critic into thinking fakes are real
    return -tf.reduce_mean(fake_output)

def gradient_penalty(critic, real, fake):
    batch_size = tf.shape(real)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)

    interpolated = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = tape.gradient(pred, [interpolated])[0]
    # Flatten gradients per sample
    grads = tf.reshape(grads, [batch_size, -1])
    gp = tf.reduce_mean((tf.norm(grads, axis=1) - 1.0) ** 2)
    print("gp_output",gp)
    return gp

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
critic_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)

# ### Train utils


LAMBDA_GP = 10.0  # Gradient penalty coefficient

@tf.function
def train_step(images):
    global generator_optimizer 
    global critic_optimizer
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    images = tf.reshape(images, [-1, 10, 640,1])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      real_output = critic(images, training=True)

      fake_output = critic(generated_images, training=True)
    #   gp = gradient_penalty(critic, images, generated_images)
      gen_loss = generator_loss(fake_output) 
      disc_loss = critic_loss(real_output, fake_output) #+ LAMBDA_GP * gp

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, critic.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    critic_optimizer.apply_gradients(zip(gradients_of_discriminator, critic.trainable_variables))
    return gen_loss, disc_loss

# ## Training loop

def normalize_flattened_batch(image_batch, mins, maxes):
    mins = tf.cast(mins, tf.float32)
    maxes = tf.cast(maxes, tf.float32)

    normalized_batch = []
    for i in range(image_batch.shape[0]):
        sample = tf.reshape(image_batch[i], [10, 640])
        sample = 2.0 * (sample - mins[:, tf.newaxis]) / (maxes[:, tf.newaxis] - mins[:, tf.newaxis]) - 1.0
        sample = tf.reshape(sample, [6400])
        normalized_batch.append(sample)

    return tf.stack(normalized_batch)


def train(dataset, epochs,mins = None, maxes = None):

   
    checkpoint_dir = os.path.join(OUTPUT,"training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=critic_optimizer,
                                    generator=generator,
                                    discriminator=critic)
        
    if mins is not None:
        mins = tf.constant(mins, dtype=tf.float32)
        maxes = tf.constant(maxes, dtype=tf.float32)

    print("TRAINING HAS BEGUN")
    for epoch in range(epochs):
        start = time.time()

        gen_loss, disc_loss = 0,0
        for idx, image_batch in enumerate(dataset):
            # print("batch: ", idx)
            data = image_batch[0]
            if mins is not None:
                data = normalize_flattened_batch(data, mins, maxes)

            gen_loss, disc_loss = train_step(data) # only need signal for gan traing

        # Produce images for the GIF as you go

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix =  str(epoch) +"_"+ checkpoint_prefix)

        print ('Time for epoch {} is {} sec gen loss {} disc loss {}'.format(epoch + 1, time.time()-start, gen_loss, disc_loss))

        with open(os.path.join(OUTPUT,"gan_log.txt"),'a+') as f:
            f.write('Time for epoch {} is {} sec gen loss {} disc loss {}'.format(epoch + 1, time.time()-start, gen_loss, disc_loss))


        generator.save(os.path.join(OUTPUT, "generator.keras"))
        critic.save(os.path.join(OUTPUT, "critic.keras"))



  




if __name__ == "__main__":
    main()