"""Preprocesss data and train model."""


import csv
import os
import cv2
import numpy as np
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

RESUME_TRAINING = False

def get_session(gpu_fraction=0.3, allow_growth=True):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_fraction,
        allow_growth=allow_growth)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



def angle_correction(image, view, steering, correction=0.2):
    steering_out = steering
    if view == 'left':
        steering_out = steering + correction
    if view == 'right':
        steering_out = steering - correction
    return steering_out



def load_data(data_file):
    """Load and Preprocesss data."""
    images = []
    measurements = []
    with open(data_file, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            for view in ['center', 'right', 'left']:
                source_path = line[view]
                filename = source_path.split('/')[-1]
                current_path = os.path.join(data_dir, 'IMG', filename)
                try:
                    image = cv2.imread(current_path)
                    # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
                    # image = cv2.resize(image, (100, 50))

                    images.append(image)
                    measurement = float(line['steering'])
                    measurement = angle_correction(image, view, measurement, correction=0.2)
                    measurements.append(measurement)

                    image_flipped = np.fliplr(image)
                    measurement_flipped = -measurement
                    images.append(image_flipped)
                    measurements.append(measurement_flipped)

                except Exception as e:
                    print(str(e))
                    print(line[3])

    x_train = np.array(images)
    y_train = np.array(measurements)
    return x_train, y_train

def lenet(input_shape=None,regularizer=regularizers.l1(0.01), kernel_initializer='normal'):
    """Deep neural network with LeNet architecture."""

    crop_top_rows = int(input_shape[1] * .35)
    crop_bottom_rows = int(input_shape[1] * .05)

    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x/255.-0.5,
                            input_shape=input_shape))
    model.add(layers.convolutional.Cropping2D(
        cropping=(
            (crop_top_rows, crop_bottom_rows),
            # (70,25),
            (0,0))))
    model.add(layers.Convolution2D(6,5,5,activation='relu',kernel_initializer=kernel_initializer))
    model.add(layers.MaxPooling2D())
    model.add(layers.Convolution2D(6,5,5,activation='relu',kernel_initializer=kernel_initializer))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activity_regularizer=regularizer,kernel_initializer=kernel_initializer))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(84, activity_regularizer=regularizer,kernel_initializer=kernel_initializer))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=opt)

    return model

def train(x_train, y_train, resume_training=False):
    """Train model.

    Notes:
        Dump the trained model as 'model.h5'. Overwrite if exists.


    Args:
        x_train (:obj:`numpy.array`) : matrix of shape (nsamples, npixels)
        y_train (:obj:`numpy.array`) : training labels - steering angle
        resume_training (bool) : if True, then resume training, else
            start training a new model (overwrite if exists)
    """
    print(x_train[0].shape)
    print(y_train.shape)
    print((int(x_train[0].shape[1] * .3), int(x_train[0].shape[1] * .05)))

    input_shape = x_train[0].shape


    if resume_training:
        model= models.load_model('model.h5')
    else:
        model = lenet(input_shape=input_shape,
                      regularizer=None)

    model.fit(x_train, y_train,
             validation_split=0.2,
             shuffle=True,
             epochs=5,
             batch_size=128)

    model.save('model.h5')

def main(resume_training=False):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(root_dir, 'data-2')
    data_file = os.path.join(data_dir, 'driving_log.csv')
    x_train, y_train = load_data(data_file)
    train(x_train, y_train, resume_training=resume_training)

if __name__ == '__main__':
    RESUME_TRAINING = True
    main(resume_training=RESUME_TRAINING)
