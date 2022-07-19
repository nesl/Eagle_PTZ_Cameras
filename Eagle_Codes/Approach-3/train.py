"""
Approach-3: Relative_location + Control

Here, we train a neural network to predict relative locations.
The neural network has similar architecture to Eagle policie.
"""

#which GPU to use
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed = 0

import sys
from PIL import Image
import numpy as np
import pickle
import cv2
import tensorflow as tf

np.random.seed(seed)
tf.random.set_seed(seed)
from sklearn.utils import shuffle
from tensorflow.python.keras.backend import set_session
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)



#Load the training data
training_image_size =240
file_name_img = 'Path/images_240_50k.npz'
file_name_labels = 'Path/labels_240_50k.npz'


with open(file_name_img, 'rb') as f:
    training_images = np.load(f)

with open(file_name_labels, 'rb') as f:
    training_labels = np.load(f)


print('Training dataset *******')
print(training_images.shape, training_labels.shape)

#get label in scaled format
def get_lables(dat):
    x1,y1,x2,y2 = dat[0], dat[1], dat[2], dat[3]

    #convert to the scale of 240 first
    x1 = (x1*training_image_size)
    y1 = (y1*training_image_size)
    x2 = (x2*training_image_size)
    y2 = (y2*training_image_size)

    x_center = (x1+x2)/2.0
    y_center = (y1+y2)/2.0

    width = training_image_size/2.0

    x_distance = (width - x_center)
    y_distance = (width - y_center)

    scaled_x_distance = x_distance/(width)
    scaled_y_distance = y_distance/(width)

    size_of_box = (y2-y1)*(x2-x1)/(training_image_size*training_image_size)

    return scaled_x_distance, scaled_y_distance, size_of_box


labels = []
for i in range(training_labels.shape[0]):
    scaled_x_distance, scaled_y_distance, size_of_box = get_lables(training_labels[i])
    labels.append([scaled_x_distance, scaled_y_distance, size_of_box])

labels = np.array(labels)
training_images = training_images.reshape(-1, training_image_size, training_image_size, 1)

import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten, Conv2D
from tensorflow.keras import Input, Model

def get_model():
    i = Input(shape=(240, 240, 1), name='input')
    m = Conv2D(64, (5, 5), strides=(2,2), activation='relu')(i)
    m = Conv2D(32, (3, 3), strides=(2,2), activation='relu')(m)
    m = Conv2D(32, (3, 3), strides=(2,2), activation='relu')(m)
    m = Conv2D(16, (3, 3), strides=(2,2), activation='relu')(m)
    m = Flatten()(m)
    m = Dense(64, activation='relu')(m)
    m = Dense(64, activation='relu')(m)
    m = Dense(64, activation='relu')(m)
    output = Dense(3, activation='linear', name='output')(m)
    model = Model(inputs=i, outputs=output)

    model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tensorflow.keras.optimizers.Adam(),
    metrics=["mse"])

    return model


model = get_model()
print(model.summary())

epochs = 150

checkpoint_filepath = 'models/trainedModel.h5'

#can tune the current checkpoint if desired
#model.load_weights(checkpoint_filepath)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


model.fit(training_images, labels,
                    verbose=1,
                    epochs=epochs,
                    batch_size=512,callbacks=[model_checkpoint_callback], validation_split = 0.2)
