import sys
print(sys.executable)

seed = 1

from PIL import Image
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)
from sklearn.utils import shuffle


training_image_size = 240

########Set the data path properly: Download the provided dataset
file_name_img = 'Path/images_240_100k.npz'
file_name_labels = 'Path/labels_240_100k.npz'

# checkpoint path to save
checkpoint_filepath = 'data/Train_Supervized_model_100k_240.h5'
filename='data/Train_Supervized_model_100k_240.csv'



import tensorflow as tf
from tensorflow.python.keras.backend import set_session
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

import pickle
import cv2

with open(file_name_img, 'rb') as f:
    training_images = np.load(f)

with open(file_name_labels, 'rb') as f:
    training_labels = np.load(f)


print('Training dataset *******')
print(training_images.shape, training_labels.shape)


from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten, Conv2D
from tensorflow.keras import Input, Model


def get_model():
    i = Input(shape=(training_image_size, training_image_size, 1), name='input')
    m = Conv2D(64, (5, 5), strides=(2,2), activation='relu')(i)
    m = Conv2D(32, (3, 3), strides=(2,2), activation='relu')(m)
    m = Conv2D(32, (3, 3), strides=(2,2), activation='relu')(m)
    m = Conv2D(16, (3, 3), strides=(2,2), activation='relu')(m)
    m = Flatten()(m)
    m = Dense(64, activation='relu')(m)
    m = Dense(64, activation='relu')(m)
    m = Dense(64, activation='relu')(m)
    output = Dense(4, activation='linear', name='output')(m)
    model = Model(inputs=i, outputs=output)

    model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["mse"])

    return model


model = get_model()

print(model.summary())

epochs = 1000


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

model.fit(training_images, training_labels,
                    verbose=1,
                    epochs=epochs,
                    batch_size=128,callbacks=[model_checkpoint_callback, history_logger], validation_split = 0.2)
