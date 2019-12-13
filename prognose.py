# venv\Scripts\activate
# cd C:\Users\felix\Dropbox\Uni-ty\6. Semester\Projektarbeit
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging
import dataCsvReader as dr
import dataDownloader as dl
import pandas as pd
import os
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

                  #dataset, dataset[:, 0], 0,TRAIN_SPLIT, past_history,future_target, STEP)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    return np.array(data), np.array(labels)


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


past_history = 72# inputtimesteps
future_target = 24  # timesteps into future
TEST_LENGTH= 12600 #Stunden
STEP = 1
BATCH_SIZE = 256
BUFFER_SIZE = 10000
EPOCHS = 2

isTraining = True
# isTraining=False

if(True):
  dl.updateWeatherHistory()
  # dl.updateForecast()
  # dl.updatePowerprice()
data = dr.getData()
data = data.drop(['diffScaledPrice'], axis=1)
# data = data.drop(['Price'], axis=1)
dataset = data.values
TRAIN_SPLIT = len(dataset)-TEST_LENGTH
# normalize the dataset using the mean and standard deviation of the training data
# data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
# data_std = dataset[:TRAIN_SPLIT].std(axis=0)
# dataset = (dataset - data_mean) / data_std
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)
print(y_val_multi)
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

# define model
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.GRU(past_history, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
#multi_step_model.add(tf.keras.layers.GRU(int(past_history/2)))
# multi_step_model.add(tf.keras.layers.GRU(past_history, return_sequences=True))
multi_step_model.add(tf.keras.layers.GRU(int(past_history/2)))
multi_step_model.add(tf.keras.layers.Dense(future_target))
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

if isTraining:
    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=2000,
                                          validation_data=val_data_multi,
                                          validation_steps=400)
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
    multi_step_model.save_weights('./checkpoints/testing')
else:
    multi_step_model.load_weights('./checkpoints/testing')

for x, y in val_data_multi.take(1):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
