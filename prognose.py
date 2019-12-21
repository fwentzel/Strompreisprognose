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
from datetime import date
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

os.system('cls')
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

        # data=np.append(data,dataset[i:i+target_size,1:],axis=1)
        # print(data)
        # print("********++")
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    # print("dataset")
    # print(np.array(data))
    return np.array(data), np.array(labels)




def multi_step_plot(history, true_future, prediction):
    # print(history)
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'b',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'r',
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
future_target= 24  # timesteps into future
TEST_LENGTH= 12600 #Stunden
STEP = 1
BATCH_SIZE = 256
BUFFER_SIZE = 10000
EPOCHS = 100
np.set_printoptions(linewidth =500)

start='2015-1-1'
end=date.today()
end='2019-12-20'
isTraining = True
# isTraining=False

if(False):
  dl.updateWeatherHistory(start=start,end=end,times=["recent"])
  # dl.updateForecast()
  # dl.updatePowerprice()
data= dr.getData(start=start,end=end)
print(data)
# data = data.drop(['diffScaledPrice'], axis=1)
# data = data.drop(['Price'], axis=1)
dataset = data.values
TRAIN_SPLIT = len(dataset)-TEST_LENGTH
# normalize the dataset using the mean and standard deviation of the training data
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset - data_mean) / data_std
def multivariate_data_Single_Step(dataset, target, start_index,end_index, history_size,
                      target_size):
    data = []
    labels = []
    start_index=start_index+history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(history_size, end_index):
        indices = range(i - history_size+1, i+1)
        data.append(dataset[indices])
        labels.append(target[i + target_size])
        # if i<10:
        #   print("data:",data,"labels",labels)
    
    # data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    data=np.array(data)
    return data.reshape(data.shape[0],-1 , data.shape[1]), np.array(labels)
x_train, y_train = multivariate_data_Single_Step(dataset, dataset[:, 0],
                                                 0,TRAIN_SPLIT, past_history,1)
x_val, y_val = multivariate_data_Single_Step(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, len(dataset)-50, past_history,1
                                             )

# train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_data = train_data.cache().shuffle(len(x_train)).batch(BATCH_SIZE).repeat()

# val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# val_data = val_data.batch(BATCH_SIZE).repeat()

# define model
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.GRU(past_history, return_sequences=True, input_shape=x_train.shape[-2:]))
multi_step_model.add(tf.keras.layers.GRU(past_history))
# multi_step_model.add(tf.keras.layers.GRU(int(past_history/2), return_sequences=True))
# multi_step_model.add(tf.keras.layers.GRU(int(past_history/2)))
multi_step_model.add(tf.keras.layers.Dense(future_target))
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae',lr=0.0004)

if isTraining:
    multi_step_history = multi_step_model.fit(x=x_train,y=y_train, epochs=EPOCHS,
                                          validation_data=[x_val, y_val])
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
    multi_step_model.save_weights('./checkpoints/testing')
else:
    multi_step_model.load_weights('./checkpoints/testing')
#prediction
def multi_single_step_predict(inputs):
  
  inputs=inputs.reshape(inputs.shape[0],-1 , inputs.shape[1])
  preds = []
  for j in range(0, future_target):
      X_in=inputs[j:past_history+j]
      # print("INPUT:",X_in)
      # print(multi_step_model.predict(X_in))
      # y_pred = sess.run(outputs, feed_dict = {X:X_in})
      # if(j<pred_steps_forward-1):
      #     inputs[0,n_steps+j+1,0]=y_pred[0,-1,0]
      # preds.append(y_pred[0,-1,0]) # add the last one
  return preds
multi_single_step_predict(dataset[-50:-50+past_history+future_target])


# for x, y in val_data.take(1):
#     multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
