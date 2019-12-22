# venv\Scripts\activate
# cd C:\Users\felix\Dropbox\Uni-ty\6. Semester\Projektarbeit
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging
import dataCsvReader as dr
import dataDownloader as dl
import pandas as pd
# tf.get_logger().setLevel('ERROR')

from datetime import date
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# os.system('TF_CPP_MIN_LOG_LEVELs')


def singlestep_predict_plot(inputs,title,plot=True):
  preds = []
  inputCopy=np.copy(inputs)#copy Array to not overwrite original data
  X_in=inputCopy[:past_history].reshape(-1,past_history,inputCopy.shape[1])
  preds.append(multi_step_model.predict(X_in)[0][-1])

  for j in range(1, future_target):
      X_in=inputCopy[j:j+past_history].reshape(-1,past_history,inputCopy.shape[1])
      X_in[-1,-1,0]=preds[j-1] # use last prediction as power Price Input for next Prediction
      preds.append(multi_step_model.predict(X_in)[0][-1])
  
  truth=inputs[-future_target:,0]
  truth=truth*data_std[0]+data_mean[0]
  preds=np.array(preds)*data_std[0]+data_mean[0]
  error= np.around(np.sqrt(np.mean(np.square(truth - preds))),2)
  if plot:
    plt.figure()
    plt.plot( preds, 'r', label='predictions; RMSE: {}'.format(error))
    plt.plot(truth, 'b', label='Truth')
    plt.legend()
    plt.title(title)
    plt.show()
  return error
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
    data=np.array(data)
    return data.reshape(data.shape[0], data.shape[1],-1), np.array(labels)

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


past_history =12# inputtimesteps
future_target= 24  # timesteps into future
TEST_LENGTH= 20 #in percent
STEP = 1
BATCH_SIZE = 256
BUFFER_SIZE = 500
EPOCHS = 100
np.set_printoptions(linewidth =500,threshold=10,edgeitems =5,suppress=True)

start='2015-1-1'
end=date.today()
end='2019-12-20'
isTraining = True
# isTraining=False

if(False):
  dl.updateWeatherHistory(start=start,end=end,times=["recent"])
  dl.updateForecast()
  dl.updatePowerprice()

data= dr.getData(start=start,end=end)
dataset = data.values
TRAIN_SPLIT = int(len(dataset)/TEST_LENGTH)
# normalize the dataset using the mean and standard deviation of the training data
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset - data_mean) / data_std

x_train, y_train = multivariate_data_Single_Step(dataset, dataset[:, 0],
                                                 0,TRAIN_SPLIT, past_history,1)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()


x_val, y_val = multivariate_data_Single_Step(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, len(dataset)-50, past_history,1)
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()
errorFrame=pd.DataFrame()
for j in np.arange(0.00001,0.0001,0.00003):
  # define model
  multi_step_model = tf.keras.models.Sequential()
  multi_step_model.add(tf.keras.layers.GRU(past_history, return_sequences=True, input_shape=(x_train.shape[-2:])))
  # multi_step_model.add(tf.keras.layers.GRU(int(past_history/2)))
  multi_step_model.add(tf.keras.layers.GRU(past_history))
  multi_step_model.add(tf.keras.layers.Dense(future_target))
  multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae',lr=j)

  if isTraining:
      multi_step_history = multi_step_model.fit(train_data, epochs=EPOCHS,steps_per_epoch=100,
                                            validation_data=val_data,validation_steps=1000)

      # plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
      multi_step_model.save_weights('./checkpoints/testing')
  else:
      multi_step_model.load_weights('./checkpoints/testing')

  #prediction
  # singlestep_predict_plot(dataset[TRAIN_SPLIT- past_history- future_target:TRAIN_SPLIT],title="ON TRAINING DATA")
  errors=[]
  for i in range(1,4):
    errors.append(singlestep_predict_plot(dataset[- past_history- future_target-(i*10):-(i*10)],title="ON TEST DATA",plot=False))
  errorFrame[str(j)]=errors
errorFrame.to_csv("Data/errors.csv")
print(errorFrame)


