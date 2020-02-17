# venv\Scripts\activate
# cd C:\Users\felix\Dropbox\Uni-ty\6. Semester\Projektarbeit
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import dataDownloader as dl
from dataCsvReader import get_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


def single_step_predict_plot(inputs, title, plot=True):
    predictions = []
    inputCopy = np.copy(inputs)  # copy Array to not overwrite original data
    x_in = inputCopy[:PAST_HISTORY].reshape(-1, PAST_HISTORY, inputCopy.shape[1])
    predictions.append(multi_step_model.predict(x_in)[0][-1])

    for j in range(1, FUTURE_TARGET):
        x_in = inputCopy[j:j + PAST_HISTORY].reshape(-1, PAST_HISTORY, inputCopy.shape[1])
        x_in[-1, -1, 0] = predictions[j - 1]  # use last prediction as power Price Input for next Prediction
        predictions.append(multi_step_model.predict(x_in)[0][-1])

    truth = inputs[-FUTURE_TARGET:, 0]
    # truth = truth * data_std[0] + data_mean[0]
    #     # predictions = np.array(predictions) * data_std[0] + data_mean[0]
    predictions = np.array(predictions)
    error = np.around(np.sqrt(np.mean(np.square(truth - predictions))), 2)
    if plot:
        plt.figure()
        plt.plot(predictions, 'r', label='predictions; RMSE: {}'.format(error))
        plt.plot(truth, 'b', label='Truth')
        plt.legend()
        plt.title(title)
        plt.show()
    return error


def multivariate_data_single_step(dataset, target, start_index, end_index,
                                  history_size, target_size):
    multivariate_data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(history_size, end_index):
        indices = range(i - history_size + 1, i + 1)
        multivariate_data.append(dataset[indices])
        labels.append(target[i + target_size])
    multivariate_data = np.array(multivariate_data)
    return multivariate_data.reshape(multivariate_data.shape[0], multivariate_data.shape[1], -1), np.array(labels)


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


def decompose_data(data):
    series = data["Price"]
    res = STL(series).fit()
    # estimated trend, seasonal and residual components
    data["Power_Residual"] = res.resid  # the estimated residuals
    data["Power_Seasonal"] = res.seasonal  # The estimated seasonal component
    data["Power_Trend"] = res.trend  # The estimated trend component
    plt.show()


def initialize_network():
    # define model
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.GRU(PAST_HISTORY,
                                             return_sequences=True,
                                             input_shape=(x_train.shape[-2:])))
    multi_step_model.add(tf.keras.layers.GRU(int(PAST_HISTORY / 2)))
    # multi_step_model.add(tf.keras.layers.GRU(past_history))
    multi_step_model.add(tf.keras.layers.Dense(FUTURE_TARGET))
    return multi_step_model


PAST_HISTORY = 12  # inputtimesteps
FUTURE_TARGET = 24  # timesteps into future
TEST_LENGTH = 20  # in percent
STEP = 1
BATCH_SIZE = 256
BUFFER_SIZE = 500
EPOCHS = 100
np.set_printoptions(linewidth=500, threshold=10, edgeitems=5, suppress=True)

START = '2015-1-1'
END = '2019-12-20'  # date.today()
IS_TRAINING = True
# IS_TRAINING=False
UPDATE_DATA = False

if UPDATE_DATA:
    dl.updateWeatherHistory(start=START, end=END, times=["recent"])
    dl.updateForecast()
    dl.update_power_price()

data = get_data(start=START, end=END)
TRAIN_SPLIT = int(len(data) / TEST_LENGTH)

decompose_data(data)

data.drop('Price', axis=1, inplace=True)
dataset = data.values
cols=data.columns.tolist()
cols = cols[-1:] + cols[:-1] # move last spot to first column
data = data[cols]
target=data["Power_Residual"]
# normalize the dataset using the mean and standard deviation of the training data
# data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
# data_std = dataset[:TRAIN_SPLIT].std(axis=0)
# dataset = (dataset - data_mean) / data_std


x_train, y_train = multivariate_data_single_step(dataset, target,
                                                 0, TRAIN_SPLIT, PAST_HISTORY, 1)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

x_val, y_val = multivariate_data_single_step(dataset, target,
                                             TRAIN_SPLIT, len(dataset) - 50,
                                             PAST_HISTORY, 1)
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()

multi_step_model = initialize_network()
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae',
                         lr=0.0003)

#Seasonal
forecast_length=24
test = data["Power_Seasonal"].iloc[-forecast_length:]

train = data["Power_Seasonal"].iloc[-forecast_length*3:-forecast_length]

model2 = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=24, damped=True,freq="H")

fit2 = model2.fit()
pred2 = fit2.forecast(forecast_length)
sse2 = np.sqrt(np.mean(np.square(test.values - pred2.values)))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[-24:], train.values[-24:]);
ax.plot(test.index, test.values, label='truth');
ax.plot(test.index, pred2, linestyle='--', color='#3c763d', label="damped (RMSE={:0.2f}, AIC={:0.2f})".format(sse2, fit2.aic));
ax.legend();
ax.set_title("Holt-Winter's Seasonal Smoothing");
plt.show()

# if IS_TRAINING:
#     multi_step_history = multi_step_model.fit(train_data, epochs=EPOCHS,
#                                               steps_per_epoch=1000 ,
#                                               validation_data=val_data,
#                                               validation_steps=400)
#     plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
#     multi_step_model.save_weights('./checkpoints/testing')
# else:
#     multi_step_model.load_weights('./checkpoints/testing')
#
# # prediction
# single_step_predict_plot(dataset[TRAIN_SPLIT - PAST_HISTORY - FUTURE_TARGET:TRAIN_SPLIT],
#                          title="ON TRAINING DATA")
# single_step_predict_plot(dataset[TRAIN_SPLIT:TRAIN_SPLIT + PAST_HISTORY + FUTURE_TARGET],
#                          title="ON TEST DATA")
