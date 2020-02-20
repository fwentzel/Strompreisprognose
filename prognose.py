from __future__ import absolute_import, division, print_function, unicode_literals



from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from csv_reader import get_data
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import data_downloader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Parameter für Konsolenausgabe
matplotlib.rcParams['figure.figsize'] = (8, 6)
matplotlib.rcParams['axes.grid'] = False


def single_step_predict_plot(inputs, target, index,plot=True,axes=None):
    predictions = []
    inputCopy = np.copy(inputs)  # copy Array to not overwrite original data
    x_in = inputCopy[:PAST_HISTORY].reshape(-1, PAST_HISTORY, inputCopy.shape[1])
    predictions.append(multi_step_model.predict(x_in)[0][-1])

    for j in range(1, FUTURE_TARGET):
        x_in = inputCopy[j:j + PAST_HISTORY].reshape(-1, PAST_HISTORY, inputCopy.shape[1])
        x_in[-1, -1, 0] = predictions[j - 1]  # use last prediction as power Price Input for next Prediction
        predictions.append(multi_step_model.predict(x_in)[0][-1])

    truth = target.iloc[-FUTURE_TARGET:].values
    predictions = np.array(predictions)
    error = np.around(np.sqrt(np.mean(np.square(truth - predictions))), 2)
    if plot:
        axes[index].plot(predictions, 'r', label='predictions; RMSE: {}'.format(error))
        axes[index].plot(truth, 'b', label='Truth')
        axes[index].legend()
    return error


def multivariate_data_single_step(_dataset, _target, history_size):
    multivariate_data = []
    labels = []
    for i in range(history_size, len(_dataset) - 1):
        indices = range(i - history_size + 1, i + 1)
        multivariate_data.append(_dataset[indices])
        labels.append(_target[i + 1])
    multivariate_data = np.array(multivariate_data)
    return multivariate_data.reshape(multivariate_data.shape[0], multivariate_data.shape[1], -1), np.array(labels)


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


def decompose_data():
    series = data["Price"]
    res = STL(series).fit()
    # estimated trend, seasonal and residual components
    data["Power_Residual"] = res.resid  # the estimated residuals
    data["Power_Seasonal"] = res.seasonal  # The estimated seasonal component
    data["Power_Trend"] = res.trend  # The estimated trend component
    plt.show()


def seasonal_prediction_plot(_seasonal_train, _seasonal_test, _seasonal_pred, _seasonal_error, _seasonal_fit):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(_seasonal_train.index[-24:], _seasonal_train.values[-24:])
    ax.plot(_seasonal_test.index, _seasonal_test.values, label='truth')
    ax.plot(_seasonal_test.index, _seasonal_pred, linestyle='--', color='#3c763d',
            label="damped (RMSE={:0.2f}, AIC={:0.2f})".format(_seasonal_error, _seasonal_fit.aic))
    ax.legend()
    ax.set_title("Holt-Winter's Seasonal Smoothing")
    plt.show()


def initialize_network():
    # define model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(PAST_HISTORY,
                                  return_sequences=True, input_shape=(x.shape[-2:])))
    model.add(tf.keras.layers.GRU(int(PAST_HISTORY / 2)))
    # multi_step_model.add(tf.keras.layers.GRU(past_history))
    model.add(tf.keras.layers.Dense(FUTURE_TARGET))
    return model


PAST_HISTORY = 96  # inputtimesteps
FUTURE_TARGET = 24  # timesteps into future
TRAIN_LENGTH = .8  # percent
STEP = 1
BATCH_SIZE = 256
BUFFER_SIZE = 500
EPOCHS = 50
np.set_printoptions(linewidth=500, threshold=10, edgeitems=5, suppress=True)

START = '2015-1-1'
# END= date.today()
END = '2020-02-17'
IS_TRAINING = False  # False True
UPDATE_DATA = False

if UPDATE_DATA:
    data_downloader.updateWeatherHistory(start=START, end=END, times=["recent", "historical"])
    data_downloader.updateForecast()
    data_downloader.update_power_price()

data = get_data(start=START, end=END)
TRAIN_SPLIT = int(len(data) * TRAIN_LENGTH)
decompose_data()

# normalize the dataset using the mean and standard deviation of the training data
# data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
# data_std = dataset[:TRAIN_SPLIT].std(axis=0)
# dataset = (dataset - data_mean) / data_std

# Residual
target = data["Power_Residual"]
dataset = data[['V_N', 'SD_SO', 'F', 'Temp', 'Weekend', 'Hour', 'Holiday', 'Power_Residual']].values
# x_train, y_train = multivariate_data_single_step(dataset, target,0, TRAIN_SPLIT, PAST_HISTORY, 1)
# x_val, y_val = multivariate_data_single_step(dataset, target,TRAIN_SPLIT, len(dataset) - 50, PAST_HISTORY, 1)

x, y = multivariate_data_single_step(dataset, target, PAST_HISTORY)

# train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# val_data = val_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()


multi_step_model = initialize_network()
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae',
                         lr=0.0001)

if IS_TRAINING:
    # multi_step_history = multi_step_model.fit(train_data, epochs=EPOCHS,
    #                                           validation_data=val_data)
    multi_step_history = multi_step_model.fit(x=x, y=y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
                                              validation_split=1-TRAIN_LENGTH, shuffle=True)
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
    multi_step_model.save_weights('./checkpoints/testing')
else:
    multi_step_model.load_weights('./checkpoints/testing')

num_predicitons=100
mean_error=0
plot_predictions=False

max=len(dataset)-PAST_HISTORY-FUTURE_TARGET-TRAIN_SPLIT
fig, ax = plt.subplots(num_predicitons,1,sharey=True)

for i in range(0,num_predicitons):

    offset=random.randrange(max)
    prediction_timeframe=slice(TRAIN_SPLIT+offset, TRAIN_SPLIT + PAST_HISTORY + FUTURE_TARGET+offset)

    mean_error+=single_step_predict_plot(inputs=dataset[prediction_timeframe],
                             target=target.iloc[prediction_timeframe],
                             axes=ax,
                            plot=plot_predictions,
                             index=i)
    print(i,mean_error/(i+1))
    i+=1
print("Mean error of the {0} predicitons: {1}".format(num_predicitons,mean_error/num_predicitons))

if plot_predictions:plt.show()





# Seasonal
# forecast_length = 24
# seasonal_test = data["Power_Seasonal"].iloc[-forecast_length:]
#
# seasonal_train = data["Power_Seasonal"].iloc[-forecast_length * 3:-forecast_length]
#
# seasonal_model = ExponentialSmoothing(seasonal_train, trend="add", seasonal="add", seasonal_periods=24, damped=True,
#                                       freq="H")
#
# seasonal_fit = seasonal_model.fit()
# seasonal_pred = seasonal_fit.forecast(forecast_length)
# seasonal_error = np.sqrt(np.mean(np.square(seasonal_test.values - seasonal_pred.values)))
#
# seasonal_prediction_plot(seasonal_train, seasonal_test, seasonal_pred, seasonal_error, seasonal_fit)
