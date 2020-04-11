import random
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import pandas as pd
from tensorflow_core.python.keras.callbacks import EarlyStopping, \
    ModelCheckpoint, LearningRateScheduler


class NeuralNetPrediction:
    TRAIN_LENGTH = .7  # percent

    def __init__(self, data, future_target, datacolumn,
                 test_split_at_hour, net_type, train_day_of_week=False):

        self.read_json(net_type)
        test_data = data.iloc[-test_split_at_hour - self.past_history:]
        train_data = data.iloc[:-test_split_at_hour]

        self.RELEVANT_COLUMNS = [datacolumn, "wind", "cloudiness",
                                 "air_temperature", "sun", 'Weekend',
                                 'Hour', 'Holiday']
        self.train_target = train_data[datacolumn]
        self.train_dataset = train_data[self.RELEVANT_COLUMNS].values

        self.test_target = test_data[datacolumn]
        self.test_dataset = test_data[self.RELEVANT_COLUMNS].values

        self.future_target = future_target  # timesteps into future
        self.x, self.y = self.multivariate_data_single_step()

        # only read in day_models when its a "complete" net
        if not net_type.startswith("day_model_"):
            self.manage_day_models(data, train_day_of_week, datacolumn,
                                   test_split_at_hour)

    def manage_day_models(self, data, train_day_of_week, datacolumn,
                          test_split_at_hour):
        self.day_models = []
        for i in range(7):

            if datacolumn == "Price":
                save_name = "complete_day_{}".format(i)
                net_type = "day_model_complete"
            else:
                save_name = "residual_day_{}".format(i)
                net_type = "day_model_residual"

            net = NeuralNetPrediction(datacolumn="Price",
                                      data=data,
                                      future_target=self.future_target,
                                      test_split_at_hour=test_split_at_hour,
                                      net_type=net_type)
            if train_day_of_week:
                net.update_train_data_day_of_week(i)
                print("training net ", i)
                net.initialize_network()
                net.train_network(
                    savename=save_name,
                    save=True,
                    lr_schedule="polynomal",
                    power=2)  # lr_schedule="polynomal" oder "step
            else:
                net.load_model(savename=save_name)
            self.day_models.append(net)

    # LatexMarkerWeekdayStart
    def update_train_data_day_of_week(self, day_of_week):
        indices = [self.train_target.index.dayofweek == day_of_week][0]
        indices = indices[self.past_history:]
        self.x = self.x[indices]
        self.y = self.y[indices]

    # LatexMarkerWeekdayEnd

    def read_json(self, net_type):
        with open('configurations.json', 'r') as f:
            models_dict = json.load(f)
        self.dropout = models_dict[net_type]["dropout"]
        self.epochs = models_dict[net_type]["epochs"]
        self.additional_layers = models_dict[net_type]["layer"]
        self.past_history = models_dict[net_type]["past_history"]
        self.batch_size = models_dict[net_type]["batch_size"]

    def initialize_network(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(self.past_history,
                                       return_sequences=True,
                                       input_shape=(self.x.shape[-2:])))
        for i in range(self.additional_layers):
            if i < self.additional_layers - 1:
                model.add(
                    tf.keras.layers.LSTM(self.past_history,
                                         return_sequences=True,
                                         dropout=self.dropout))  # 0,3 3,7
            else:
                model.add(tf.keras.layers.LSTM(int(self.past_history)))

        # model.add(tf.keras.layers.Dense(self.future_target))
        model.add(tf.keras.layers.Dense(1))
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1),
        #     loss="mae")
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mae")
        self.model = model

    def load_model(self, savename, day_model=False):

        model = tf.keras.models.load_model(
            '.\checkpoints\{0}'.format(savename))
        model_input = model.layers[0].input.shape[1]
        if self.past_history != model_input and not "_day_" in savename:  # only throw "error" when its a "completed" net
            print(
                "Saved model expects {} Input steps. past History adjusted to fit this requirement".format(
                    model_input))
            self.past_history = model_input
        self.model = model

    # LatexMarkerDataStart
    def multivariate_data_single_step(self):
        multivariate_data = []
        labels = []
        for i in range(self.past_history, len(self.train_dataset)):
            indices = range(i - self.past_history, i)
            multivariate_data.append(self.train_dataset[indices])
            labels.append(self.train_target[i])
        multivariate_data = np.array(multivariate_data).astype(float)
        return multivariate_data.reshape(multivariate_data.shape[0],
                                         multivariate_data.shape[1],
                                         -1), np.array(labels).astype(
            float)

    # LatexMarkerDataEnd

    def train_network(self, savename, power=1, initAlpha=0.001,
                      lr_schedule="polynomal", save=True):
        if lr_schedule == "polynomal":
            if power is None:
                power = 1
            schedule = PolynomialDecay(maxEpochs=self.epochs,
                                       initAlpha=initAlpha, power=power)
        elif lr_schedule == "step":
            schedule = StepDecay(initAlpha=initAlpha, factor=0.8,
                                 dropEvery=15)
        # schedule.plot(self.epochs)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                           patience=5,
                           restore_best_weights=True)  # restore_best_weights=True
        history = self.model.fit(x=self.x, y=self.y,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 verbose=1,
                                 validation_split=1 -
                                                  self.TRAIN_LENGTH,
                                 shuffle=True,
                                 callbacks=[es,
                                            LearningRateScheduler(
                                                schedule)])
        # tf.keras.callbacks.LearningRateScheduler(schedule)

        # self.plot_train_history(history, 'Multi-Step Training
        # and validation loss')
        if save:
            self.model.save('.\checkpoints\{0}'.format(savename))

    def plot_train_history(self, history, title):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(title)
        plt.legend()
        plt.show()

    # LatexSingleStepMarkerStart
    def single_step_predict(self, inputs, model, target=None):
        model_past_history = model.layers[0].input.shape[1]
        predictions = []
        inputCopy = np.copy(inputs)
        for j in range(self.future_target):
            x_in = inputCopy[j:j + model_past_history].reshape(1,
                                                               model_past_history,
                                                               inputCopy.shape[
                                                                   1])
            if j > 0:
                x_in[-1, -1, 0] = predictions[
                    j - 1]  # replace last power price with forecast

            predictions.append(model.predict(x_in)[0][-1])

        target_rows = target.iloc[-self.future_target:]
        self.truth = target_rows
        self.pred = pd.Series(
            np.array(predictions).reshape(self.future_target),
            index=target_rows.index)
        self.error = np.around(
            np.sqrt(
                np.mean(np.square(self.truth.values - predictions))), 2)
        self.single_errors = np.around(np.sqrt(
            np.square(self.truth.values - predictions)), 2)

    # LatexSingleStepMarkerEnd

    def multi_step_predict(self, inputs, model, target=None):
        inputCopy = np.copy(
            inputs)  # copy Array to not overwrite original train_data
        model_past_history = model.layers[0].input.shape[1]
        x_in = inputCopy[:model_past_history].reshape(1,
                                                      model_past_history,
                                                      inputCopy.shape[
                                                          1])
        prediction = model.predict(x_in)
        target_rows = target.iloc[-self.future_target:]
        self.truth = target_rows
        self.pred = pd.Series(
            np.array(prediction).reshape(self.future_target),
            index=target_rows.index)
        self.error = np.around(
            np.sqrt(np.mean(np.square(self.truth.values - prediction))),
            2)
        self.single_errors = np.sqrt(
            np.square(self.truth.values - prediction))

    def predict(self, use_day_model=False, offset=0):
        if use_day_model:
            dataset = self.day_models[0].test_dataset
            target = self.day_models[0].test_target
            # use random day model for past_history. ATM doesnt matter which one taken
            model_past_history = self.day_models[0].past_history
        else:
            dataset = self.test_dataset
            target = self.test_target
            model_past_history = self.past_history

        prediction_timeframe = slice(offset,model_past_history + self.future_target +
                                     offset)
        input = dataset[prediction_timeframe]
        target = target.iloc[model_past_history + offset:model_past_history + offset +
                                             self.future_target]
        if use_day_model:
            start_day = target.index[0].dayofweek
            print("Predicting using model for day:", start_day)
            model = self.day_models[start_day].model
        else:
            print("Predicting using normal model:", offset)
            model = self.model
        shape = model.layers[-1].output.shape[1]

        if shape == 1:
            self.single_step_predict(inputs=input, model=model,
                                     target=target)
        else:
            self.multi_step_predict(inputs=input, model=model,
                                    target=target)

    def plot_mass_error_over_day(self, mean_errorlist):

        i = 0
        hours = np.array([0.0 for x in range(24)])
        for row in mean_errorlist:
            for j in range(len(row)):
                hour = i + j
                while hour > 23:
                    hour -= 24
                hours[hour] += float(row[j])
            i += 1
        hours /= i
        index = [x for x in range(0, 24)]
        plt.bar(index, hours)
        plt.xlabel("Tageszeit")
        plt.ylabel("Durchschnittlicher Fehler ")
        plt.title(
            "Durchschittlicher Fehlerwert in Abhängigkeit zur Tageszeit")
        # plt.xticks([8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,0,1,2,3,4,5,6,7,8])
        plt.show()

    def mass_predict(self, iterations, step=1, use_day_model=False):
        j = 0
        single_errorlist = np.empty(
            [round(iterations / step), self.future_target])
        offsets = range(0, iterations, step)
        error_array = np.empty((iterations + self.future_target, 1))
        error_array[:] = np.nan
        for i in offsets:
            print("\rmass predict: {}/{}".format(j, round(
                iterations / step)),
                  sep=' ', end='', flush=True)
            self.predict(offset=i, use_day_model=use_day_model)
            single_errorlist[j] = self.single_errors
            arr = np.nanmean([error_array[i:i + self.future_target],
                              single_errorlist[j].reshape(
                                  self.future_target, 1)], axis=0)
            error_array[i:i + self.future_target] = arr
            j += 1

        mean_errorlist = np.around(np.mean(single_errorlist, axis=0),
                                   decimals=2)

        mean_mean_error_over_time = [np.mean(error_array[x - 12:x + 12])
                                     for x in
                                     range(12, len(error_array) - 12)]
        plt.plot(error_array, label="RMSE: {}".format(
            np.around(np.mean(mean_errorlist), 2)))
        plt.plot(range(12, len(error_array) - 12),
                 mean_mean_error_over_time,
                 label="Mean error of t-12 - t+12 window")
        plt.xticks(
            [x for x in range(0, iterations + self.future_target, 12)])
        plt.title(
            "Mean Error at every time step of mass prediciton with {} iterations and {} stepsize".format(
                iterations, step))
        plt.legend()
        plt.show()
        mean_error_over_time = [np.mean(mean_errorlist[:x]) for x in
                                range(1, len(mean_errorlist) + 1)]

        mean_error = np.around(mean_errorlist.mean(), 2)

    def plot_predictions(self, ax):
        time_slice = slice(self.past_history,
                           self.past_history + self.future_target)

        xticks = self.train_target.index[time_slice]

        ax[1].plot(xticks, self.pred,
                   label='predictions; RMSE: {}'.format(self.error))
        ax[1].plot(xticks, self.truth, label='Truth')
        ax[1].legend()
        ax[1].set_ylabel("RESIDUAL")


class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        self.level_start = 5
        lrs = [self(i) for i in range(0, epochs)]
        # the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(range(0, epochs), lrs, "o")
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        plt.show()


class PolynomialDecay(LearningRateDecay):
    def __init__(self, initAlpha, maxEpochs=100, power=1.0,
                 headstart=0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.headstart = headstart
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch):
        if epoch < self.headstart:
            return self.initAlpha
        # compute the new learning rate based on polynomial decay
        decay = (1 - ((epoch - self.headstart) / float(
            self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay
        # return the new learning rate
        return float(alpha)


class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.01, factor=0.75, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        # return the learning rate
        return float(alpha)
