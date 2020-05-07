
import matplotlib.pyplot as plt
import numpy as np
import os
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
        self.net_type = net_type
        self.read_json(net_type)
        test_data = data.iloc[-test_split_at_hour - self.past_history:]
        train_data = data.iloc[:-test_split_at_hour]

        self.RELEVANT_COLUMNS = [datacolumn, "wind", "cloudiness",
                                 "air_temperature", "sun", 'DayOfWeek',
                                 'Hour', 'Holiday']
        self.data = data
        self.datacolumn = datacolumn
        self.test_split_at_hour = test_split_at_hour
        self.train_target = train_data[datacolumn]
        self.train_dataset = train_data[self.RELEVANT_COLUMNS].values

        self.test_target = test_data[datacolumn]
        self.test_dataset = test_data[self.RELEVANT_COLUMNS].values

        self.future_target = future_target  # timesteps into future

        self.model=None
        self.day_models = [None for x in range(7)]

        # only read in day_models when its a "complete" net
        if not net_type.startswith("day_model_"):
            if (train_day_of_week):
                self.manage_day_models(train_day_of_week)

    def manage_day_models(self, train_day_of_week, index=-1):

        day_indeces = range(7) if index == -1 else [index]
        for i in day_indeces:
            print("Loading day model {} for {} prediciton".format(
                i, self.datacolumn))
            if self.datacolumn == "Price":
                save_name = "price_day_{}UTC".format(i)
                net_type = "price_day"
            else:
                save_name = "remainder_day_{}UTC".format(i)
                net_type = "remainder_day"

            net = NeuralNetPrediction(datacolumn=self.datacolumn,
                                      data=self.data,
                                      future_target=self.future_target,
                                      test_split_at_hour=self.test_split_at_hour,
                                      net_type=net_type)
            if train_day_of_week:
                net.update_train_data_day_of_week(i)
                print("training net ", i)
                net.initialize_network()
                net.train_network(
                    savename=save_name,
                    save=False,
                    lr_schedule="polynomal",
                    power=3)  # lr_schedule="polynomal" oder "STEP
            else:
                net.load_model(savename=save_name)
            self.day_models[i] = net

    # LatexMarkerWeekdayStart
    def update_train_data_day_of_week(self, day_of_week):
        if self.x==None:
            self.x, self.y = self.multivariate_data_single_step()
        indices = [self.train_target.index.dayofweek == day_of_week][0]
        indices = indices[self.past_history:]
        self.x = self.x[indices]
        self.y = self.y[indices]

    # LatexMarkerWeekdayEnd

    def read_json(self, net_type):
        with open('configurations.json', 'r') as f:
            models_dict = json.load(f)
        self.dropout = models_dict[net_type]["dropout"] / 10
        self.epochs = models_dict[net_type]["epochs"]
        self.additional_layers = models_dict[net_type]["layer"]
        self.past_history = models_dict[net_type]["past_history"]
        self.batch_size = models_dict[net_type]["batch_size"]

    # LatexMarkerNetInitStart
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
                                         dropout=self.dropout))
            else:
                model.add(tf.keras.layers.LSTM(int(self.past_history)))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mae")
        self.model = model

    # LatexMarkerNetInitEnd

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
                      lr_schedule="polynomal", save=True,
                      day_model=None):

        if lr_schedule == "polynomal":
            if power is None:
                power = 1
            schedule = PolynomialDecay(maxEpochs=self.epochs,
                                       initAlpha=initAlpha, power=power)
        elif lr_schedule == "STEP":
            schedule = StepDecay(initAlpha=initAlpha, factor=0.8,
                                 dropEvery=15)
        # schedule.plot(self.epochs)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                           patience=5,
                           restore_best_weights=True)  # restore_best_weights=True

        if self.x==None:
            self.x, self.y = self.multivariate_data_single_step()
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

    def predict(self, use_day_model=False, offset=0,axis=None):
        if use_day_model:
            if self.day_models[0] is None:
                self.manage_day_models(index=0, train_day_of_week=False)
            dataset = self.day_models[0].test_dataset
            target = self.day_models[0].test_target
            # use random day model for past_history. ATM doesnt matter which one taken
            model_past_history = self.day_models[0].past_history
        else:
            dataset = self.test_dataset
            target = self.test_target
            model_past_history = self.past_history

        prediction_timeframe = slice(offset,
                                     model_past_history + self.future_target +
                                     offset)
        input = dataset[prediction_timeframe]
        target = target.iloc[
                 model_past_history + offset:model_past_history + offset +
                                             self.future_target]
        if use_day_model:
            start_day = target.index[0].dayofweek
            if self.day_models[start_day] is None:
                self.manage_day_models(index=start_day,
                                       train_day_of_week=False)
            model = self.day_models[start_day].model
        else:
            model = self.model
        shape = model.layers[-1].output.shape[1]

        if shape == 1:
            self.single_step_predict(inputs=input, model=model,
                                     target=target)
        else:
            self.multi_step_predict(inputs=input, model=model,
                                    target=target)
        if axis is not None:
            self.plot_prediction(axis,self.datacolumn)


    # LatexSingleStepMarkerStart
    def single_step_predict(self, inputs, model, target=None):
        model_past_history = model.layers[0].input.shape[1]
        predictions = []
        inputCopy = np.copy(inputs)
        for j in range(self.future_target):
            x_in = inputCopy[j:j + model_past_history]\
                            .reshape(1,model_past_history,
                                     inputCopy.shape[1])
            if j > 0:
                # replace last power price with forecast
                x_in[-1, -1, 0] = predictions[j - 1]

            predictions.append(model.predict(x_in)[0][-1])
            del x_in

        target_rows = target.iloc[-self.future_target:]
        self.truth = target_rows
        self.pred = pd.Series(np.array(predictions).reshape(
                    self.future_target),index=target_rows.index)
        self.error = np.around(np.sqrt(np.mean(np.square(
                    self.truth.values - predictions))), 2)
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


    def mass_predict(self, axis,iterations, step=1, use_day_model=False):
        j = 0
        single_errorlist = np.empty(
            [round(iterations / step), self.future_target])
        offsets = range(0, iterations, step)
        error_array = np.empty((iterations + self.future_target, 1))
        error_array[:] = np.nan
        naive_error = 0
        max = round(iterations / step)
        for i in offsets:
            print("\rmass predict {}net: {}/{}".format(self.net_type,j, max),
                  sep=' ', end='', flush=True)
            self.predict(offset=i, use_day_model=use_day_model)
            single_errorlist[j] = self.single_errors
            arr = np.nanmean([error_array[i:i + self.future_target],
                              single_errorlist[j].reshape(
                                  self.future_target, 1)], axis=0)
            error_array[i:i + self.future_target] = arr

            if self.datacolumn == "Price":
                naive_pred = self.data["Price"].iloc[
                             - self.test_split_at_hour - 2 + i:-self.test_split_at_hour - 2 + i + self.future_target]
            else:
                naive_pred = np.zeros(self.future_target)

            naive_error += np.sqrt(np.mean(np.square(
                self.truth.values - naive_pred)))

            j += 1
        mean_naive_error = np.around(naive_error / j, 2)
        cumulative_errorlist = np.around(np.mean(single_errorlist, axis=0),
                                   decimals=2)

        mean_error_over_time = [np.mean(error_array[x - 12:x + 12])
                                     for x in
                                     range(12, len(error_array) - 12)]
        axis.plot(error_array,
                 label="mean error at timestep. Overall mean: {}".format(
                     np.around(np.mean(cumulative_errorlist), 2)))
        axis.plot(range(12, len(error_array) - 12),
                 mean_error_over_time,
                 label="Moving average in 25 hour window")
        axis.set_title(self.net_type)
        axis.legend()

    def plot_prediction(self, ax,method):
        ax.plot(self.truth.index, self.pred,
                   label='prediction; RMSE: {}'.format(self.error))
        ax.plot(self.truth.index, self.truth, label='Truth')
        ax.set_title(method)
        ax.legend()


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
