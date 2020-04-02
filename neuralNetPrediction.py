import random
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import pandas as pd
from tensorflow_core.python.keras.callbacks import EarlyStopping, \
    ModelCheckpoint, LearningRateScheduler


class NeuralNetPrediction:
    TRAIN_LENGTH = .7  # percent
    BATCH_SIZE = 32

    def __init__(self, train_data, test_data, future_target,
                 past_history,
                 datacolumn, epochs):
        self.RELEVANT_COLUMNS = [datacolumn, "wind", "cloudiness",
                                 "air_temperature", "sun", 'Weekend',
                                 'Hour',
                                 'Holiday']
        self.train_target = train_data[datacolumn]
        self.train_dataset = train_data[self.RELEVANT_COLUMNS].values
        self.test_target = test_data[datacolumn]
        self.test_dataset = test_data[self.RELEVANT_COLUMNS].values
        self.future_target = future_target  # timesteps into future
        self.past_history = past_history  # inputtimesteps
        self.epochs = epochs
        self.x, self.y = self.multivariate_data_single_step()

    def initialize_network(self, dropout, additional_layers,
                           learning_rate):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(self.past_history,
                                       return_sequences=True,
                                       input_shape=(self.x.shape[-2:])))
        for i in range(additional_layers):
            if i < additional_layers - 1:
                model.add(
                    tf.keras.layers.LSTM(self.past_history,
                                         return_sequences=True,
                                         dropout=dropout))  # 0,3 3,7
            else:
                model.add(tf.keras.layers.LSTM(int(self.past_history)))

        # model.add(tf.keras.layers.Dense(self.future_target))
        model.add(tf.keras.layers.Dense(1))
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1),
        #     loss="mae")
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate), loss="mae")
        self.model = model

    def load_model(self, savename):
        self.model = tf.keras.models.load_model(
            '.\checkpoints\{0}'.format(savename))
        model_input=self.model.layers[0].input.shape[1]
        if self.past_history != model_input:
            print("Saved model expects {} Input steps. past History adjusted to fit this requirement".format(model_input))
            self.past_history=model_input

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
                                         -1), np.array(
            labels).astype(float)

    def train_network(self, savename, power=1, initAlpha=0.01,
                      lr_schedule="polynomal", save=True):
        if lr_schedule == "polynomal":
            if power is None:
                power = 1
            schedule = PolynomialDecay(maxEpochs=self.epochs,
                                       initAlpha=initAlpha, power=power)
        elif lr_schedule == "step":
            schedule = StepDecay(initAlpha=initAlpha, factor=0.8,
                                 dropEvery=15)
        #schedule.plot(self.epochs)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                           patience=5,restore_best_weights=True)  # restore_best_weights=True
        history = self.model.fit(x=self.x, y=self.y,
                                 epochs=self.epochs,
                                 batch_size=self.BATCH_SIZE,
                                 verbose=1,
                                 validation_split=1 -
                                                  self.TRAIN_LENGTH,
                                 shuffle=True,
                                 callbacks=[es,
                                     LearningRateScheduler(schedule)])
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
    def single_step_predict(self, inputs, target=None):
        predictions = []
        inputCopy = np.copy(inputs)
        for j in range(self.future_target):
            x_in = inputCopy[j:j + self.past_history].reshape(1,
                                                              self.past_history,
                                                              inputCopy.shape[
                                                                  1])
            if j > 0:
                x_in[-1, -1, 0] = predictions[
                    j - 1]  # replace last power price with forecast

            predictions.append(self.model.predict(x_in)[0][-1])


        target_rows = target.iloc[-self.future_target:]
        self.truth = target_rows
        self.pred = pd.Series(
            np.array(predictions).reshape(self.future_target),
            index=target_rows.index)
        self.error = np.around(
            np.sqrt(
                np.mean(np.square(self.truth.values - predictions))), 2)
        self.single_errors = np.around(np.sqrt(
            np.square(self.truth.values - predictions)),2)

    # LatexSingleStepMarkerEnd

    def multi_step_predict(self, inputs, target=None):
        inputCopy = np.copy(
            inputs)  # copy Array to not overwrite original train_data
        x_in = inputCopy[:self.past_history].reshape(1,
                                                     self.past_history,
                                                     inputCopy.shape[1])
        prediction = self.model.predict(x_in)
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

    def predict(self, offset=0):

        dataset = self.test_dataset
        target = self.test_target
        prediction_timeframe = slice(offset,
                                     self.past_history + self.future_target +
                                     offset)
        input = dataset[prediction_timeframe]
        target = target.iloc[
                 self.past_history + offset:self.past_history + offset +
                                            self.future_target]
        shape = self.model.layers[-1].output.shape[1]
        if shape == 1:
            self.single_step_predict(inputs=input, target=target)
        else:
            self.multi_step_predict(inputs=input, target=target)

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

    def mass_predict(self, iterations, filename, learning_rate,
                     past_history, layers, step=1):
        j = 0
        single_errorlist = np.empty([round(iterations/step), self.future_target])
        offsets = range(0, iterations, step)
        plt.plot(self.test_target.index[
                 past_history:self.future_target + iterations + past_history],
                 self.test_target.iloc[
                 past_history:self.future_target + iterations + past_history],
                 label="TRUTH", lw=5)
        for i in offsets:
            self.predict(offset=i)
            print("errors for prediciton {}:   {} , {}  ".format(j,self.error,
                                                       self.single_errors))
            single_errorlist[j] = self.single_errors
            j += 1
            plt.plot(self.pred,label="prediciton {}: RMSE {}".format(i,self.error))

        plt.legend()
        #plt.show()
        mean_errorlist = np.around(np.mean(single_errorlist, axis=0),
                                   decimals=2)
        # plt.plot(offsets, mean_errorlist, label="mean Error over time")
        # plt.legend()
        # plt.show()
        # self.plot_mass_error_over_day(mean_errorlist)
        mean_error=np.around(mean_errorlist.mean(),2)
        min_error = 100
        min_config = None
        with open("Results/best_config_{}.csv".format(
                filename)) as config:
            reader = csv.reader(config, delimiter=',')
            for row in reader:
                min_config = row
            min_error = float(min_config[-1])
        print("mean errors of predicitons: ", mean_error, mean_errorlist)

        with open('Results/{}_results.csv'.format(filename), 'a',
                  newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(
                [learning_rate, past_history, layers,iterations, mean_error,
                 mean_errorlist.tolist()])

        if mean_error < min_error:
            min_error = mean_error
            min_config = [learning_rate, past_history, layers,
                          min_error]
            self.model.save('.\checkpoints\{}_best'.format(filename))
            with open('Results/best_config_{}.csv'.format(filename),
                      'w',
                      newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow(min_config)

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
        plt.plot(range(0, epochs), lrs,"o")
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
