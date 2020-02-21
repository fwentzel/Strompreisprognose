import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class ResidualPrediction:
    def __init__(self, complete_data, future_target, start):
        self.target = complete_data["Residual"]
        self.dataset = complete_data[
            ['V_N', 'SD_SO', 'F', 'Temp', 'Weekend', 'Hour', 'Holiday', 'Residual']].values
        self.future_target = future_target  # timesteps into future
        self.start = start

        self.PAST_HISTORY = 96  # inputtimesteps

        self.TRAIN_LENGTH = .8  # percent
        self.STEP = 1
        self.BATCH_SIZE = 256
        self.BUFFER_SIZE = 500
        self.EPOCHS = 50
        self.TRAIN_SPLIT = int(len(self.dataset) * self.TRAIN_LENGTH)

        self.model = None
        self.x, self.y = self.multivariate_data_single_step()

        self.prediciton_truth_error = []

    def initialize_network(self, learning_rate):
        # define model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.GRU(self.PAST_HISTORY,
                                      return_sequences=True, input_shape=(self.x.shape[-2:])))
        model.add(tf.keras.layers.GRU(int(self.PAST_HISTORY / 2)))
        # multi_step_model.add(tf.keras.layers.GRU(past_history))
        model.add(tf.keras.layers.Dense(self.future_target))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae',
                      lr=learning_rate)
        self.model = model

    def multivariate_data_single_step(self):
        multivariate_data = []
        labels = []
        for i in range(self.PAST_HISTORY, len(self.dataset) - 1):
            indices = range(i - self.PAST_HISTORY + 1, i + 1)
            multivariate_data.append(self.dataset[indices])
            labels.append(self.target[i + 1])
        multivariate_data = np.array(multivariate_data)
        return multivariate_data.reshape(multivariate_data.shape[0], multivariate_data.shape[1], -1), np.array(labels)

    def train_network(self, train, checkpoint):
        if train:
            multi_step_history = self.model.fit(x=self.x, y=self.y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                                                verbose=2,
                                                validation_split=1 - self.TRAIN_LENGTH, shuffle=True)
            self.plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
            self.model.save_weights('./checkpoints/{0}'.format(checkpoint))
        else:
            self.model.load_weights('./checkpoints/{0}'.format(checkpoint))

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

    def single_step_predict(self, inputs, target):
        predictions = []
        inputCopy = np.copy(inputs)  # copy Array to not overwrite original data
        x_in = inputCopy[:self.PAST_HISTORY].reshape(-1, self.PAST_HISTORY, inputCopy.shape[1])
        predictions.append(self.model.predict(x_in)[0][-1])

        for j in range(1, self.future_target):
            x_in = inputCopy[j:j + self.PAST_HISTORY].reshape(-1, self.PAST_HISTORY, inputCopy.shape[1])
            x_in[-1, -1, 0] = predictions[j - 1]  # use last prediction as power Price Input for next Prediction
            predictions.append(self.model.predict(x_in)[0][-1])

        truth = target.iloc[-self.future_target:].values
        predictions = np.array(predictions)
        error = np.around(np.sqrt(np.mean(np.square(truth - predictions))), 2)

        return predictions, truth, error

    def predict(self, num_predicitons=2, random_offset=True):
        max_offset = len(self.dataset) - self.PAST_HISTORY - self.future_target - self.start
        mean_error = 0
        for i in range(0, num_predicitons):
            if random_offset:
                offset = random.randrange(max_offset)  # random offset for the new data
            else:
                offset = i
            prediction_timeframe = slice(self.start - self.PAST_HISTORY - self.future_target + offset,
                                         self.start + offset)

            self.prediciton_truth_error.append(self.single_step_predict(inputs=self.dataset[prediction_timeframe],
                                                                        target=self.target.iloc[
                                                                               self.start:self.start + self.future_target]))
            mean_error += self.prediciton_truth_error[-1][2]  # error component of last prediction
            i += 1

        print("Mean error of the {0} predicitons: {1}".format(num_predicitons, mean_error / num_predicitons))

    def plot_predictions(self, ax):
        num_predicitons = len(self.prediciton_truth_error)
        if num_predicitons > 1:
            for i in range(0, len(self.prediciton_truth_error)):
                ax[i].plot(self.prediciton_truth_error[i][0], 'r',
                           label='predictions; RMSE: {}'.format(self.prediciton_truth_error[i][2]))
                ax[i].plot(self.prediciton_truth_error[i][1], 'b', label='Truth')
                ax[i].legend()
        else:
            xticks = self.target.index[self.start:self.start + self.future_target]
            ax[1].plot(xticks, self.prediciton_truth_error[0][0],
                       label='predictions; RMSE: {}'.format(self.prediciton_truth_error[0][2]))
            ax[1].plot(xticks, self.prediciton_truth_error[0][1], label='Truth')
            ax[1].legend()
            ax[1].set_ylabel("RESIDUAL")
