import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class ResidualPrediction:

    def __init__(self, train_data,test_data, future_target, past_history, start_index_from_max_length):
        self.relevant_columns = ['V_N', 'SD_SO', 'F', 'Temp', 'Weekend', 'Hour', 'Holiday', 'Residual']
        self.train_target = train_data["Residual"]
        self.train_dataset = train_data[self.relevant_columns].values
        self.test_target = test_data["Residual"]
        self.test_dataset = test_data[self.relevant_columns].values
        self.future_target = future_target  # timesteps into future
        self.start_index_from_max_length=start_index_from_max_length
        self.PAST_HISTORY = past_history  # inputtimesteps

        self.TRAIN_LENGTH = .8  # percent
        self.STEP = 1
        self.BATCH_SIZE = 256
        self.BUFFER_SIZE = 500
        self.EPOCHS = 50
        self.TRAIN_SPLIT = int(len(self.train_dataset) * self.TRAIN_LENGTH)
        self.model = None
        self.x, self.y = self.multivariate_data_single_step()
        self.prediciton_truth_error = []

        #TODO workarund
        self.predicted_test=False

    def initialize_network(self, learning_rate):
        # define model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.GRU(self.PAST_HISTORY,
                                      return_sequences=True, input_shape=(self.x.shape[-2:])))
        model.add(tf.keras.layers.GRU(int(self.PAST_HISTORY / 2)))
        # multi_step_model.add(tf.keras.layers.GRU(past_history))
        model.add(tf.keras.layers.Dense(self.future_target))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),loss="mae")
        self.model = model

    def multivariate_data_single_step(self):
        multivariate_data = []
        labels = []
        for i in range(self.PAST_HISTORY, len(self.train_dataset) - 1):
            indices = range(i - self.PAST_HISTORY + 1, i + 1)
            multivariate_data.append(self.train_dataset[indices])
            labels.append(self.train_target[i + 1])
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

    def single_step_predict(self, inputs, target=None):
        predictions = []
        inputCopy = np.copy(inputs)  # copy Array to not overwrite original train_data
        x_in = inputCopy[:self.PAST_HISTORY].reshape(-1, self.PAST_HISTORY, inputCopy.shape[1])
        prediction=self.model.predict(x_in)
        predictions.append(prediction[0][-1])

        for j in range(1, self.future_target):
            x_in = inputCopy[j:j + self.PAST_HISTORY].reshape(-1, self.PAST_HISTORY, inputCopy.shape[1])
            x_in[-1, -1, 0] = predictions[j - 1]  # use last prediction as power Price Input for next Prediction
            predictions.append(self.model.predict(x_in)[0][-1])
        truth = target.iloc[-self.future_target:].values
        predictions = np.array(predictions)
        error = np.around(np.sqrt(np.mean(np.square(truth - predictions))), 2)

        return predictions, truth, error

    def predict(self,predict_test=False, num_predicitons=2, random_offset=True):
        if(predict_test ==False):
            dataset=self.train_dataset
        else:
            dataset = self.test_dataset
            self.predicted_test=True
        start= len(dataset) - self.start_index_from_max_length
        max_offset = len(dataset) - self.PAST_HISTORY - self.future_target -start
        mean_error = 0
        for i in range(0, num_predicitons):
            if random_offset:
                offset = random.randrange(max_offset)  # random offset for the new train_data
            else:
                offset = i
            prediction_timeframe = slice(start - self.PAST_HISTORY - self.future_target + offset,
                                         start + offset)
            input=dataset[prediction_timeframe]
            self.prediciton_truth_error.append(self.single_step_predict(inputs=input,
                                                                        target=self.train_target.iloc[
                                                                               start:start + self.future_target]))
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
            if(self.predicted_test):
                start = len(self.test_dataset) - self.start_index_from_max_length
                xticks = self.test_target.index[start:start + self.future_target]
            else:
                start = len(self.train_dataset) - self.start_index_from_max_length
                xticks = self.train_target.index[start:start + self.future_target]

            ax[1].plot(xticks, self.prediciton_truth_error[0][0],
                       label='predictions; RMSE: {}'.format(self.prediciton_truth_error[0][2]))
            ax[1].plot(xticks, self.prediciton_truth_error[0][1], label='Truth')
            ax[1].legend()
            ax[1].set_ylabel("RESIDUAL")
