import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class ResidualPrediction:
    RELEVANT_COLUMNS = ['V_N', 'SD_SO', 'F', 'Temp', 'Weekend', 'Hour',
                             'Holiday', 'Residual']
    TRAIN_LENGTH = .8  # percent
    BATCH_SIZE = 256
    EPOCHS = 1

    def __init__(self, train_data,test_data, future_target, past_history, start_index_from_max_length):
        self.train_target = train_data["Residual"]
        self.train_dataset = train_data[self.RELEVANT_COLUMNS].values
        self.test_target = test_data["Residual"]
        self.test_dataset = test_data[self.RELEVANT_COLUMNS].values
        self.future_target = future_target  # timesteps into future
        self.start_index_from_max_length=start_index_from_max_length
        self.past_history = past_history  # inputtimesteps

        self.x, self.y = self.multivariate_data_single_step()
        self.test_split = int(len(self.train_dataset) * self.TRAIN_LENGTH)
        self.prediction_truth_error = []

        #TODO workarund
        self.predicted_test=False

    def initialize_network(self, learning_rate):
        # define model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.GRU(self.past_history,
                                      return_sequences=True, input_shape=(self.x.shape[-2:])))
        model.add(tf.keras.layers.GRU(int(self.past_history / 2)))
        # multi_step_model.add(tf.keras.layers.GRU(past_history))
        model.add(tf.keras.layers.Dense(self.future_target))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),loss="mae")
        self.model = model

    def load_model(self,checkpoint):
        self.model = tf.keras.models.load_model('./checkpoints/{0}'.format(checkpoint))

    def multivariate_data_single_step(self):
        multivariate_data = []
        labels = []
        for i in range(self.past_history, len(self.train_dataset) - 1):
            indices = range(i - self.past_history + 1, i + 1)
            multivariate_data.append(self.train_dataset[indices])
            labels.append(self.train_target[i + 1])
        multivariate_data = np.array(multivariate_data)
        return multivariate_data.reshape(multivariate_data.shape[0], multivariate_data.shape[1], -1), np.array(labels)

    def train_network(self, checkpoint):
        multi_step_history = self.model.fit(x=self.x, y=self.y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                                            verbose=2,
                                            validation_split=1 - self.TRAIN_LENGTH, shuffle=True)
        self.plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
        self.model.save('./checkpoints/{0}'.format(checkpoint))


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
        x_in = inputCopy[:self.past_history].reshape(-1, self.past_history, inputCopy.shape[1])
        prediction=self.model.predict(x_in)
        predictions.append(prediction[0][-1])

        for j in range(1, self.future_target):
            x_in = inputCopy[j:j + self.past_history].reshape(-1, self.past_history, inputCopy.shape[1])
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
        max_offset = len(dataset) - self.past_history - self.future_target - start
        mean_error = 0
        for i in range(0, num_predicitons):
            if random_offset:
                offset = random.randrange(max_offset)  # random offset for the new train_data
            else:
                offset = i
            prediction_timeframe = slice(start - self.past_history - self.future_target + offset,
                                         start + offset)
            input=dataset[prediction_timeframe]
            result=self.single_step_predict(inputs=input,target=self.train_target.iloc[start:start + self.future_target])
            self.prediction_truth_error.append(result)
            mean_error += self.prediction_truth_error[-1][2]  # error component of last prediction
            i += 1

        print("Mean error of the {0} predicitons: {1}".format(num_predicitons, mean_error / num_predicitons))

    def plot_predictions(self, ax):
        if(self.predicted_test):
            start = len(self.test_dataset) - self.start_index_from_max_length
            xticks = self.test_target.index[start:start + self.future_target]
        else:
            start = len(self.train_dataset) - self.start_index_from_max_length
            xticks = self.train_target.index[start:start + self.future_target]

        ax[1].plot(xticks, self.prediction_truth_error[0][0],
                   label='predictions; RMSE: {}'.format(self.prediction_truth_error[0][2]))
        ax[1].plot(xticks, self.prediction_truth_error[0][1], label='Truth')
        ax[1].legend()
        ax[1].set_ylabel("RESIDUAL")
