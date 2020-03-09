import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow_core.python.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


class RemainderPrediction:


    TRAIN_LENGTH = .6  # percent
    BATCH_SIZE = 64
    EPOCHS =500

    def __init__(self, train_data,test_data, future_target, past_history, start_index_from_max_length,datacolumn):
        self.RELEVANT_COLUMNS = ['Wind', 'Sun', 'Clouds', 'Temperature', 'Weekend', 'Hour',
                            'Holiday', datacolumn]

        self.train_target = train_data[datacolumn]
        self.train_dataset = train_data[self.RELEVANT_COLUMNS].values
        self.test_target = test_data[datacolumn]
        self.test_dataset = test_data[self.RELEVANT_COLUMNS].values
        self.future_target = future_target  # timesteps into future
        self.start_index_from_max_length=start_index_from_max_length
        self.past_history = past_history  # inputtimesteps

        self.x, self.y = self.multivariate_data_single_step()

        #TODO workarund
        self.predicted_test=False

    def initialize_network(self, dropout, additional_layers):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(self.past_history, return_sequences=True, input_shape=(self.x.shape[-2:])))
        for i in range(additional_layers):
            model.add(tf.keras.layers.LSTM(self.past_history, return_sequences=True, dropout=dropout))#0,3 3,7

        model.add(tf.keras.layers.LSTM(int(self.past_history)))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1), loss="mae")
        self.model = model

    def load_model(self,savename):
        try:
            self.model = tf.keras.models.load_model('.\checkpoints\{0}'.format(savename))
        except:
            self.model = tf.keras.models.load_model('.\checkpoints\default')


    def multivariate_data_single_step(self):
        multivariate_data = []
        labels = []
        for i in range(self.past_history, len(self.train_dataset) - 1):
            indices = range(i - self.past_history + 1, i + 1)
            multivariate_data.append(self.train_dataset[indices])
            labels.append(self.train_target[i + 1])
        multivariate_data = np.array(multivariate_data)
        return multivariate_data.reshape(multivariate_data.shape[0], multivariate_data.shape[1], -1), np.array(labels)

    def train_network(self,savename,power=1,initAlpha=0.1,lr_schedule="polynomal"):
        if lr_schedule=="polynomal":
            if power is None:
                power=1
            schedule = PolynomialDecay(maxEpochs=self.EPOCHS, initAlpha=initAlpha, power=power)
        elif lr_schedule=="step":
            schedule = StepDecay(initAlpha=initAlpha, factor=0.8, dropEvery=10)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30)
        mc = ModelCheckpoint('%s.h5p'%savename, monitor='val_loss', mode='min', save_best_only=True,verbose=1)
        multi_step_history = self.model.fit(x=self.x, y=self.y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                                            verbose=0,validation_split=1 - self.TRAIN_LENGTH, shuffle=True,callbacks=[es])#TODO include ModelCheckpoint callback
        #schedule.plot(self.EPOCHS)
        #self.plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

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
        target_rows=target.iloc[-self.future_target:]
        self.truth = target_rows.values
        self.pred =  pd.Series(np.array(predictions).reshape(self.future_target),index= target_rows.index)
        self.error = np.around(np.sqrt(np.mean(np.square(self.truth - predictions))), 2)


    def predict(self,predict_test=False, offset=0):
        if(predict_test ==False):
            dataset=self.train_dataset
            target=self.train_target
        else:
            dataset = self.test_dataset
            target = self.test_target
            self.predicted_test=True
        start= len(dataset) - self.start_index_from_max_length
        prediction_timeframe = slice(start - self.past_history - self.future_target + offset,
                                     start + offset)
        input=dataset[prediction_timeframe]
        self.single_step_predict(inputs=input,target=target.iloc[start+offset:start+offset + self.future_target])

    def mass_predict(self,iterations,predict_on_test_data,step=1,):
        error=0
        j=0
        for i in range(0, iterations, step):
            j+=1
            self.predict(predict_test=predict_on_test_data, offset=i)
            error+=self.error
        return error/j



    def plot_predictions(self, ax):
        if(self.predicted_test):
            start = len(self.test_dataset) - self.start_index_from_max_length
            xticks = self.test_target.index[start:start + self.future_target]
        else:
            start = len(self.train_dataset) - self.start_index_from_max_length
            xticks = self.train_target.index[start:start + self.future_target]

        ax[1].plot(xticks, self.pred,
                   label='predictions; RMSE: {}'.format(self.error ))
        ax[1].plot(xticks, self.truth, label='Truth')
        ax[1].legend()
        ax[1].set_ylabel("RESIDUAL")



class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch

		lrs = [self(i) for i in range(0, epochs)]
		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(range(0, epochs), lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")


class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power
	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
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

