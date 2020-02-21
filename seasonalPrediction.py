from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import matplotlib as plt

class SeasonalPrediction:

    def __init__(self,data,forecast_length,train_length,start):

        self.forecast_length=forecast_length
        self.truth = data["Power_Seasonal"].iloc[start:start + forecast_length]
        self.train = data["Power_Seasonal"].iloc[start-train_length:start]

        self.model = ExponentialSmoothing(self.train, trend="add", seasonal="add",seasonal_periods=24,
                                                   damped=True,freq="H")



    def fit_model(self):
        self.fit = self.model.fit()

    def predict(self):
        self.pred = self.fit.forecast(self.forecast_length)
        self.error = np.sqrt(np.mean(np.square(self.truth.values - self.pred.values)))
        self.prediciton_truth_error=[self.pred, self.truth.values, self.error]


    def plot_predictions(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.train.index[-24:], self.train.values[-24:])
        ax.plot(self.truth.index, self.truth.values, label='truth')
        ax.plot(self.truth.index, self.pred, linestyle='--', color='#3c763d',
                label="damped (RMSE={:0.2f}, AIC={:0.2f})".format(self.error, self.fit.aic))
        ax.legend()
        ax.set_title("Holt-Winter's Seasonal Smoothing")
        plt.show()

