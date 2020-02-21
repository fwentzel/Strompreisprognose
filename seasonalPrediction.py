from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np


class SeasonalPrediction:

    def __init__(self, data, forecast_length, train_length, start):
        self.forecast_length = forecast_length
        self.truth = data["Seasonal"].iloc[start:start + forecast_length]
        self.train = data["Seasonal"].iloc[start - train_length:start]

        self.model = ExponentialSmoothing(self.train, trend="add", seasonal="add", seasonal_periods=24,
                                          damped=True, freq="H")

    def fit_model(self):
        self.fit = self.model.fit()

    def predict(self):
        self.pred = self.fit.forecast(self.forecast_length)
        self.error = np.sqrt(np.mean(np.square(self.truth.values - self.pred.values)))

    def plot_predictions(self, ax):
        ax[2].plot(self.truth.index, self.truth.values, label='truth')
        ax[2].plot(self.truth.index, self.pred, label="prediction; Error: {0}".format(self.error))
        ax[2].legend()
        ax[2].set_ylabel("SEASONAL")
