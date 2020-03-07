import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AR


class SeasonalPrediction:

    def __init__(self, data, forecast_length, start_index_from_max_length,offset=0):
        self.data = data
        self.start=len(data)-start_index_from_max_length+offset
        self.forecast_length = forecast_length
        self.truth = data["Seasonal"].iloc[self.start:self.start + forecast_length]

#best Length 190 Best ARIMA(8, 0, 2) MSE=2.422

    def AR_prediction(self):
        train = self.data["Seasonal"].iloc[:self.start]
        model = AR(train,freq="H")
        model_fit = model.fit()
        self.pred = model_fit.predict(start=len(train), end=len(train)+self.forecast_length-1, dynamic=False)
        self.error = np.sqrt(np.mean(np.square(self.truth.values - self.pred.values)))

    def exponential_smoothing_prediction(self,smoothing_level):
        train = self.data["Seasonal"].iloc[self.start - 72:self.start]
        model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=24,
                                          damped=True, freq="H")
        fit = model.fit(smoothing_level=smoothing_level)
        #print(fit.mle_retvals)
        start=len(train)
        end=len(train)+self.forecast_length-1

        self.pred = fit.predict(start,end)
        self.error = np.sqrt(np.mean(np.square(self.truth.values - self.pred.values)))


    def arima_prediction(self):
        train = self.data["Seasonal"].iloc[:self.start]
        model = ARIMA(train, order=(15,0,2), freq="H")
        model_fit = model.fit(disp=0)
        start_index = len(train)
        end_index = start_index + self.forecast_length-1
        prediction = model_fit.predict(start=start_index, end=end_index)
        self.pred=prediction
        self.error = np.around(np.sqrt(np.mean(np.square(self.truth - prediction))), 2)

    def plot_predictions(self, ax):
        ax[2].plot(self.truth.index, self.truth.values, label='truth')
        ax[2].plot(self.truth.index, self.pred, label="prediction; Error: {0}".format(self.error))
        ax[2].legend()
        ax[2].set_ylabel("SEASONAL")

    def test_orders(self):
        p_values = range(15, 90)
        q_values = range(0, 5)
        d_values = range(0, 3)

        best_score, best_cfg,best_length = float("inf"), None,None
        data=self.data["Seasonal"].iloc[:self.start]
        start_index = len(data)
        end_index = start_index + self.forecast_length - 1
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)# immer 0, da Series schon stationary ist
                    try:
                        model = ARIMA(data, order=order, freq="H")
                        model_fit = model.fit(disp=0)
                        prediction = model_fit.predict(start=start_index, end=end_index)
                        rmse = np.sqrt(np.mean(np.square(self.truth - prediction)))
                        if rmse < best_score:
                            best_score, best_cfg = rmse, order
                        print(' ARIMA%s RMSE=%.3f ' % (order, rmse))
                    except:
                        continue
        print('Best ARIMA%s MSE=%.3f ' % (best_cfg, best_score))