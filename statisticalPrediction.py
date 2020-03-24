import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AR


class StatisticalPrediction:

    def __init__(self, data, forecast_length,
                 neural_past_history,offset=0):
        self.data = data
        self.start  = neural_past_history + offset
        self.forecast_length = forecast_length
        self.truth = data["Seasonal"].iloc[self.start:self.start + forecast_length]

    # best Length 190 Best ARIMA(8, 0, 2) MSE=2.422
    def predict(self, method,component ):
        if method == "AR":
            self.AR_prediction(component)
        elif method == "ARIMA":
            self.arima_prediction(component)
        elif method == "exp":
            self.exponential_smoothing_prediction(component)

    def AR_prediction(self,component):
        train = self.data["Seasonal"].iloc[:self.start]
        model = AR(train, freq="H")
        model_fit = model.fit()
        self.pred = model_fit.predict(start=len(train),
                                      end=len(train) + self.forecast_length - 1,
                                      dynamic=False)
        self.error = np.sqrt(
            np.mean(np.square(self.truth.values - self.pred.values)))

    def exponential_smoothing_prediction(self,component):
        train = self.data[component].iloc[self.start - 72:self.start]
        model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                     seasonal_periods=24,
                                     damped=True, freq="H")
        # Search Best Smoothing Level
        min_error = 100
        best_smoothing_level = .2
        # for i in np.arange(0, 1, .01):
        fit = model.fit(smoothing_level=.2)
        # print(fit.mle_retvals)
        start = len(train)
        end = len(train) + self.forecast_length - 1
        pred = fit.predict(start, end)
        error = np.sqrt(np.mean(np.square(self.truth.values - pred.values)))
        if error < min_error:
            min_error = error
            self.pred = pred
            self.error = min_error
        return self.pred

    def arima_prediction(self, test_orders=False, order=(15, 0, 2)):
        if test_orders == True:
            order = self.test_orders()

        train = self.data["Seasonal"].iloc[:self.start]
        model = ARIMA(train, order=order, freq="H")
        model_fit = model.fit(disp=0)
        start_index = len(train)
        end_index = start_index + self.forecast_length - 1
        prediction = model_fit.predict(start=start_index, end=end_index)
        self.pred = prediction
        self.error = np.around(
            np.sqrt(np.mean(np.square(self.truth - prediction))), 2)



    def test_orders(self):
        p_values = range(15, 90)
        q_values = range(0, 5)
        d_values = range(0, 3)

        best_score, best_cfg, best_length = float("inf"), None, None
        data = self.data["Seasonal"].iloc[:self.start]
        start_index = len(data)
        end_index = start_index + self.forecast_length - 1
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)  # immer 0, da Series schon stationary ist
                    try:
                        model = ARIMA(data, order=order, freq="H")
                        model_fit = model.fit(disp=0)
                        prediction = model_fit.predict(start=start_index,
                                                       end=end_index)
                        rmse = np.sqrt(
                            np.mean(np.square(self.truth - prediction)))
                        if rmse < best_score:
                            best_score, best_cfg = rmse, order
                    except:
                        continue
        print('Best ARIMA%s MSE=%.3f ' % (best_cfg, best_score))
        return best_cfg
