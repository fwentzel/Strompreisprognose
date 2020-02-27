import numpy as np
from statsmodels.tsa.arima_model import ARIMA


class SeasonalPrediction:

    def __init__(self, data, forecast_length, start_index_from_max_length):
        self.data = data
        self.start=len(data)-start_index_from_max_length
        self.forecast_length = forecast_length
        self.truth = data["Seasonal"].iloc[self.start:self.start + forecast_length]
        self.train = data["Seasonal"].iloc[:self.start]
#best Length 150 Best ARIMA(6, 0, 2) MSE=3.671



    def fit_model(self):
        self.model = ARIMA(self.train, order=(5,1,0), freq="H")
        self.model_fit = self.model.fit(disp=0)

    def predict(self):
        start_index = len(self.train)
        end_index = start_index + self.forecast_length-1
        prediction = self.model_fit.predict(start=start_index, end=end_index)
        self.pred=prediction
        self.error = np.around(np.sqrt(np.mean(np.square(self.truth - prediction))), 2)

    def plot_predictions(self, ax):
        ax[2].plot(self.truth.index, self.truth.values, label='truth')
        ax[2].plot(self.truth.index, self.pred, label="prediction; Error: {0}".format(self.error))
        ax[2].legend()
        ax[2].set_ylabel("SEASONAL")

    def test_orders(self):
        p_values = [0, 1, 2,3, 4,5, 6,7, 8]
        d_values = range(0, 3)
        q_values = range(0, 3)
        lengths=range(150,250)

        best_score, best_cfg,best_length = float("inf"), None,None
        for l in lengths:
            data=self.data["Seasonal"].iloc[self.start-l:self.start]
            start_index = len(data)
            end_index = start_index + self.forecast_length - 1
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        order = (p, d, q)
                        try:
                            model = ARIMA(data, order=order, freq="H")
                            model_fit = model.fit(disp=0)
                            prediction = model_fit.predict(start=start_index, end=end_index)
                            rmse = np.sqrt(np.mean(np.square(self.truth - prediction)))
                            if rmse < best_score:
                                best_score, best_cfg,best_length = rmse, order,l
                            print('%i ARIMA%s RMSE=%.3f ' % (l,order, rmse))
                        except:
                            continue
        print('best Length %i Best ARIMA%s MSE=%.3f ' % (best_length,best_cfg, best_score))