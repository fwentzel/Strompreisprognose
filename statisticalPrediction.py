import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


class StatisticalPrediction:

    def __init__(self, data, future_target,
                 neural_past_history, component, offset=0):
        self.component = component
        self.data = data
        self.start = neural_past_history + offset
        self.future_target = future_target
        self.truth = data[component].iloc[
                     self.start:self.start + future_target]

    # best Length 190 Best ARIMA(8, 0, 2) MSE=2.422
    def predict(self, method):
        if method == "AutoReg":
            self.AutoReg_prediction()
        elif method == "ARIMA":
            self.arima_prediction()
        elif method == "naive":
            self.naive_prediction()
        try:  # TODO Hacky workaround für umgehen von .value call auf numpy array
            self.error = np.sqrt(
                np.mean(
                    np.square(self.truth.values - self.pred.values)))
        except:
            self.error = np.sqrt(
                np.mean(
                    np.square(self.truth - self.pred)))

    def naive_prediction(self):
        self.pred = self.data[self.component].iloc[
                    self.start - 2:self.start - 2 + self.future_target]

    # self.pred = np.ones(shape=(self.future_target))*self.data[self.component].iloc[self.start-1]
    # LatexAutoRegMarkerStart
    def AutoReg_prediction(self):
        train = self.data[self.component].iloc[:self.start].asfreq("H")
        lags = ar_select_order(endog=train,
                               maxlag=len(train) - 50)
        model = AutoReg(train, lags=lags.ar_lags)
        model_fit = model.fit()
        self.pred = model_fit.predict(start=len(train),
                                      end=len(
                                          train) + self.future_target - 1,
                                      dynamic=False)
        # LatexAutoRegMarkerEnd
        # **** IF USE EXOG VARIABLES ****
        # exog_components = ["wind", "cloudiness", "air_temperature",
        #                    "sun", 'Weekend', 'Hour', 'Holiday']
        # exog = self.data[exog_components].iloc[
        #        :self.start + self.future_target].asfreq("H")
        # lags = ar_select_order(endog=train,
        #                        exog=exog.iloc[:self.start],
        #                        maxlag=len(train) - 50)
        # lags = lags.ar_lags
        # model = AutoReg(train, lags=lags,
        #                 exog=exog.iloc[:self.start])
        # model_fit = model.fit()
        # self.pred = model_fit.predict(start=len(train),
        #                               end=len(
        #                                   train) + self.future_target - 1,
        #                               dynamic=False,
        #                               exog_oos=exog.iloc[
        #                                        self.start:])

    def arima_prediction(self, test_orders=False, order=(15, 0, 2)):
        if test_orders == True:
            order = self.test_orders()

        train = self.data[self.component].iloc[:self.start]
        model = ARIMA(train, order=order, freq="H")
        model_fit = model.fit(disp=0)
        start_index = len(train)
        end_index = start_index + self.future_target - 1
        prediction = model_fit.predict(start=start_index, end=end_index)
        self.pred = prediction

    def test_orders(self):
        p_values = range(15, 90)
        q_values = range(0, 5)
        d_values = range(0, 3)

        best_score, best_cfg, best_length = float("inf"), None, None
        data = self.data[self.component].iloc[:self.start]
        start_index = len(data)
        end_index = start_index + self.future_target - 1
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (
                        p, d,
                        q)  # immer 0, da Series schon stationary ist
                    try:
                        model = ARIMA(data, order=order, freq="H")
                        model_fit = model.fit(disp=0)
                        prediction = model_fit.predict(
                            start=start_index,
                            end=end_index)
                        rmse = np.sqrt(
                            np.mean(np.square(self.truth - prediction)))
                        if rmse < best_score:
                            best_score, best_cfg = rmse, order
                    except:
                        continue
        print('Best ARIMA%s MSE=%.3f ' % (best_cfg, best_score))
        return best_cfg
