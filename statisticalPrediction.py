import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import pmdarima as pm
import pickle
import matplotlib.pyplot as plt


class StatisticalPrediction:

    def __init__(self, data, future_target,
                 test_split_at_hour):
        self.start = -test_split_at_hour
        self.data = data
        self.future_target = future_target

    # best Length 190 Best ARIMA(8, 0, 2) MSE=2.422
    def predict(self, component, method, offset=0,
                use_auto_arima=False):
        if method == "AutoReg":
            self.autoreg(offset=offset, data_component=component)
        elif method == "arima":
            self.arima(offset=offset, use_auto_arima=use_auto_arima,
                       data_component=component)
        elif method == "naiveLagged":
            self.naive_lagged(data_component=component)
        elif method == "naive0":
            self.naive0(data_component=component)
        else:
            print(
                "{} is not recognized. Make sure it is written correctly. Efaulting to naive0 Prediction".format(
                    method))
            self.naive_lagged(data_component=component)
        self.truth = self.data[component].iloc[
                     self.start + offset:self.start + self.future_target + offset]

        try:  # TODO Hacky workaround für umgehen von .value call auf numpy array
            self.error = np.around(np.sqrt(np.mean(
                np.square(self.truth.values - self.pred.values))), 2)
        except:
            self.error = np.around(
                np.sqrt(np.mean(np.square(self.truth - self.pred))), 2)

    def naive_lagged(self, data_component):
        self.pred = self.data[data_component].iloc[
                    self.start - 2:self.start - 2 + self.future_target]

    def naive0(self, data_component):
        self.pred = np.zeros(self.future_target)

    # LatexAutoRegMarkerStart
    def autoreg(self, data_component, offset=0):
        train = self.data[data_component].iloc[self.start + offset - 300
                                               :self.start + offset].asfreq(
            "H")
        lags = ar_select_order(endog=train, maxlag=70)
        model = AutoReg(train, lags=lags.ar_lags)
        model_fit = model.fit()
        self.pred = model_fit.predict(start=len(train),
                                      end=len(train)
                                          + self.future_target - 1,
                                      dynamic=False)

    # LatexAutoRegMarkerEnd

    def arima(self, use_auto_arima, data_component, offset=0):
        start = self.start + offset
        train = self.data[data_component].iloc[start - 200:start]
        if use_auto_arima == True:
            model = pm.auto_arima(train, start_p=1, start_q=1,
                                  test='adf',
                                  # use adftest to find optimal 'd'
                                  max_p=4, max_q=12,  # maximum p and q
                                  m=24,  # frequency of series
                                  d=None,  # let model determine 'd'
                                  seasonal=True,
                                  start_P=0,
                                  D=0,
                                  trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)
            print("model summary:", model.summary())
            if offset == 0:
                with open('./checkpoints/arima_model.pkl', 'wb') as pkl:
                    pickle.dump(model, pkl)
        else:
            with open('./checkpoints/arima_model.pkl', 'rb') as pkl:
                model = pickle.load(pkl)

        prediction = model.predict(n_periods=self.future_target)
        self.pred = prediction

    def mass_predict(self, iterations,method,component, use_auto_arima=False,
                     step=1, ):
        j = 0
        single_errorlist = np.empty(
            [round(iterations / step), self.future_target])
        offsets = range(0, iterations, step)
        error_array = np.empty((iterations + self.future_target, 1))
        error_array[:] = np.nan
        max = round(iterations / step)
        for i in offsets:
            print("\rmass predict {}: {}/{}".format(method,j, max),
                  sep=' ', end='', flush=True)

            self.predict(offset=i, method=method, component=component,
                         use_auto_arima=use_auto_arima)

            # overwrite error since it doesnt account for offset
            start = self.start + i
            truth = self.data[component].iloc[
                    start:start + self.future_target]
            self.error = np.around(
                np.sqrt(np.mean(np.square(truth - self.pred))), 2)
            plt.plot(truth.index, self.pred,
                     label="pred {}".format(self.error))
            plt.plot(truth.index, truth,
                     label="truth")
            plt.legend()
            plt.savefig(
                "./Abbildungen/{}/prediction_{}.png".format(method,
                    i),
                dpi=300, bbox_inches='tight')
            plt.clf()
            # plt.show()

            single_errorlist[j] = np.around(
                np.sqrt(np.square(self.truth.values - self.pred)), 2)

            arr = np.nanmean([error_array[i:i + self.future_target],
                              single_errorlist[j].reshape(
                                  self.future_target, 1)], axis=0)
            error_array[i:i + self.future_target] = arr
            j += 1
        cumulative_errorlist = np.around(
            np.mean(single_errorlist, axis=0),
            decimals=2)

        mean_error_over_time = [np.mean(error_array[x - 12:x + 12])
                                for x in
                                range(12, len(error_array) - 12)]
        plt.plot(error_array,
                 label="mean error at timestep. Overall mean: {}".format(
                     np.around(np.mean(cumulative_errorlist), 2)))
        plt.plot(range(12, len(error_array) - 12),
                 mean_error_over_time,
                 label="Moving average in 25 hour window")
        plt.xticks(
            [x for x in range(0, iterations + self.future_target, 12)])
        plt.legend()
        plt.show()
