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
        self.sarima_model=None

    def predict(self, component, method, offset=0,
                use_auto_arima=False, axis=None):
        if method == "AutoReg":
            self.autoreg(offset=offset, data_component=component)
        elif method == "sarima":
            self.sarima(offset=offset, use_auto_arima=use_auto_arima,
                        data_component=component)
        elif method == "naive_persistence":
            self.naive_lagged(data_component=component,offset=offset)
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
        if axis is not None:
            self.plot_prediction(axis, method)

    def naive_lagged(self, data_component,offset=0):
        self.pred = self.data[data_component].iloc[
                    self.start+offset - 2:self.start+offset - 2 + self.future_target]

    def naive0(self, data_component):
        self.pred = np.zeros(self.future_target)

    # LatexAutoRegMarkerStart
    def autoreg(self, data_component, offset=0):
        train = self.data[data_component].iloc[self.start + offset - 200
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
    def fit_sarima_model(self,datacolumn,start,exog):
        length=200
        train = self.data[datacolumn].iloc[start - length:start]
        model = pm.auto_arima(train,
                              #exogenous=exog.iloc[start - length:start],
                              start_p=1, start_q=1,
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
        with open('./checkpoints/arima_model.pkl', 'wb') as pkl:
            pickle.dump(model, pkl)

    def sarima(self, use_auto_arima, data_component, offset=0):
        start = self.start + offset
        exog=self.data[[data_component, "wind", "cloudiness",
                                 "air_temperature", "sun", 'DayOfWeek',
                                 'Hour', 'Holiday']]
        if self.sarima_model is None:
            if use_auto_arima:
                self.fit_sarima_model(data_component,start,exog)
            with open('./checkpoints/arima_model.pkl', 'rb') as pkl:
                self.sarima_model = pickle.load(pkl)
        interval=1
        if offset%interval==0 and offset >0:
            update_values = self.data[data_component].iloc[start-interval:start]
            self.sarima_model.update(update_values)#,exogenous=exog.iloc[start-interval:start]

        prediction = self.sarima_model.predict(n_periods=self.future_target)#,exogenous=exog.iloc[start:start+self.future_target]
        self.pred = prediction

    def mass_predict(self, iterations, axis, method, component,
                     step=1, save=False):
        j = 0
        single_errorlist = np.empty(
            [round(iterations / step), self.future_target])
        offsets = range(0, iterations, step)
        error_array = np.empty((iterations + self.future_target, 1))
        error_array[:] = np.nan
        max_iter = round(iterations / step)
        for i in offsets:

            print("\rmass predict {}: {}/{}".format(method, j, max_iter),
                  sep=' ', end='', flush=True)

            self.predict(offset=i, method=method, component=component,
                         use_auto_arima=False)

            # overwrite error since it doesnt account for offset
            start = self.start + i
            truth = self.data[component].iloc[
                    start:start + self.future_target]
            self.error = np.around(
                np.sqrt(np.mean(np.square(truth - self.pred))), 2)
            if save:
                plt.plot(truth.index, self.pred,
                         label="pred {}".format(self.error))
                plt.plot(truth.index, truth,
                         label="truth")
                plt.legend()
                if method == "sarima":
                    plt.savefig(
                        "./Abbildungen/{}/prediction_{}.png".format(
                            method,i), dpi=300,
                        bbox_inches='tight')
                else:
                    plt.savefig(
                        "./Abbildungen/{}/prediction_{}.png".format(
                            method, i), dpi=300, bbox_inches='tight')
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

        mean_error_over_time = [np.mean(error_array[x - 24:x + 12])
                                for x in
                                range(12, len(error_array) - 12)]
        x_ticks = self.data.index[
                  self.start:self.start + iterations + self.future_target]
        max_mean_error = max(error_array)
        max_timestep=np.where(error_array==max_mean_error)
        min_mean_error = min(error_array)
        min_timestep=np.where(error_array==min_mean_error)
        print(method,"max :",max_mean_error,"at:",x_ticks[max_timestep[0]][0], "min: ",min_mean_error,"at:",x_ticks[min_timestep[0]][0])
        axis.plot(x_ticks, error_array,
                  label="mean error at timestep. Overall mean: {}".format(
                      np.around(np.mean(cumulative_errorlist), 2)))
        axis.plot(x_ticks[12:-12], mean_error_over_time,
                  label="Moving average in 25 hour window")
        axis.set_title(method)
        axis.legend()

    def plot_prediction(self, ax, method):

        ax.plot(self.truth.index, self.pred,
                label='prediction; RMSE: {}'.format(self.error))
        ax.plot(self.truth.index, self.truth, label='Truth')
        ax.set_title(method)
        ax.legend()
