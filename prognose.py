from __future__ import absolute_import, division, print_function, \
    unicode_literals

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from neuralNetPrediction import NeuralNetPrediction
from statisticalPrediction import StatisticalPrediction
import ipykernel  # fix progress bar

register_matplotlib_converters()
future_target = 24

mass_predict_neural = True
iterations = 24 * 7  # amount of predicitons for mass predict
step = 3

train_complete = False
train_residual = False
train_day_of_week = False

parser = ArgumentParser()
parser.add_argument('-start_hour', type=int, default=0)

args = parser.parse_args()

test_pred_start_hour = args.start_hour

predict_complete = False
predict_remainder = False
predict_decomposed = True
predict_naive = False
predict_with_day = False

predict_arima = True

test_length = future_target + 24 * 7  # Timesteps for testing.
data = get_data()
test_split_at_hour = data.index[
                         -test_length].hour - test_pred_start_hour + test_length

full_prediciton = None
if predict_complete:
    full_prediciton = NeuralNetPrediction(datacolumn="Price",
                                          data=data,
                                          future_target=future_target,
                                          test_split_at_hour=test_split_at_hour,
                                          net_type="complete",
                                          train_day_of_week=train_day_of_week)

    if train_complete:
        full_prediciton.initialize_network()
        full_prediciton.train_network(
            savename="trainedLSTM_completeUTC",
            save=False,
            lr_schedule="polynomal",
            power=2)  # lr_schedule="polynomal" oder "step

    else:
        full_prediciton.load_model(savename="trainedLSTM_completeUTC")

    if mass_predict_neural:
        full_prediciton.mass_predict(iterations=iterations,
                                     step=step,
                                     use_day_model=predict_with_day)

    else:
        full_prediciton.predict(offset=0,
                                use_day_model=predict_with_day)

res_prediction = NeuralNetPrediction(datacolumn="Remainder",
                                     data=data,
                                     future_target=future_target,
                                     test_split_at_hour=test_split_at_hour,
                                     net_type="remainder",
                                     train_day_of_week=train_day_of_week)
statistical_pred = StatisticalPrediction(data=data,
                                         future_target=future_target,
                                         test_split_at_hour=test_split_at_hour,
                                         )


if predict_arima:
    statistical_pred.predict(method="ARIMA", component="Price",use_auto_arima=False)
    statistical_pred.mass_predict(iterations=iterations,
                                  step=1)

decomp_error = 0
sum_pred = 0
j = 0
single_errorlist = np.empty([round(iterations / step), future_target])
error_array = np.empty((iterations + future_target, 1))
error_array[:] = np.nan
decomposed_iterations = iterations if mass_predict_neural else 1
max = round(iterations / step)
offsets = range(0, iterations, step)

for i in offsets:
    if decomposed_iterations > 1:
        print("\rmass predict decomposed: {}/{}".format(j, max),
              sep=' ', end='', flush=True)
    if predict_remainder or predict_decomposed:
        # Residual
        if train_residual:
            res_prediction.initialize_network()
            res_prediction.train_network(
                savename="trainedLSTM_residUTC",
                save=False,
                lr_schedule="polynomal",
                power=2)
            # lr_schedule="polynomal" oder "step
        else:
            res_prediction.load_model(savename="trainedLSTM_residUTC")

        if mass_predict_neural and not predict_decomposed:
            res_prediction.mass_predict(iterations=iterations,
                                        step=step,
                                        use_day_model=predict_with_day)
        else:
            # Remainder
            res_prediction.predict(offset=i,
                                   use_day_model=predict_with_day)

    if predict_decomposed:
        # copy so original doesnt get overwritten when adding other components
        sum_pred = res_prediction.pred.copy()

        # Seasonal
        statistical_pred.predict(method="AutoReg", component="Seasonal")
        sum_pred += statistical_pred.pred

        # Trend
        statistical_pred.predict(method="AutoReg", component="Trend")
        sum_pred += statistical_pred.pred

        x, x_diff = data['Price'].iloc[
                        i + test_split_at_hour - 1], sum_pred
        sum_pred = np.r_[x, x_diff].cumsum().astype(float)[1:]

        # add error
        timeframe = slice(i - test_split_at_hour,
                          future_target + i - test_split_at_hour)
        truth = data["Price"].iloc[timeframe]
        decomp_error = np.around(
            np.sqrt(np.mean(np.square(truth - sum_pred))), 2)
        single_errors = np.around(
            np.sqrt(np.square(truth.values - sum_pred)), 2)
        if mass_predict_neural:
            single_errorlist[j] = single_errors
            arr = np.nanmean([error_array[i:i + future_target],
                              single_errorlist[j].reshape(
                                  future_target, 1)], axis=0)
            error_array[i:i + future_target] = arr
        j += 1
if mass_predict_neural:
    cumulative_errorlist = np.around(np.mean(single_errorlist, axis=0),
                                     decimals=2)

    mean_error_over_time = [np.mean(error_array[x - 12:x + 12])
                            for x in
                            range(12, len(error_array) - 12)]
    plt.plot(error_array,
             label="Cumulative error. Overall mean: {}".format(
                 np.around(np.mean(cumulative_errorlist), 2)))
    plt.plot(range(12, len(error_array) - 12),
             mean_error_over_time,
             label="Moving average in 25 hour window")
    plt.xticks(
        [x for x in range(0, iterations + future_target, 12)])
    plt.legend()
    plt.show()

if predict_naive:

    statistical_pred.predict(method="naiveLagged",component="Price")
    print("Price naive Prediction error: {}".format(statistical_pred.error))
    statistical_pred.predict(method="naive0",component="Remainder")
    print("Remainder naive Prediction error: {}".format(statistical_pred.error))

