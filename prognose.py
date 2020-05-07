from __future__ import absolute_import, division, print_function, \
    unicode_literals
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from neuralNetPrediction import NeuralNetPrediction
from statisticalPrediction import StatisticalPrediction
import GUI
import ipykernel  # fix progress bar

register_matplotlib_converters()
use_setup_settings = False
settings_dict,needed_plots, used_plots=None,10,0
if use_setup_settings:
    config =GUI.main()
    if config is None:
        # User cancelled program
        import sys
        sys.exit()
    else:
        settings_dict,needed_plots = config


figure, axes = plt.subplots(nrows=3,ncols=2)

FUTURE_TARGET = 24
MASS_PREDICT = False
ITERATIONS = 24 * 7  # amount of predicitons for mass predict
STEP = 1

train_complete = settings_dict[
    "train_complete"] if use_setup_settings else False
train_remainder = settings_dict[
    "train_remainder"] if use_setup_settings else False
train_day_of_week = settings_dict[
    "train_day_of_week"] if use_setup_settings else False

predict_complete = settings_dict[
    "predict_complete"] if use_setup_settings else True
predict_remainder = settings_dict[
    "predict_remainder"] if use_setup_settings else True
predict_decomposed = settings_dict[
    "predict_decomposed"] if use_setup_settings else True
predict_naive_lagged = settings_dict[
    "predict_naive_lagged"] if use_setup_settings else True
predict_naive_0 = settings_dict[
    "predict_naive_0"] if use_setup_settings else True
predict_arima = settings_dict[
    "predict_arima"] if use_setup_settings else True

predict_with_day = settings_dict[
    "predict_with_day"] if use_setup_settings else False
test_pred_start_hour = settings_dict[
    "test_pred_start_hour"] if use_setup_settings else 0

if test_pred_start_hour <0:
    MASS_PREDICT=True
    test_pred_start_hour=0
else:
    MASS_PREDICT=False

test_length = FUTURE_TARGET + 24 * 7  # Timesteps for testing.
data = get_data()
test_split_at_hour = data.index[
                         -test_length].hour - test_pred_start_hour + test_length

if predict_complete:
    price_prediciton = NeuralNetPrediction(datacolumn="Price",
                                           data=data,
                                           future_target=FUTURE_TARGET,
                                           test_split_at_hour=test_split_at_hour,
                                           net_type="price_complete",
                                           train_day_of_week=train_day_of_week)

    if train_complete:
        price_prediciton.initialize_network()
        price_prediciton.train_network(
            savename="trainedLSTM_priceUTC",
            save=False,
            lr_schedule="polynomal",
            power=2)  # lr_schedule="polynomal" oder "STEP

    else:
        price_prediciton.load_model(savename="trainedLSTM_priceUTC")

    if MASS_PREDICT:
        price_prediciton.mass_predict(iterations=ITERATIONS,
                                      step=STEP,
                                      use_day_model=predict_with_day,
                                      axis=axes[0,0])

    else:
        price_prediciton.predict(offset=0,
                                 use_day_model=predict_with_day,
                                 axis=axes[0,0])
        used_plots+=1

rem_prediction = NeuralNetPrediction(datacolumn="Remainder",
                                     data=data,
                                     future_target=FUTURE_TARGET,
                                     test_split_at_hour=test_split_at_hour,
                                     net_type="remainder_complete",
                                     train_day_of_week=train_day_of_week)
statistical_pred = StatisticalPrediction(data=data,
                                         future_target=FUTURE_TARGET,
                                         test_split_at_hour=test_split_at_hour,
                                         )

if predict_arima:
    if MASS_PREDICT:
        # statistical_pred.predict(method="ARIMA", component="Price",
        #                          use_auto_arima=True, offset=0)
        statistical_pred.mass_predict(iterations=ITERATIONS,
                                      step=1,
                                      component="Price",
                                      use_auto_arima=False,
                                      method="arima",
                                      axis=axes[1,1])
    else:
        i = 0
        statistical_pred.predict(method="arima", component="Price",
                                 use_auto_arima=False, offset=i,axis=axes[1,1])
        used_plots+=1
        timeframe = slice(i - test_split_at_hour,
                          FUTURE_TARGET + i - test_split_at_hour)
        truth = data["Price"].iloc[timeframe]

components_combined_error = 0
sum_pred = 0
j = 0
single_errorlist = np.empty([round(ITERATIONS / STEP), FUTURE_TARGET])
error_array = np.empty((ITERATIONS + FUTURE_TARGET, 1))
error_array[:] = np.nan
decomposed_iterations = ITERATIONS if MASS_PREDICT and predict_decomposed else 1
max = round(decomposed_iterations / STEP)
offsets = range(0, decomposed_iterations, STEP)

for i in offsets:
    if decomposed_iterations > 1:
        print("\rmass predict decomposed: {}/{}".format(j, max),
              sep=' ', end='', flush=True)
    if predict_remainder or predict_decomposed:
        # Remainder
        if train_remainder:
            rem_prediction.initialize_network()
            rem_prediction.train_network(
                savename="trainedLSTM_remainderUTC",
                save=False,
                lr_schedule="polynomal",
                power=2)
            # lr_schedule="polynomal" oder "STEP
        elif rem_prediction.model is None:
            rem_prediction.load_model(savename="trainedLSTM_remainderUTC")

        if MASS_PREDICT and not predict_decomposed:
            rem_prediction.mass_predict(iterations=ITERATIONS,
                                        step=STEP,
                                        use_day_model=predict_with_day,
                                        axis=axes[1,0])
        else:
            # Remainder
            rem_prediction.predict(offset=i,
                                   use_day_model=predict_with_day, axis=axes[1,0])
        used_plots+=1

    if predict_decomposed:
        timeframe = slice(i - test_split_at_hour,
                          FUTURE_TARGET + i - test_split_at_hour)
        truth = data["Price"].iloc[timeframe]

        # copy so original doesnt get overwritten when adding other components
        sum_pred = rem_prediction.pred.copy()
        # plt.plot(rem_prediction.truth.index, sum_pred,
        #          label="pred {}".format(rem_prediction.error))
        # Seasonal
        statistical_pred.predict(method="AutoReg", component="Seasonal",
                                 offset=i)
        sum_pred += statistical_pred.pred
        # plt.plot(truth.index, statistical_pred.pred,
        #          label="pred {}".format(statistical_pred.error))
        # Trend
        statistical_pred.predict(method="AutoReg", component="Trend",
                                 offset=i)
        sum_pred += statistical_pred.pred
        # plt.plot(truth.index, statistical_pred.pred,
        #          label="pred {}".format(statistical_pred.error))
        x, x_diff = data['Price'].iloc[
                        i - test_split_at_hour - 1], sum_pred
        sum_pred = np.r_[x, x_diff].cumsum().astype(float)[1:]

        # add error

        components_combined_error = np.around(
            np.sqrt(np.mean(np.square(truth - sum_pred))), 2)
        single_errors = np.around(
            np.sqrt(np.square(truth.values - sum_pred)), 2)
        if MASS_PREDICT:
            single_errorlist[j] = single_errors
            arr = np.nanmean([error_array[i:i + FUTURE_TARGET],
                              single_errorlist[j].reshape(
                                  FUTURE_TARGET, 1)], axis=0)
            error_array[i:i + FUTURE_TARGET] = arr
            plt.plot(truth.index, sum_pred,
                     label="pred {}".format(components_combined_error))
            plt.plot(truth.index, truth,
                     label="truth")
            plt.legend()
            plt.savefig(
                "./Abbildungen/decomposed/prediction_{}.png".format(
                    i),
                dpi=300, bbox_inches='tight')
            plt.clf()
        else:

            axes[0,1].plot(truth.index, sum_pred,
                                  label='prediction; RMSE: {}'.format(components_combined_error))
            axes[0,1].plot(truth.index, truth, label='Truth')
            axes[0,1].set_title("components combined")
            axes[0,1].legend()


        j += 1
if MASS_PREDICT and predict_decomposed:
    cumulative_errorlist = np.around(np.mean(single_errorlist, axis=0),
                                     decimals=2)

    mean_error_over_time = [np.mean(error_array[x - 12:x + 12])
                            for x in
                            range(12, len(error_array) - 12)]

    axes[0,1].plot(error_array,
             label="mean error at timestep. Overall mean: {}".format(
                 np.around(np.mean(cumulative_errorlist), 2)))
    axes[0,1].plot(range(12, len(error_array) - 12),
             mean_error_over_time,
             label="Moving average in 25 hour window")
    axes[0,1].set_title=("combined components")
    axes[0,1].legend()
    axes[0,1].show()

if predict_naive_0:
    if MASS_PREDICT:
        statistical_pred.mass_predict(method="naive0",
                                      component="Remainder",
                                      iterations=ITERATIONS, step=STEP,
                                      axis=axes[2,1])
    else:
        statistical_pred.predict(method="naive0", component="Remainder",axis=axes[2,1])
        used_plots+=1
if predict_naive_lagged:
    if MASS_PREDICT:
        statistical_pred.mass_predict(method="naive_persistence",
                                      component="Price",
                                      iterations=ITERATIONS, step=STEP,axis=axes[2,1])
    else:
        statistical_pred.predict(method="naive_persistence",
                                 component="Price",axis=axes[2,0])
        used_plots+=1

plt.show()