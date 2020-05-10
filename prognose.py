from __future__ import absolute_import, division, print_function, \
    unicode_literals
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from neuralNetPrediction import NeuralNetPrediction
from statisticalPrediction import StatisticalPrediction
import GUI
import ipykernel  # fix progress bar

register_matplotlib_converters()

FUTURE_TARGET = 24
ITERATIONS = 24 * 2 # amount of predicitons for mass predict
STEP = 1

use_setup_settings = True
settings_dict = None
config = GUI.main()
if config is None:
    # User cancelled program
    import sys
    sys.exit()
else:
    settings_dict = config

figure, axes = plt.subplots(nrows=3, ncols=2)

train_complete = settings_dict["train_complete"]
train_remainder = settings_dict["train_remainder"]
train_day_of_week = settings_dict["train_day_of_week"]

predict_complete = settings_dict["predict_complete"]
predict_remainder = settings_dict["predict_remainder"]
predict_decomposed = settings_dict["predict_decomposed"]
predict_naive_lagged = settings_dict["predict_naive_lagged"]
predict_naive_0 = settings_dict["predict_naive_0"]
predict_sarima = settings_dict["predict_sarima"]

predict_with_day = settings_dict["predict_with_day"]

test_pred_start_hour = settings_dict["test_pred_start_hour"]

if test_pred_start_hour < 0:
    mass_predict = True
    test_pred_start_hour = 0
    print("Calculating forecast for up to 168 hours after training in a massforecast".format(
        test_pred_start_hour))
else:
    print("Calculating forecast for: {} hours after Trainings".format(test_pred_start_hour))
    mass_predict = False

test_length = FUTURE_TARGET + 24 * 7  # Timesteps for testing.
data = get_data()
test_split_at_hour = data.index[
                         -test_length].hour - test_pred_start_hour + test_length


def price_pred():
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


    if mass_predict:
        price_prediciton.mass_predict(iterations=ITERATIONS,
                                      step=STEP,
                                      use_day_model=predict_with_day,
                                      axis=axes[0, 0])

    else:
        print("Calculating price forecast...")
        price_prediciton.predict(offset=0,
                                 use_day_model=predict_with_day,
                                 axis=axes[0, 0])


def sarima_pred():
    if mass_predict:
        # statistical_pred.predict(method="ARIMA", component="Price",
        #                          use_auto_arima=True, offset=0)
        statistical_pred.mass_predict(iterations=ITERATIONS,
                                      step=STEP,
                                      component="Price",
                                      use_auto_arima=False,
                                      method="sarima",
                                      axis=axes[1, 1])
    else:
        print("Calculating SARIMA forecast...")
        i = 0
        statistical_pred.predict(method="sarima", component="Price",
                                 use_auto_arima=False, offset=i,
                                 axis=axes[1, 1])

        timeframe = slice(i - test_split_at_hour,
                          FUTURE_TARGET + i - test_split_at_hour)
        truth = data["Price"].iloc[timeframe]


def predict_decomposed_or_remainder():
    rem_prediction = NeuralNetPrediction(datacolumn="Remainder",
                                         data=data,
                                         future_target=FUTURE_TARGET,
                                         test_split_at_hour=test_split_at_hour,
                                         net_type="remainder_complete",
                                         train_day_of_week=train_day_of_week)
    if train_remainder:
        rem_prediction.initialize_network()
        rem_prediction.train_network(
            savename="trainedLSTM_remainderUTC", save=False,
            lr_schedule="polynomal",power=2)
    elif rem_prediction.model is None:
        rem_prediction.load_model(savename="trainedLSTM_remainderUTC")

    if predict_remainder:
        # Remainder
        if mass_predict:
            rem_prediction.mass_predict(iterations=ITERATIONS,
                                        step=STEP,
                                        use_day_model=predict_with_day,
                                        axis=axes[1, 0])
        else:
            # Remainder
            rem_prediction.predict(offset=0,
                                   use_day_model=predict_with_day,
                                   axis=axes[
                                       1, 0] if not predict_decomposed else None)

    if predict_decomposed:
        components_combined_error = 0
        sum_pred = 0
        j = 0
        single_errorlist = np.empty(
            [round(ITERATIONS / STEP), FUTURE_TARGET])
        error_array = np.empty((ITERATIONS + FUTURE_TARGET, 1))
        error_array[:] = np.nan
        decomposed_iterations = ITERATIONS if mass_predict and predict_decomposed else 1
        max = round(decomposed_iterations / STEP)
        offsets = range(0, decomposed_iterations, STEP)
        for i in offsets:
            if decomposed_iterations > 1:
                print("\rmass predict decomposed: {}/{}".format(j, max),
                      sep=' ', end='', flush=True)
            else:
                print("Calculating component forecast...")


            timeframe = slice(i - test_split_at_hour,
                              FUTURE_TARGET + i - test_split_at_hour)
            truth = data["Price"].iloc[timeframe]

            rem_prediction.predict(offset=i,
                                   use_day_model=predict_with_day,
                                   axis=axes[
                                       1, 0] if not predict_decomposed else None)
            # copy so original doesnt get overwritten when adding other components

            sum_pred = rem_prediction.pred.copy()
            # plt.plot(rem_prediction.truth.index, sum_pred,
            #          label="pred {}".format(rem_prediction.error))
            # Seasonal
            statistical_pred.predict(method="AutoReg",
                                     component="Seasonal",
                                     offset=i)
            sum_pred += statistical_pred.pred
            # plt.plot(truth.index, statistical_pred.pred,
            #          label="pred {}".format(statistical_pred.error))
            # Trend
            statistical_pred.predict(method="AutoReg",
                                     component="Trend",
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
            if mass_predict:
                single_errorlist[j] = single_errors
                arr = np.nanmean([error_array[i:i + FUTURE_TARGET],
                                  single_errorlist[j].reshape(
                                      FUTURE_TARGET, 1)], axis=0)
                error_array[i:i + FUTURE_TARGET] = arr
                # plt.plot(truth.index, sum_pred,
                #          label="pred {}".format(
                #              components_combined_error))
                # plt.plot(truth.index, truth,
                #          label="truth")
                # plt.legend()
                # plt.savefig(
                #     "./Abbildungen/decomposed/prediction_{}.png".format(
                #         i),
                #     dpi=300, bbox_inches='tight')
                # plt.clf()
            else:

                axes[0, 1].plot(truth.index, sum_pred,
                                label='prediction; RMSE: {}'.format(
                                    components_combined_error))
                axes[0, 1].plot(truth.index, truth, label='Truth')
                axes[0, 1].set_title("components combined")
                axes[0, 1].legend()
            j += 1

        if mass_predict :
            cumulative_errorlist = np.around(np.mean(single_errorlist, axis=0),
                decimals=2)

            mean_error_over_time = [np.mean(error_array[x - 12:x + 12])
                                    for x in range(12, len(error_array) - 12)]
            x_ticks=data.index[test_split_at_hour:test_split_at_hour+ITERATIONS+FUTURE_TARGET]
            axes[0, 1].plot(x_ticks,error_array,
                            label="mean error at timestep. Overall mean: {}".format(
                                np.around(np.mean(cumulative_errorlist),2)))
            axes[0, 1].plot(x_ticks[12:-12],mean_error_over_time,
                            label="Moving average in 25 hour window")
            axes[0, 1].set_title ("combined components")
            axes[0, 1].legend()

def naive_lagged_pred():
    if mass_predict:
        statistical_pred.mass_predict(method="naive_persistence",
                                      component="Price",
                                      iterations=ITERATIONS, step=STEP,
                                      axis=axes[2, 0])
    else:
        print("Calculating naive persistance forecast...")
        statistical_pred.predict(method="naive_persistence",
                                 component="Price", axis=axes[2, 0])

def naive_0_pred():
    if mass_predict:
        statistical_pred.mass_predict(method="naive0",
                                      component="Remainder",
                                      iterations=ITERATIONS, step=STEP,
                                      axis=axes[2, 1])
    else:
        print("Calculating naive0 forecast...")
        statistical_pred.predict(method="naive0", component="Remainder",
                                 axis=axes[2, 1])


statistical_pred = StatisticalPrediction(data=data,
                                         future_target=FUTURE_TARGET,
                                         test_split_at_hour=test_split_at_hour,
                                         )

if predict_complete:
    price_pred()
if predict_sarima:
    sarima_pred()
if predict_decomposed or predict_remainder:
    predict_decomposed_or_remainder()
if predict_naive_0:
    naive_0_pred()
if predict_naive_lagged:
    naive_lagged_pred()
plt.show()
