from __future__ import absolute_import, division, print_function, \
    unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import csv
from argparse import ArgumentParser
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from neuralNetPrediction import NeuralNetPrediction, LearningRateDecay
from statisticalPrediction import StatisticalPrediction
import ipykernel  # fix for training progressbar in Pycharm

register_matplotlib_converters()

future_target = 24
iterations = 168  # amount of predicitons for mass predict
step = 1
epochs = 500
learning_rate_list = np.arange(0.0001, 0.001, 0.0001)
# argument parsing for grid search
parser = ArgumentParser()
parser.add_argument("-lr",
                    default=7)  # learning rate index for learning_rate_list
parser.add_argument("-p", default=120)  # input timesteps
parser.add_argument("-d", default=0)  # dropout percentage
parser.add_argument("-l", default=0)  # additional layers
parser.add_argument("-cp",
                    default=True)  # predict complete part of time series
parser.add_argument("-dp",
                    default=False)  # predict decomposed part of time series
parser.add_argument("-mp",
                    default=True)  # use mass prediction for error calculation

args = parser.parse_args()

past_history = int(args.p)
layers = int(args.l)
dropout = int(args.d)
predict_complete = bool(args.cp)
predict_decomposed = bool(args.dp)
mass_predict_neural = bool(args.mp)
learning_rate = learning_rate_list[int(args.lr)]
# learning_rate = .0004
plot_all = mass_predict_neural == False
plot_all = True
test_length = future_target + past_history + 400  # 400 Timesteps for testing.

train_data, test_data = get_data(test_length=test_length,
                                 update_price_data=False,
                                 update_weather_data=False)  #

dropout_decimal = dropout / 10

print("config: ", past_history, layers, dropout, learning_rate)
# complete timeseries
complete_prediciton = None
if predict_complete:
    complete_prediciton = NeuralNetPrediction(datacolumn="Price",
                                              train_data=train_data,
                                              test_data=test_data,
                                              future_target=future_target,
                                              past_history=past_history,
                                              epochs=epochs)

    train = False
    if train:
        complete_prediciton.initialize_network(dropout=dropout_decimal,
                                               additional_layers=layers,
                                               learning_rate=learning_rate)
        complete_prediciton.train_network(
            savename="trainedLSTM_complete",
            save=mass_predict_neural == False, lr_schedule="polynomal",
            power=1)  # lr_schedule="polynomal" oder "step

    else:
        complete_prediciton.load_model(savename="trainedLSTM_complete")

    if mass_predict_neural:
        complete_prediciton.mass_predict(iterations=iterations,
                                         filename="complete",
                                         learning_rate=learning_rate,
                                         past_history=past_history,
                                         layers=layers)
    else:
        complete_prediciton.predict(offset=0)
    with open('Results/complete_results.csv', 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(
            [past_history, learning_rate, complete_prediciton.error,
             complete_prediciton.single_errors.tolist()])
        sum_pred = complete_prediciton.pred.copy()

residual_prediction = None
statistical_pred = None
i = 0
decomp_error = 0
sum_pred = 0
if predict_decomposed:
    # Residual
    residual_prediction = NeuralNetPrediction(datacolumn="Remainder",
                                              train_data=train_data,
                                              test_data=test_data,
                                              future_target=future_target,
                                              past_history=past_history,
                                              epochs=epochs)

    train = False
    if train:
        residual_prediction.initialize_network(dropout=dropout_decimal,
                                               additional_layers=layers,
                                               learning_rate=learning_rate)
        residual_prediction.train_network(savename="trainedLSTM_resid",
                                          save=mass_predict_neural == False,
                                          lr_schedule="polynomal",
                                          power=2)
        # lr_schedule="polynomal" oder "step
    else:
        residual_prediction.load_model(savename="trainedLSTM_resid")

    if mass_predict_neural:
        residual_prediction.mass_predict(iterations=iterations,
                                         filename="residual",
                                         learning_rate=learning_rate,
                                         past_history=past_history,
                                         layers=layers)
    else:
        # Remainder
        residual_prediction.predict(offset=i)
        sum_pred = residual_prediction.pred.copy()

        # Seasonal
        statistical_pred = StatisticalPrediction(data=test_data,
                                                 forecast_length=future_target,
                                                 offset=i,
                                                 neural_past_history=past_history)
        statistical_pred.predict("exp", "Seasonal")
        sum_pred += statistical_pred.pred

        # Trend
        trend_pred = test_data["Trend"].iloc[
                     past_history + i: past_history + i + future_target]
        sum_pred += trend_pred

        # add error

        decomp_error += np.around(np.sqrt(
            np.mean(np.square(test_data["Price"].iloc[
                              past_history + i: past_history + i + future_target] - sum_pred))),
            2)
        i += 1
    with open('Results/residual_results.csv', 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow([learning_rate, residual_prediction.error,
                         residual_prediction.single_errors.tolist()])

# decomp_error /= (i + 1)
plot_all = False
if plot_all:
    fig, ax = plt.subplots(4, 1, sharex=True)
    timeframe = slice(i + past_history,
                      past_history + future_target + i)
    index = test_data.iloc[timeframe].index
    ax[0].plot(index, test_data["Price"].iloc[timeframe],
               label="Truth")
    ax[1].plot(index, test_data["Remainder"].iloc[timeframe],
               label="Truth")
    ax[2].plot(index, test_data["Seasonal"].iloc[timeframe],
               label='truth')

    ax[0].plot(index, sum_pred,
               label='decomposed; mean RMSE of 168 predictions: '
                     '{}'.format(
                   decomp_error))
    ax[0].plot(index, complete_prediciton.pred,
               label='complete; mean RMSE of 168 predicitions: {}'.format(
                   complete_prediciton.error))

    ax[1].plot(index, residual_prediction.pred,
               label='Remainder prediciton '
                     '{}'.format(
                   residual_prediction.error))
    ax[2].plot(index, statistical_pred.pred,
               label="prediction ; Error: {}".format(
                   statistical_pred.error))
    ax[3].plot(index,
               test_data["Trend"].iloc[timeframe])

    ax[2].set_ylabel("Seasonal")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    # Plot the predictions of components and their combination with the
    # corresponding truth
    plt.show()
