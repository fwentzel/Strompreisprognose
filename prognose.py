from __future__ import absolute_import, division, print_function, \
    unicode_literals
import csv
import os
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from neuralNetPrediction import NeuralNetPrediction
from statisticalPrediction import StatisticalPrediction
import ipykernel  # fix progress bar

register_matplotlib_converters()
future_target = 24
iterations = 168  # amount of predicitons for mass predict
step = 24
epochs = 100
train_complete = False
train_residual = False

# argument parsing for grid search
parser = ArgumentParser()
parser.add_argument("-lr",
                    default=7, type=int,
                    help=" learning rate index for learning_rate list")
parser.add_argument("-p", default=48, type=int,
                    help="amount of input timesteps")
parser.add_argument("-l", default=2, type=int,
                    help="additional layers for neural net")
parser.add_argument("-d", default=2, type=int,
                    help="dropout percentage in additional layer")
parser.add_argument("-cp",
                    default=True, type=bool,
                    help="predict complete part of time series")
parser.add_argument("-dp",
                    default=True, type=bool,
                    help=" predict decomposed part of time series")
parser.add_argument("-mp",
                    default=False, type=bool,
                    help="use mass prediction for error calculation")
parser.add_argument("-s",
                    default=0, type=int,
                    help="hour of the day, where the prediciton should start")

args = parser.parse_args()
past_history = args.p
layers = args.l
dropout = args.d
predict_complete = args.cp
predict_decomposed = args.dp
mass_predict_neural = args.mp
learning_rate = np.arange(0.0001, 0.001, 0.0001)[args.lr]
test_pred_start_hour=args.s
# learning_rate = .0004
plot_all = mass_predict_neural == False
test_length = future_target + iterations + 800  # Timesteps for testing.

train_data, test_data = get_data(test_length=test_length,
                                 test_pred_start_hour=test_pred_start_hour,
                                 past_history=past_history)
#test_data.Price.plot()
# plt.show()

dropout_decimal = dropout / 10
print("configuation: ", past_history, layers, dropout, learning_rate)

full_prediciton = None
if predict_complete:
    full_prediciton = NeuralNetPrediction(datacolumn="Price",
                                          train_data=train_data,
                                          test_data=test_data,
                                          future_target=future_target,
                                          past_history=past_history,
                                          epochs=epochs)

    if train_complete:
        full_prediciton.initialize_network(dropout=dropout_decimal,
                                           additional_layers=layers,
                                           learning_rate=learning_rate)
        full_prediciton.train_network(savename="trainedLSTM_complete",
                                      save=True,
                                      lr_schedule="polynomal",
                                      power=3)  # lr_schedule="polynomal" oder "step

    else:
        full_prediciton.load_model(savename="complete_best")

    if mass_predict_neural:
        full_prediciton.mass_predict(iterations=iterations,
                                     filename="complete",
                                     learning_rate=learning_rate,
                                     past_history=past_history,
                                     layers=layers,
                                     step=step)
    else:
        full_prediciton.predict(offset=0)

res_prediction = None
statistical_pred = None
i = 0
decomp_error = 0
sum_pred = 0
if predict_decomposed:
    # Residual
    res_prediction = NeuralNetPrediction(datacolumn="Remainder",
                                         train_data=train_data,
                                         test_data=test_data,
                                         future_target=future_target,
                                         past_history=past_history,
                                         epochs=epochs)

    if train_residual:
        res_prediction.initialize_network(dropout=dropout_decimal,
                                          additional_layers=layers,
                                          learning_rate=learning_rate)
        res_prediction.train_network(savename="trainedLSTM_resid",
                                     save=True,
                                     lr_schedule="polynomal",
                                     power=2)
        # lr_schedule="polynomal" oder "step
    else:
        res_prediction.load_model(savename="residual_best")

    if mass_predict_neural:
        res_prediction.mass_predict(iterations=iterations,
                                    filename="residual",
                                    learning_rate=learning_rate,
                                    past_history=past_history,
                                    layers=layers)
    else:
        # Remainder
        res_prediction.predict(offset=i)
        sum_pred = res_prediction.pred.copy()

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

        decomp_error += np.around(np.sqrt(np.mean(np.square(
            test_data["Price"].iloc[
            past_history + i: past_history + i + future_target] - sum_pred))),
            2)
        i += 1
    # with open('Results/residual_results.csv', 'a', newline='') as fd:
    #     writer = csv.writer(fd)
    #     writer.writerow([learning_rate, res_prediction.error,
    #                      res_prediction.single_errors.tolist()])

# decomp_error /= (i + 1)
if mass_predict_neural==False:
    fig, ax = plt.subplots(4, 1, sharex=True,figsize=(10.0, 10.0))#


    timeframe = slice(i - 1 + past_history,
                      past_history + future_target + i - 1)
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
    ax[0].plot(index, full_prediciton.pred,
               label='complete; mean RMSE of 168 predicitions: {}'.format(
                   full_prediciton.error))

    ax[1].plot(index, res_prediction.pred,
               label='Remainder prediciton '
                     '{}'.format(
                   res_prediction.error))
    ax[2].plot(index, statistical_pred.pred,
               label="prediction ; Error: {}".format(
                   statistical_pred.error))
    ax[3].plot(index,
               test_data["Trend"].iloc[timeframe])

    ax[0].set_ylabel("Komplett")
    ax[1].set_ylabel("Remainder")
    ax[2].set_ylabel("Seasonal")
    ax[3].set_ylabel("Trend")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    # Plot the predictions of components and their combination with the
    # corresponding truth
    #fig.suptitle("24-Stunden Prognose der einzelnen Zeireihenkomponenten und der kompletten Zeitreihe")
    #plt.savefig("Abbildungen/prediction_{}.png".format(test_pred_start_hour),dpi=300,bbox_inches='tight')
    #plt.show()
