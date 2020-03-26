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


def neuronal_mass_predict(filename, neural_net_prediciton):
    min_error = 100
    min_config = None
    with open("best_config_{}.csv".format(filename)) as config:
        reader = csv.reader(config, delimiter=',')
        for row in reader:
            min_config = row
        min_error = float(min_config[-1])
    error, single_errors = neural_net_prediciton.mass_predict(iterations=iterations,
                                                              predict_on_test_data=predict_on_test_data)
    print("ERRORS: ", error, single_errors)
    with open('{}_results.csv'.format(filename), 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow([learning_rate, past_history, layers, dropout, error, single_errors.tolist()])
    if error < min_error:
        min_error = error
        min_config = [learning_rate, past_history, layers, dropout, min_error]
        neural_net_prediciton.model.save(
            '.\checkpoints\{}_best'.format(filename))
        print(min_config)
        with open('best_config_{}.csv'.format(filename), 'w', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(min_config)
    return error


future_target = 24
predict_on_test_data = True
iterations = 168  # amount of predicitons for mass predict
step = 1
epochs = 500
learning_rate_list = np.arange(0.0001, 0.001, 0.0001)
# argument parsing for grid search
parser = ArgumentParser()
parser.add_argument("-lr", default=1)  # learning rate index for learning_rate_list
parser.add_argument("-p", default=24)  # input timesteps
parser.add_argument("-d", default=0)  # dropout percentage
parser.add_argument("-l", default=0)  # additional layers
parser.add_argument("-cp", default=True)  # predict complete part of time series
parser.add_argument("-dp", default=True)  # predict decomposed part of time series
parser.add_argument("-mp", default=False)  # use mass prediction for error calculation

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
plot_all = False
test_length = future_target + past_history + 400  # 400 Timesteps for testing.

train_data, test_data = get_data(test_length=test_length, update_price_data=False, update_weather_data=False)  #
data = test_data if predict_on_test_data else train_data

dropout_decimal = dropout / 10

fig, ax = plt.subplots(4, 1, sharex=True)
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

    train = True
    if train:
        complete_prediciton.initialize_network(dropout=dropout_decimal,
                                               additional_layers=layers, learning_rate=learning_rate)
        complete_prediciton.train_network(savename="trainedLSTM_complete",
                                          save=mass_predict_neural == False, lr_schedule="polynomal",
                                          power=1)  # lr_schedule="polynomal" oder "step

    else:
        complete_prediciton.load_model(savename="trainedLSTM_complete")

    if mass_predict_neural:
        neuronal_mass_predict("complete", complete_prediciton)
    else:
        complete_prediciton.predict(predict_test=predict_on_test_data, offset=0)
    with open('complete_results.csv', 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(
            [past_history, learning_rate, complete_prediciton.error, complete_prediciton.single_errors.tolist()])
        sum_pred = complete_prediciton.pred.copy()
residual_prediction = None
statistical_pred = None
i = 0
decomp_error = 0
sum_pred = None
if predict_decomposed:
    # Residual
    residual_prediction = NeuralNetPrediction(datacolumn="Remainder",
                                              train_data=train_data,
                                              test_data=test_data,
                                              future_target=future_target,
                                              past_history=past_history,
                                              epochs=epochs)

    train = True
    if train:
        residual_prediction.initialize_network(dropout=dropout_decimal,
                                               additional_layers=layers, learning_rate=learning_rate)
        residual_prediction.train_network(savename="trainedLSTM_resid",
                                          save=mass_predict_neural == False, lr_schedule="polynomal",
                                          power=2)  # lr_schedule="polynomal" oder "step
    else:
        residual_prediction.load_model(savename="trainedLSTM_resid")

    if mass_predict_neural:
        neuronal_mass_predict("residual", residual_prediction)
    else:
        # Remainder
        residual_prediction.predict(predict_test=predict_on_test_data,
                                    offset=i)
        sum_pred = residual_prediction.pred.copy()

        # Seasonal
        statistical_pred = StatisticalPrediction(data=data,
                                                 forecast_length=future_target,
                                                 offset=i, neural_past_history=past_history)
        statistical_pred.predict("exp", "Seasonal")
        sum_pred += statistical_pred.pred

        # Trend
        trend_pred = data["Trend"].iloc[past_history + i: past_history + i + future_target]
        sum_pred += trend_pred

        # add error

        decomp_error += np.around(np.sqrt(
            np.mean(np.square(data["Price"].iloc[past_history + i: past_history + i + future_target] - sum_pred))), 2)
        i += 1
    with open('residual_results.csv', 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow([learning_rate, residual_prediction.error, residual_prediction.single_errors.tolist()])

decomp_error /= (i + 1)
if mass_predict_neural == False and complete_prediciton is not None and residual_prediction is not None and statistical_pred is not None and plot_all == True:
    timeframe = slice(i + past_history, past_history + future_target + i)
    index = data.iloc[timeframe].index
    ax[0].plot(index, data["Price"].iloc[timeframe],
               label="Truth")
    ax[1].plot(index, data["Remainder"].iloc[timeframe], label="Truth")
    ax[2].plot(index, data["Seasonal"].iloc[timeframe], label='truth')

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
               label="prediction ; Error: {}".format(statistical_pred.error))
    ax[3].plot(index,
               data["Trend"].iloc[timeframe])

    ax[2].set_ylabel("Seasonal")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    # Plot the predictions of components and their combination with the
    # corresponding truth
    plt.show()
