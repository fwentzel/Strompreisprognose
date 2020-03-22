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
    error,single_errors = neural_net_prediciton.mass_predict(iterations=iterations,
                                               predict_on_test_data=predict_on_test_data)
    with open('{}_results.csv'.format(filename), 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow([past_history, layers, dropout, error,single_errors.tolist()])
    if error < min_error:
        min_error = error
        min_config = [past_history, layers, dropout, min_error]
        neural_net_prediciton.model.save(
            '.\checkpoints\{}_best'.format(filename))
        print(min_config)
        with open('best_config_{}.csv'.format(filename), 'w', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(min_config)
    return error


future_target = 24
past_history = 12  # input timesteps
predict_on_test_data = True
iterations = 168  # amount of predicitons for mass predict
step = 1
epochs = 1000

# argument parsing for grid search
parser = ArgumentParser()
parser.add_argument("-p", default=past_history)
parser.add_argument("-d", default=0)
parser.add_argument("-l", default=0)
parser.add_argument("-cp", default=True)
parser.add_argument("-dp", default=False)
parser.add_argument("-mp", default=True)
parser.add_argument("-plt", default=False)
args = parser.parse_args()
print("args ", args.p, args.l, args.d)
past_history = int(args.p)
layers = int(args.l)
dropout = int(args.d)
predict_complete = bool(args.cp)
predict_decomposed = bool(args.dp)
mass_predict_neural = bool(args.mp)
plot_all = bool(args.plt)

test_length = future_target + past_history + 500  # 300 Timesteps for testing.
train_data, test_data = get_data(test_length=test_length, update_price_data=False, update_weather_data=False)  #
data = test_data if predict_on_test_data else train_data

dropout_decimal = dropout / 10

fig, ax = plt.subplots(4, 1, sharex=True)

# complete timeseries
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
                                               additional_layers=layers)
        complete_prediciton.train_network(savename="trainedLSTM_complete",
                                          save=False,power=2)

    else:
        complete_prediciton.load_model(savename="trainedLSTM_complete")

    if mass_predict_neural:
        neuronal_mass_predict("complete", complete_prediciton)
    else:
        complete_prediciton.predict(
            predict_test=predict_on_test_data, offset=0)
    if plot_all:
        ax[0].plot(complete_prediciton.truth.index, complete_prediciton.pred,
                   label='complete; mean RMSE of 168 predicitions: {}'.format(
                       complete_prediciton.error))

predict_decomposed=False
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
                                               additional_layers=layers)
        residual_prediction.train_network(savename="trainedLSTM_resid")
    else:
        residual_prediction.load_model(savename="trainedLSTM_resid")

    if mass_predict_neural:
        neuronal_mass_predict("residual", residual_prediction)
    else:
        decomp_error = 0
        for i in range(0, iterations, step):
            # Remainder
            residual_prediction.predict(predict_test=predict_on_test_data,
                                        offset=i)

            # Seasonal
            statistical_pred = StatisticalPrediction(data=data,
                                                     forecast_length=future_target,
                                                     offset=i)
            statistical_pred.predict("exp")

            # Trend
            predict_from = len(data) + i
            trend_truth = data["Trend"].iloc[
                          predict_from:predict_from + future_target]

            # Combine predictions
            sum_pred = residual_prediction.pred + statistical_pred.pred + \
                       trend_truth

            # add error
            truth = data["Price"].iloc[
                    predict_from:predict_from + future_target]
            decomp_error += np.around(
                np.sqrt(np.mean(np.square(truth - sum_pred))),
                2)
            if plot_all:
                statistical_pred.plot_predictions(ax)
                ax[0].plot(truth.index, sum_pred,
                           label='decomposed; mean RMSE of 168 predictions: '
                                 '{}'.format(
                               decomp_error))
            if plot_all:
                ax[3].plot(trend_truth, label="Trend")
                ax[0].plot(truth.index, truth, label="Price truth")
        decomp_error /= (iterations / step)
        if plot_all:
            residual_prediction.plot_predictions(ax)

if plot_all:
    # Plot the predictions of components and their combination with the
    # corresponding truth

    ax[3].legend()
    ax[3].set_ylabel("TREND")
    ax[0].legend()
    ax[0].set_ylabel("COMBINED")
    plt.show()
