from __future__ import absolute_import, division, print_function, unicode_literals
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

def neuronal_mass_predict(filename,neural_net_prediciton):
    global min_error, min_config, config, reader, row, fd, writer
    min_error = 100
    min_config = None
    with open("best_config_{}.csv".format(filename)) as config:
        reader = csv.reader(config, delimiter=',')
        for row in reader:
            min_config = row
            min_error = float(min_config[-1])
    res_error = neural_net_prediciton.mass_predict(iterations=iterations, predict_on_test_data=predict_on_test_data)
    with open('{}_results.csv'.format(filename), 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow([past_history, layers, dropout, res_error])
    if res_error < min_error:
        min_error = res_error
        min_config = [past_history, layers, dropout, min_error]
        complete_prediciton.model.save('.\checkpoints\{}_best'.format(filename))
        print(min_config)
        with open('best_config_{}.csv'.format(filename), 'w', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(min_config)



future_target = 24
past_history = 60  # input timesteps
predict_on_test_data = True
iterations = 168  # amount of predicitons for mass predict
step = 1
epochs = 1000

#argument parsing for grid search
parser = ArgumentParser()
parser.add_argument("-p",default=past_history)
parser.add_argument("-d", default=1)
parser.add_argument("-l", default=1)
parser.add_argument("-cp", default=True)
parser.add_argument("-dp", default=True)
parser.add_argument("-mp",default=False)
args = parser.parse_args()
print("args ",args.p,args.l,args.d)
past_history = int(args.p)
layers = int(args.l)
dropout = int(args.d)
predict_complete=bool(args.cp)
predict_decomposed=bool(args.dp)
mass_Predict=bool(args.mp)

test_length = future_target + past_history + 500  # 300 Timesteps for testing.
train_data, test_data = get_data(test_length=test_length, start='2015-1-1', end='2020-02-17',
                                 update_data=False)  # end=date.today()
data = test_data if predict_on_test_data else train_data


dropout_decimal = dropout / 10

#complete timeseries
if predict_complete:
    complete_prediciton = NeuralNetPrediction(datacolumn="Price", train_data=train_data, test_data=test_data, future_target=future_target,
                                              past_history=past_history, epochs=epochs)

    train = True
    if train:
        complete_prediciton.initialize_network(dropout=dropout_decimal, additional_layers=layers)
        complete_prediciton.train_network(savename="trainedLSTM_complete", save=False)

    else:
        complete_prediciton.load_model(savename="trainedLSTM_complete")

    if mass_Predict:
        neuronal_mass_predict("compelete", complete_prediciton)

if predict_decomposed:
    # Residual
    residual_prediction = NeuralNetPrediction(datacolumn="Remainder", train_data=train_data, test_data=test_data,
                                              future_target=future_target, past_history=past_history, epochs=epochs)

    train=True
    if train:
        residual_prediction.initialize_network(dropout=dropout_decimal, additional_layers=layers)
        residual_prediction.train_network(savename="trainedLSTM_resid")
    else:
        residual_prediction.load_model(savename="trainedLSTM_resid")

    if mass_Predict:
        neuronal_mass_predict("residual",residual_prediction)

    decomp_error = 0
    decomp_error=residual_prediction.mass_predict(step=step,iterations=iterations,predict_on_test_data=predict_on_test_data)
    for i in range(0,iterations,step):
        # Remainder
        residual_prediction.predict(predict_test=predict_on_test_data, offset=i)

        # Seasonal
        seasonal_pred = StatisticalPrediction(data=data, forecast_length=future_target,
                                           offset=i)
        seasonal_pred.predict("exp")

        # Trend
        predict_from=len(data)+i
        trend_truth = data["Trend"].iloc[predict_from:predict_from + future_target]

        # Combine predictions
        sum_pred = residual_prediction.pred + seasonal_pred.pred + trend_truth

        #add error
        truth = data["Price"].iloc[predict_from:predict_from + future_target]
        decomp_error += np.around(np.sqrt(np.mean(np.square(truth - sum_pred))), 2)
    decomp_error/=(iterations/step)

plot_all=False
if predict_decomposed and predict_complete and plot_all:
    #Plot the predictions of components and their combination with the corresponding truth
    fig, ax = plt.subplots(4, 1, sharex=True)
    residual_prediction.plot_predictions(ax)
    seasonal_pred.plot_predictions(ax)
    ax[3].plot(trend_truth, label="Trend")
    ax[3].legend()
    ax[3].set_ylabel("TREND")
    ax[0].plot(truth.index,truth, label="Price truth")
    ax[0].plot(truth.index,sum_pred, label='decomposed; mean RMSE of 168 predicitions: {}'.format(decomp_error))
    ax[0].plot(truth.index,complete_prediciton.complete_prediciton, label='complete; mean RMSE of 168 predicitions: {}'.format(complete_error))
    ax[0].legend()
    ax[0].set_ylabel("COMBINED")
    plt.show();
