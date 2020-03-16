from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import csv
from argparse import ArgumentParser
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from neuralNetPrediction import NeuralNetPrediction,LearningRateDecay
from statisticalPrediction import StatisticalPrediction
import ipykernel #fix for training progressbar in Pycharm
register_matplotlib_converters()


future_target = 24
past_history = 60  # input timesteps
predict_on_test_data=True
iterations = 168 #amount of predicitons for mass predict
step=1
epochs=1000

parser = ArgumentParser()
parser.add_argument("-p")
parser.add_argument("-d")
parser.add_argument("-l")
args = parser.parse_args()

past_history=args.p
layers=args.l
dropout=args.d

test_length=future_target+past_history+500 #300 Timesteps for testing.
train_data,test_data=get_data(test_length=test_length, start='2015-1-1', end='2020-02-17', update_data=False)  # end=date.today()
data=test_data if predict_on_test_data else train_data
# if predict_on_test_data:
#     data=test_data
# else:
#     data =train_data
row=None
with open("complete_results.csv") as tr:
    reader=csv.reader(tr,delimiter=',')
    for newrow in reader:
        if len(newrow)>2:
            row=newrow
        pass

# pred = RemainderPrediction(datacolumn="Price", train_data=train_data, test_data=test_data, future_target=future_target, past_history=past_history,epochs=epochs)
# pred.load_model(savename="trainedLSTM_complete")
# complete_error = pred.mass_predict(step=step, iterations=iterations,
#                                                    predict_on_test_data=predict_on_test_data)
# print("complete",complete_error)

min_error=100
min_config=None
with open("best_config.csv") as config:
    reader=csv.reader(config,delimiter=',')
    for row in reader:
        min_config=row
        min_error=float(min_config[-1])

catching_up=row is not None
for past_history in range(12, 169, 12):
    if catching_up:
        if int(row[0])>past_history:
            continue
    pred = NeuralNetPrediction(datacolumn="Price", train_data=train_data, test_data=test_data, future_target=future_target, past_history=past_history, epochs=epochs)
    residual_prediction = NeuralNetPrediction(datacolumn="Remainder", train_data=train_data, test_data=test_data, future_target=future_target, past_history=past_history, epochs=epochs)
    for layers in range(0,6):

        if catching_up:
            if int(row[1]) > layers:
                continue

        for dropout in np.arange(0,6):
            if catching_up:
                if int(row[2])>dropout:
                    continue
                else:
                    catching_up=False
                    continue
            dropout_decimal=dropout/10

            print(past_history,layers,dropout)

            train=True
            if train:
                pred.initialize_network(dropout=dropout_decimal, additional_layers=layers)
                pred.train_network(savename="trainedLSTM_complete",save=False)

            else:
                pred.load_model(savename="trainedLSTM_complete")
            complete_error = pred.mass_predict(step=step, iterations=iterations,
                                                   predict_on_test_data=predict_on_test_data)
            #Remainder

            # train=True
            # if train:
            #     residual_prediction.initialize_network(dropout=dropout_decimal, additional_layers=layers)
            #     residual_prediction.train_network(savename="trainedLSTM_resid")
            # else:
            #     residual_prediction.load_model(savename="trainedLSTM_resid")

            #res_error=residual_prediction.mass_predict(iterations=iterations,predict_on_test_data=predict_on_test_data)

            #truth=data["Price"].iloc[:iterations+future_target]
            # plt.plot(truth.index,truth,label="Truth.")

            # Mass predict

            decomp_error=0
            # decomp_error=residual_prediction.mass_predict(step=step,iterations=iterations,predict_on_test_data=predict_on_test_data)
            # for i in range(0,iterations,step):
            #     # Remainder
            #     residual_prediction.predict(predict_test=predict_on_test_data, offset=i)
            #
            #     # Seasonal
            #     seasonal_pred = SeasonalPrediction(data=data, forecast_length=future_target,
            #                                        offset=i)
            #     seasonal_pred.predict("exp")
            #
            #     # Trend
            #     predict_from=len(data)+i
            #     trend_truth = data["Trend"].iloc[predict_from:predict_from + future_target]
            #
            #     # Combine predictions
            #     sum_pred = residual_prediction.pred + seasonal_pred.pred + trend_truth
            #
            #     #add error
            #     truth = data["Price"].iloc[predict_from:predict_from + future_target]
            #     decomp_error += np.around(np.sqrt(np.mean(np.square(truth - sum_pred))), 2)
            #
            # decomp_error/=(iterations/step)
            with open('complete_results.csv', 'a', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow([past_history,layers,dropout,complete_error])
            if complete_error<min_error:
                min_error=complete_error
                min_config=[past_history,layers,dropout,min_error]
                pred.model.save('.\checkpoints\complete_best')
                print(min_config)
                with open('best_config.csv', 'w', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(min_config)


#Plot the predictions of components and their combination with the corresponding truth
#fig, ax = plt.subplots(4, 1, sharex=True)
# residual_prediction.plot_predictions(ax)
# seasonal_pred.plot_predictions(ax)
# ax[3].plot(trend_truth, label="Trend")
# ax[3].legend()
# ax[3].set_ylabel("TREND")
# ax[0].plot(truth.index,truth, label="Price truth")
# ax[0].plot(truth.index,sum_pred, label='decomposed; mean RMSE of 168 predicitions: {}'.format(decomp_error))
# ax[0].plot(truth.index,pred.pred, label='complete; mean RMSE of 168 predicitions: {}'.format(complete_error))
# ax[0].legend()
# ax[0].set_ylabel("COMBINED")
# plt.show();
