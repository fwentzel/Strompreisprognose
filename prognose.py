from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import csv
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from remainderPrediction import RemainderPrediction,LearningRateDecay
from seasonalPrediction import SeasonalPrediction
import ipykernel #fix for training progressbar in Pycharm
register_matplotlib_converters()

future_target = 24
past_history = 24  # input timesteps
predict_on_test_data=True
iterations = 168 #amount of predicitons for mass predict
step=10
epochs=200

test_length=future_target+past_history+500 #300 Timesteps for testing.
train_data,test_data=get_data(test_length=test_length, start='2015-1-1', end='2020-02-17', update_data=False)  # end=date.today()
data=test_data if predict_on_test_data else train_data
# if predict_on_test_data:
#     data=test_data
# else:
#     data =train_data


for past_history in range(168, 169, 12):
    pred = RemainderPrediction(datacolumn="Price", train_data=train_data, test_data=test_data, future_target=future_target, past_history=past_history,epochs=epochs)
    residual_prediction = RemainderPrediction(datacolumn="Remainder", train_data=train_data, test_data=test_data, future_target=future_target, past_history=past_history,epochs=epochs)
    Remainder=0
    for layers in range(0,6):
        with open('temp_results.csv', 'a', newline='') as fd:
            writer = csv.writer(fd)
            for dropout in np.arange(0,6):
                dropout_decimal=dropout/10
                train=True
                if train:
                    pred.initialize_network(dropout=dropout_decimal, additional_layers=layers)
                    pred.train_network(savename="trainedLSTM_complete")

                else:
                    pred.load_model(savename="trainedLSTM_complete")
                complete_error = pred.mass_predict(step=step, iterations=iterations,
                                                       predict_on_test_data=predict_on_test_data)
                #Remainder

                train=True
                if train:
                    residual_prediction.initialize_network(dropout=dropout_decimal, additional_layers=layers)
                    residual_prediction.train_network(savename="trainedLSTM_resid")
                else:
                    residual_prediction.load_model(savename="trainedLSTM_resid")

                #res_error=residual_prediction.mass_predict(iterations=iterations,predict_on_test_data=predict_on_test_data)

                #truth=data["Price"].iloc[:iterations+future_target]
                # plt.plot(truth.index,truth,label="Truth.")

                # Mass predict

                #decomp_error=0
                decomp_error=residual_prediction.mass_predict(step=step,iterations=iterations,predict_on_test_data=predict_on_test_data)
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


                writer.writerow([past_history,layers,dropout,decomp_error, complete_error])
                print(past_history,layers,dropout)

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