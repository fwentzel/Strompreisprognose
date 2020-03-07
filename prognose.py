from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from residualPrediction import ResidualPrediction,LearningRateDecay
from seasonalPrediction import SeasonalPrediction
import ipykernel #fix for training progressbar in Pycharm
register_matplotlib_converters()
complete_data = get_data(start='2015-1-1', end='2020-02-17', update_data=False)  # end=date.today()
future_target = 24
past_history = 24  # input timesteps
start_index_from_max_length=future_target+170
predict_on_test_data=True

test_length=future_target+past_history+300 #300 Timesteps for testing.
test_data= complete_data.iloc[-test_length:]# Part of train_data the network wont see during Training and validation
train_data= complete_data.iloc[:-test_length]

if predict_on_test_data:
    data=test_data
else:
    data =train_data

#Complete
pred = ResidualPrediction(datacolumn="Price", train_data=train_data, test_data=test_data, future_target=future_target, past_history=past_history, start_index_from_max_length=start_index_from_max_length)
train=True
if train:
    pred.initialize_network(residualModel=False)
    pred.train_network(savename="trainedLSTM_complete2",lr_schedule="polynomal",power=2)#1 8.67
else:
    pred.load_model(savename="trainedLSTM_complete2")

#Residual
res_pred = ResidualPrediction(datacolumn="Residual", train_data=train_data, test_data=test_data, future_target=future_target, past_history=past_history, start_index_from_max_length=start_index_from_max_length)
train=False
if train:
    res_pred.initialize_network(residualModel=True)
    res_pred.train_network(savename="trainedLSTM_resid2",lr_schedule="polynomal",power=1)
else:
    res_pred.load_model(savename="trainedLSTM_resid2")


# Mass predict
complete_error=0
decomp_error=0
j = 0
iterations = 168
# truth=data["Price"].iloc[-start_index_from_max_length:-start_index_from_max_length+iterations+future_target]
# plt.plot(truth.index,truth,label="Truth.")
for i in range(0, iterations, 1):
    j+=1
    #Complete
    pred.predict(predict_test=predict_on_test_data, offset=i)
    complete_error += pred.error

    # Residual
    res_pred.predict(predict_test=predict_on_test_data,offset=i)

    # Seasonal
    seasonal_pred = SeasonalPrediction(data=data, forecast_length=future_target,
                                       start_index_from_max_length=start_index_from_max_length,offset=i)
    #seasonal_pred.AR_prediction()

    #Search Best Smoothing Level
    error=100
    best_smoothing_level=.2
    # for i in np.arange(0,1,.01):
    #     seasonal_pred.exponential_smoothing_prediction(smoothing_level=i)
    #     if seasonal_pred.error<error:
    #         error=seasonal_pred.error
    #         best_smoothing_level=i
    #print("best smoothing level:",best_smoothing_level,"with error",error)

    seasonal_pred.exponential_smoothing_prediction(smoothing_level=best_smoothing_level)

    #seasonal_pred.test_orders()
    #seasonal_pred.arima_prediction()

    # Trend
    predict_from=len(data)-start_index_from_max_length
    trend_truth = data["Trend"].iloc[predict_from:predict_from + future_target]

    # Combine predictions
    sum_pred = res_pred.pred + seasonal_pred.pred + trend_truth
    truth = data["Price"].iloc[predict_from:predict_from + future_target]
    decomp_error += np.around(np.sqrt(np.mean(np.square(truth - sum_pred))), 2)

print("decomposed: ",decomp_error/j, " normal: ",complete_error/j)



#Plot the predictions of components and their combination with the corresponding truth
# fig, ax = plt.subplots(4, 1, sharex=True)
# res_pred.plot_predictions(ax)
# seasonal_pred.plot_predictions(ax)
# ax[3].plot(trend_truth, label="Trend")
# ax[3].legend()
# ax[3].set_ylabel("TREND")
# ax[0].plot(truth.index,truth, label="Price truth")
# ax[0].plot(truth.index,sum_pred, label='decomposed prediction; RMSE: {}'.format(error))
# ax[0].plot(truth.index,pred.pred, label='price prediction; RMSE: {}'.format(pred.error))
# ax[0].legend()
# ax[0].set_ylabel("COMBINED")
#plt.show();