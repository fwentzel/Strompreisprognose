from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from remainderPrediction import RemainderPrediction,LearningRateDecay
from seasonalPrediction import SeasonalPrediction
import ipykernel #fix for training progressbar in Pycharm
register_matplotlib_converters()

future_target = 24
past_history = 24  # input timesteps
start_index_from_max_length=future_target+170
predict_on_test_data=True
iterations = 168 #amount of predicitons for mass predict


complete_data = get_data(start='2015-1-1', end='2020-02-17', update_data=False)  # end=date.today()
test_length=future_target+past_history+300 #300 Timesteps for testing.
test_data= complete_data.iloc[-test_length:]# Part of train_data the network wont see during Training and validation
train_data= complete_data.iloc[:-test_length]
data=test_data if predict_on_test_data else train_data
# if predict_on_test_data:
#     data=test_data
# else:
#     data =train_data

pred = RemainderPrediction(datacolumn="Price", train_data=train_data, test_data=test_data, future_target=future_target, past_history=past_history, start_index_from_max_length=start_index_from_max_length)
residual_prediction = RemainderPrediction(datacolumn="Remainder", train_data=train_data, test_data=test_data, future_target=future_target, past_history=past_history, start_index_from_max_length=start_index_from_max_length)
Remainder=0

train=False
if train:
    pred.initialize_network(dropout=.5, additional_layers=0)
    pred.train_network(savename="trainedLSTM_complete")

else:
    pred.load_model(savename="trainedLSTM_complete")
#Remainder

train=False
if train:
    residual_prediction.initialize_network(dropout=0, additional_layers=0)
    residual_prediction.train_network(savename="trainedLSTM_resid")
else:
    residual_prediction.load_model(savename="trainedLSTM_resid")

#res_error=residual_prediction.mass_predict(iterations=iterations,predict_on_test_data=predict_on_test_data)

#truth=data["Price"].iloc[-start_index_from_max_length:-start_index_from_max_length+iterations+future_target]
# plt.plot(truth.index,truth,label="Truth.")

# Mass predict
complete_error=pred.mass_predict(iterations=iterations,predict_on_test_data=predict_on_test_data)
decomp_error=0

for i in range(iterations):
    # Remainder
    residual_prediction.predict(predict_test=predict_on_test_data, offset=i)

    # Seasonal
    seasonal_pred = SeasonalPrediction(data=data, forecast_length=future_target,
                                       start_index_from_max_length=start_index_from_max_length,offset=i)
    seasonal_pred.predict("exp")

    # Trend
    predict_from=len(data)-start_index_from_max_length+i
    trend_truth = data["Trend"].iloc[predict_from:predict_from + future_target]

    # Combine predictions
    sum_pred = residual_prediction.pred + seasonal_pred.pred + trend_truth

    #add error
    truth = data["Price"].iloc[predict_from:predict_from + future_target]
    decomp_error += np.around(np.sqrt(np.mean(np.square(truth - sum_pred))), 2)

decomp_error/=iterations
print("Remainder Error:",decomp_error,"; complete Error:",complete_error)


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