from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from csvReader import get_data
from residualPrediction import ResidualPrediction
from seasonalPrediction import SeasonalPrediction
register_matplotlib_converters()

complete_data = get_data(start='2015-1-1', end='2020-02-17', update_data=False)  # end=date.today()
future_target = 24
past_history = 96  # inputtimesteps
start_index_from_max_length=future_target + 1
use_test_data=False

test_length=future_target+past_history+48
test_data= complete_data.iloc[-test_length:]# Part of train_data the network wont see during Training and validation
train_data= complete_data.iloc[:-test_length]

if use_test_data:
    data=test_data
else:
    data = train_data
# Residual
res_pred = ResidualPrediction(train_data=train_data,test_data=test_data ,future_target=future_target, past_history=past_history, start_index_from_max_length=start_index_from_max_length)
res_pred.initialize_network(learning_rate=0.001)
res_pred.train_network(train=False, checkpoint="testing")
res_pred.predict(predict_test=use_test_data,num_predicitons=1, random_offset=False)

# Seasonal
seasonal_pred = SeasonalPrediction(data=data, forecast_length=future_target, train_length=future_target * 2,
                                   start_index_from_max_length=start_index_from_max_length)
seasonal_pred.fit_model()
seasonal_pred.predict()

# Trend
predict_from=len(data)-start_index_from_max_length
trend_truth = data["Trend"].iloc[predict_from:predict_from + future_target]

# Combine predictions
sum_pred = res_pred.prediciton_truth_error[0][0] + seasonal_pred.pred + trend_truth
truth = data["Price"].iloc[predict_from:predict_from + future_target]
error = np.around(np.sqrt(np.mean(np.square(truth - sum_pred))), 2)

#Plot the predictions of components and their combination with the corresponding truth
fig, ax = plt.subplots(4, 1, sharex=True)
res_pred.plot_predictions(ax)
seasonal_pred.plot_predictions(ax)
ax[3].plot(trend_truth, label="Trend")
ax[3].legend()
ax[3].set_ylabel("TREND")
ax[0].plot(truth, label="Price truth")
ax[0].plot(sum_pred, label='Prediction; RMSE: {}'.format(error))
ax[0].legend()
ax[0].set_ylabel("COMBINED")
plt.show();