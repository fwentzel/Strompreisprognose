from __future__ import absolute_import, division, print_function, unicode_literals
from csv_reader import get_data
from residualPrediction import ResidualPrediction
from seasonalPrediction import SeasonalPrediction
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import numpy as np
register_matplotlib_converters()

data = get_data(start='2015-1-1', end='2020-02-17', update_data=False)  # end=date.today()
future_target = 24
predict_from = len(data) - future_target
fig, ax = plt.subplots(4, 1, sharex=True)

# Residual
res_pred = ResidualPrediction(complete_data=data, future_target=future_target, start=predict_from)
res_pred.initialize_network(learning_rate=0.001)
res_pred.train_network(train=False, checkpoint="testing")
# res_pred.predict(num_predicitons=1,  start=res_pred.TRAIN_SPLIT,random_offset=True)
res_pred.predict(num_predicitons=1, random_offset=False)
res_pred.plot_predictions(ax)

# Seasonal
seasonal_pred = SeasonalPrediction(data=data, forecast_length=future_target, train_length=future_target * 3,
                                   start=predict_from)
seasonal_pred.fit_model()
seasonal_pred.predict()
seasonal_pred.plot_predictions(ax)

# Trend
trend_truth=data["Trend"].iloc[predict_from:predict_from + future_target]
ax[3].plot(trend_truth, label="Trend")
ax[3].legend()
ax[3].set_ylabel("TREND")

# Combine predictions
sum_pred = res_pred.prediciton_truth_error[0][0] + seasonal_pred.pred+trend_truth
truth=data["Price"].iloc[predict_from:predict_from + future_target]
error=np.around(np.sqrt(np.mean(np.square(truth - sum_pred))), 2)
ax[0].plot(truth, label="Price truth")
ax[0].plot(sum_pred, label='Prediction; RMSE: {}'.format(error))
ax[0].legend()
ax[0].set_ylabel("COMBINED")
plt.show()
