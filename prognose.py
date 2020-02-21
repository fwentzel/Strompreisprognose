from __future__ import absolute_import, division, print_function, unicode_literals
from csv_reader import get_data
import os
import matplotlib.pyplot as plt
from residualPrediction import ResidualPrediction
from seasonalPrediction import SeasonalPrediction

data = get_data(start= '2015-1-1', end='2020-02-17',update_data=False)#end=date.today()

# Residual
res_pred = ResidualPrediction(data)
res_pred.initialize_network(learning_rate=0.001)
res_pred.train_network(train=False, checkpoint="testing")
res_pred.multi_step_predict(num_predicitons=2,  starting_point=res_pred.TRAIN_SPLIT,random_offset=True)
res_pred.plot_predictions()
# print(res_pred.prediciton_truth_error)

# Seasonal
seasonal_pred=SeasonalPrediction(data=data,forecast_length=24,train_length=72,start=len(data)-100)
seasonal_pred.fit_model()
seasonal_pred.predict()
seasonal_pred.plot_predictions()

# print(seasonal_pred.prediciton_truth_error)
