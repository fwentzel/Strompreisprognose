from __future__ import absolute_import, division, print_function, \
    unicode_literals

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from neuralNetPrediction import NeuralNetPrediction
from statisticalPrediction import StatisticalPrediction
import ipykernel  # fix progress bar

register_matplotlib_converters()
future_target = 24

mass_predict_neural = False
iterations = 168  # amount of predicitons for mass predict
step = 1

train_complete = False
train_residual = False
train_day_of_week = False

test_pred_start_hour = 0

predict_complete = True
predict_remainder = True
predict_decomposed = True
predict_naive=True

predict_with_day = True


test_length = future_target + iterations + 172 # Timesteps for testing.
data = get_data()
test_split_at_hour = data.index[
                         -test_length].hour - test_pred_start_hour + test_length

full_prediciton = None
if predict_complete:
    full_prediciton = NeuralNetPrediction(datacolumn="Price",
                                          data=data,
                                          future_target=future_target,
                                          test_split_at_hour=test_split_at_hour,
                                          net_type="complete",
                                          train_day_of_week=train_day_of_week)

    if train_complete:
        full_prediciton.initialize_network()
        full_prediciton.train_network(savename="trainedLSTM_complete",
                                      save=False,
                                      lr_schedule="polynomal",
                                      power=2)  # lr_schedule="polynomal" oder "step

    else:
        full_prediciton.load_model(savename="trainedLSTM_complete")

    if mass_predict_neural:
        full_prediciton.mass_predict(iterations=iterations,
                                     step=step,
                                     use_day_model=predict_with_day)

    else:
        full_prediciton.predict(offset=0, use_day_model=predict_with_day)

res_prediction = None
seasonal_pred = None
i = 0
decomp_error = 0
sum_pred = 0
if predict_remainder or predict_decomposed:
    # Residual
    res_prediction = NeuralNetPrediction(datacolumn="Remainder",
                                         data=data,
                                         future_target=future_target,
                                         test_split_at_hour=test_split_at_hour,
                                         net_type="remainder",
                                         train_day_of_week=train_day_of_week)

    if train_residual:
        res_prediction.initialize_network()
        res_prediction.train_network(savename="trainedLSTM_resid",
                                     save=False,
                                     lr_schedule="polynomal",
                                     power=2)
        # lr_schedule="polynomal" oder "step
    else:
        res_prediction.load_model(savename="trainedLSTM_resid")


    if mass_predict_neural:
        res_prediction.mass_predict(iterations=iterations,
                                    step=step,
                                    use_day_model=predict_with_day)
    else:
        # Remainder
        res_prediction.predict(offset=0, use_day_model=predict_with_day)

timeframe = slice(i - test_split_at_hour,
                          future_target + i - test_split_at_hour)
if predict_decomposed:
        # copy so original doesnt get overwritten when adding other components
        sum_pred = res_prediction.pred.copy()

        # Seasonal
        seasonal_pred = StatisticalPrediction(data=data,
                                              future_target=future_target,
                                              offset=i,
                                              test_split_at_hour=test_split_at_hour,
                                              component="Seasonal")
        seasonal_pred.predict("AutoReg")
        sum_pred += seasonal_pred.pred

        # Trend
        trend_pred = StatisticalPrediction(data=data,
                                           future_target=future_target,
                                           offset=i,
                                           test_split_at_hour=test_split_at_hour,
                                           component="Trend")
        trend_pred.predict("AutoReg")
        sum_pred += trend_pred.pred

        # add error
        decomp_error += np.around(np.sqrt(np.mean(np.square(
            data["Price"].iloc[timeframe] - sum_pred))),
            2)
        i += 1

if predict_naive:
    naive_complete_pred = StatisticalPrediction(data=data,
                                                future_target=future_target,
                                                offset=0,
                                                test_split_at_hour=test_split_at_hour,
                                                component="Price")
    naive_complete_pred.predict(method="naive")

if mass_predict_neural == False:
    num_axes=0
    used_axes=0
    if predict_decomposed:
        num_axes+=4
    elif predict_remainder:
        num_axes+=1
    if predict_naive:
        num_axes += 1
    if predict_complete and not predict_decomposed:
        num_axes += 1

    fig, ax = plt.subplots(num_axes, 1, sharex=True, figsize=(10.0, 10.0))  #


    index = data.iloc[timeframe].index
    if predict_complete or predict_decomposed:
        ax[0].plot(index, data["Price"].iloc[timeframe],
                   label="Truth")
        used_axes += 1


    if predict_decomposed:
        ax[0].plot(index, sum_pred,
                   label='decomposed; RMSE : {}'.format(
                       decomp_error))

        ax[2].plot(index, data["Seasonal"].iloc[timeframe],
                   label='truth')
        ax[2].plot(index, seasonal_pred.pred,
                   label="prediction ; Error: {}".format(
                       seasonal_pred.error))
        ax[3].plot(index, data["Trend"].iloc[timeframe],
                   label='truth')
        ax[3].plot(index, trend_pred.pred,
                   label="prediction ; Error: {}".format(
                       trend_pred.error))

        ax[2].set_ylabel("Seasonal")
        ax[3].set_ylabel("Trend")

        ax[2].legend()
        ax[3].legend()
        used_axes += 2

    if predict_complete:

        if predict_with_day:
            ax[0].plot(index, full_prediciton.pred,
                       label='complete, day_model: {}; RMSE: {}'.format(
                           predict_with_day, full_prediciton.error))
            full_prediciton.predict(use_day_model=False)
            ax[0].plot(index, full_prediciton.pred,
                   label='complete, normal_model; RMSE: {}'.format(
                       full_prediciton.error))
        else:
            ax[0].plot(index, full_prediciton.pred,
                       label='complete, normal_model: {}; RMSE: {}'.format(
                           predict_with_day, full_prediciton.error))
            full_prediciton.predict(use_day_model=True)
            ax[0].plot(index, full_prediciton.pred,
                       label='complete, day_model ; RMSE: {}'.format(
                           full_prediciton.error))

    if predict_decomposed or predict_remainder:
        ax[1].plot(index, data["Remainder"].iloc[timeframe],
                   label="Truth")
        if predict_with_day:
            ax[1].plot(index, res_prediction.pred,
                       label='remainder, day_model: {}; RMSE: {}'.format(
                           predict_with_day, res_prediction.error))
            res_prediction.predict(use_day_model=False)
            ax[1].plot(index, res_prediction.pred,
                   label='remainder, normal_model; RMSE: {}'.format(
                       res_prediction.error))
        else:
            ax[1].plot(index, res_prediction.pred,
                       label='remainder, normal_model: {}; RMSE: {}'.format(
                           predict_with_day, res_prediction.error))
            res_prediction.predict(use_day_model=True)
            ax[1].plot(index, res_prediction.pred,
                       label='remainder, day_model ; RMSE: {}'.format(
                           res_prediction.error))


        ax[1].set_ylabel("Remainder")
        ax[1].legend()
        used_axes+=1

    if predict_naive:
        ax[used_axes].plot(index, naive_complete_pred.truth,
               label="Truth")
        ax[used_axes].plot(index, naive_complete_pred.pred,
                   label="prediction; RMSE: {}".format(
                       naive_complete_pred.error))

        ax[used_axes].set_ylabel("Naive Prediction")
        ax[used_axes].legend()

    if predict_complete or predict_decomposed:
        ax[0].set_ylabel("Komplett")
        ax[0].legend()

    if False:
        plt.savefig(
           "Abbildungen/prediction_{}.png".format(test_pred_start_hour),
           dpi=300, bbox_inches='tight')
    else:
        fig.suptitle(
            "Start: {} Stunden nach Trainingsende.".format(
                test_pred_start_hour))
        plt.show()
