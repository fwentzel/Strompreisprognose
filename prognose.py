from __future__ import absolute_import, division, print_function, \
    unicode_literals
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from csv_reader import get_data
from neuralNetPrediction import NeuralNetPrediction
from statisticalPrediction import StatisticalPrediction
import ipykernel  # fix progress bar

register_matplotlib_converters()
future_target = 24
iterations = 168  # amount of predicitons for mass predict
step = 1
train_complete = False
train_residual = False
train_day_of_week = False

predict_complete = True
predict_decomposed = True
predict_with_day = False
mass_predict_neural = False
test_pred_start_hour = 0
plot_all = mass_predict_neural == False
test_length = future_target + iterations + 172 # Timesteps for testing.
data = get_data()
test_split_at_hour = data.index[
                         -test_length].hour - test_pred_start_hour + test_length
# test_data.Price.plot()
# plt.show()


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
                                     use_day_model=True)

    else:
        full_prediciton.predict(offset=0, use_day_model=predict_with_day)

res_prediction = None
seasonal_pred = None
i = 0
decomp_error = 0
sum_pred = 0
if predict_decomposed:
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
                                    step=step)
    else:
        # Remainder
        res_prediction.predict(offset=0, use_day_model=predict_with_day)




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
        timeframe = slice(i - test_split_at_hour,
                          future_target + i - test_split_at_hour)
        decomp_error += np.around(np.sqrt(np.mean(np.square(
            data["Price"].iloc[timeframe] - sum_pred))),
            2)
        i += 1
    # with open('Results/residual_results.csv', 'a', newline='') as fd:
    #     writer = csv.writer(fd)
    #     writer.writerow([learning_rate, res_prediction.error,
    #                      res_prediction.single_errors.tolist()])

naive_complete_pred = StatisticalPrediction(data=data,
                                            future_target=future_target,
                                            offset=0,
                                            test_split_at_hour=test_split_at_hour,
                                            component="Price")
naive_complete_pred.predict(method="naive")

# decomp_error /= (i + 1)
if mass_predict_neural == False and predict_complete == True and predict_decomposed == True:
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(10.0, 10.0))  #


    index = data.iloc[timeframe].index
    ax[0].plot(index, data["Price"].iloc[timeframe],
               label="Truth")
    ax[1].plot(index, data["Remainder"].iloc[timeframe],
               label="Truth")
    ax[2].plot(index, data["Seasonal"].iloc[timeframe],
               label='truth')
    ax[3].plot(index, data["Trend"].iloc[timeframe],
               label='truth')
    ax[4].plot(index, naive_complete_pred.truth,
               label="Truth")

    ax[0].plot(index, sum_pred,
               label='decomposed; RMSE : {}'.format(
                   decomp_error))
    ax[0].plot(index, full_prediciton.pred,
               label='complete; RMSE: {}'.format(
                   full_prediciton.error))

    ax[1].plot(index, res_prediction.pred,
               label='Remainder prediciton ; RMSE: '
                     '{}'.format(
                   res_prediction.error))
    ax[2].plot(index, seasonal_pred.pred,
               label="prediction ; Error: {}".format(
                   seasonal_pred.error))
    ax[3].plot(index, trend_pred.pred,
               label="prediction ; Error: {}".format(
                   trend_pred.error))
    ax[4].plot(index, naive_complete_pred.pred,
               label="prediction; RMSE: {}".format(
                   naive_complete_pred.error))

    ax[0].set_ylabel("Komplett")
    ax[1].set_ylabel("Remainder")
    ax[2].set_ylabel("Seasonal")
    ax[3].set_ylabel("Trend")
    ax[4].set_ylabel("Naive Prediction")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[4].legend()
    # Plot the predictions of components and their combination with the
    # corresponding truth

    #fig.suptitle(
    #    "Start: {} Stunden nach Trainingsende des anderen Versuchs".format(
     #       test_pred_start_hour))
    #plt.savefig(
    #    "Abbildungen/prediction_{}.png".format(test_pred_start_hour),
    #    dpi=300, bbox_inches='tight')
    plt.show()
