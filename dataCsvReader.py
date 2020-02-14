import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import holidays
from datetime import date

powerScaler = MinMaxScaler(feature_range=(-1, 1))
tempScaler = MinMaxScaler(feature_range=(0, 1))
hourScaler = MinMaxScaler(feature_range=(0, 1))


def differenceData(data, scaler):
    diff = list()
    diff.append(0)
    for i in range(1, len(data)):
        value = data[i] - data[i - 1]
        diff.append(value)
    diff = np.array(diff).reshape(-1, 1)

    scaled_values = scaler.fit_transform(diff)
    return scaled_values


# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return np.array(inverted).reshape(len(inverted))


# inverse data transform on forecasts
def inverse_transform(forecast, scaler, last_ob):
    # create array from forecast
    forecast = forecast.reshape(1, len(forecast))
    # invert scaling
    inv_scale = scaler.inverse_transform(forecast)
    inv_scale = inv_scale[0, :]

    # invert differencing
    inv_diff = inverse_difference(last_ob, inv_scale)
    # store
    return inv_diff


def getData(start='2016-1-1', end='2019-12-16', weatherparameter=["air_temperature", "cloudiness", "sun", "wind"]):
    weatherFrame = pd.DataFrame(pd.date_range(start=start, end=end, freq="H"), columns=["MESS_DATUM"])
    weatherFrame.set_index("MESS_DATUM", inplace=True)
    for param in weatherparameter:
        paramFrame = pd.read_csv("Data/{}_historical.csv".format(param), index_col="MESS_DATUM")
        paramFrame2 = pd.read_csv("Data/{}_recent.csv".format(param), index_col="MESS_DATUM")
        if len(paramFrame.index) > 2:
            paramFrame = paramFrame.append(paramFrame2)
        else:
            paramFrame = paramFrame2
        weatherFrame = weatherFrame.join(paramFrame)

    weatherFrame.columns = ["TT_TU", "V_N", "SD_SO", "F"]
    # forecastFrame=pd.read_csv("Data/forecast.csv")
    # forecastFrame["Date"]=pd.to_datetime(forecastFrame['Date'])
    # forecastFrame.columns=["MESS_DATUM","TT_TU","V_N","SD_SO","F"]
    # forecastFrame.set_index("MESS_DATUM",inplace=True)
    # forecastFrame=forecastFrame.tz_localize(None)
    # weatherFrame=weatherFrame.append(forecastFrame)

    powerPrice = pd.read_csv('Data/powerpriceData.csv')
    powerPrice['Date'] = pd.to_datetime(powerPrice['Date'], unit='ms')
    powerPrice = powerPrice.set_index('Date')
    powerPrice['Price'] = pd.to_numeric(powerPrice['Price'], errors='coerce')
    # powerPrice['diffScaledPrice']=differenceData(powerPrice['Price'],powerScaler)

    data = powerPrice.join(weatherFrame, how='outer')
    # data['scaledTemp']= tempScaler.fit_transform(np.array(data['TT_TU']).reshape(-1,1))
    data['Temp'] = data['TT_TU']
    data.drop('TT_TU', axis=1, inplace=True)
    data['Weekend'] = (pd.DatetimeIndex(data.index).dayofweek > 5).astype(int)
    # data['Hour']=hourScaler.fit_transform(np.array(data.index.hour).reshape(-1,1))
    data["Hour"] = data.index.hour

    holidaysGer = holidays.Germany()
    data["Holiday"] = (pd.DatetimeIndex(data.index).date)
    data["Holiday"] = data["Holiday"].apply(lambda dateToCheck: dateToCheck in holidaysGer).astype(float)

    data = data.fillna(value={"SD_SO": 0, "V_N": -1, "F": 0, "scaledTemp": 0, "Temp": 0})
    data.dropna(inplace=True)
    return data  # , forecastFrame.index[0]
