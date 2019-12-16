import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
# import holidays
from datetime import date



powerScaler = MinMaxScaler(feature_range=(-1, 1))
tempScaler = MinMaxScaler(feature_range=(0, 1))
hourScaler=MinMaxScaler(feature_range=(0,1))

def differenceData(data,scaler):
    diff = list()
    diff.append(0)
    for i in range(1, len(data)):
        value = data[i] - data[i - 1]
        diff.append(value)
    diff=np.array(diff).reshape(-1,1)

    scaled_values = scaler.fit_transform(diff)
    return scaled_values

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return np.array(inverted).reshape(len(inverted))

# inverse data transform on forecasts
def inverse_transform( forecast, scaler,last_ob):

    # create array from forecast
    forecast = forecast.reshape(1, len(forecast))
    # invert scaling
    inv_scale = scaler.inverse_transform(forecast)
    inv_scale = inv_scale[0, :]

    # invert differencing
    inv_diff = inverse_difference(last_ob, inv_scale)
    # store
    return inv_diff


def getData(weatherparameter=["air_temperature","cloudiness","sun","wind"]):
    weatherFrame=pd.DataFrame()
    for param in weatherparameter:
        paramFrame=pd.read_csv("Data/{}_historical.csv".format(param),index_col="MESS_DATUM")
        paramFrame=paramFrame.append(pd.read_csv("Data/{}_recent.csv".format(param),index_col="MESS_DATUM"))
        weatherFrame=pd.concat([weatherFrame,paramFrame],axis=1,sort=True)
    powerPrice=pd.read_csv('Data/powerpriceData.csv')
    powerPrice['Date'] = pd.to_datetime(powerPrice['Date'],unit='ms')
    powerPrice = powerPrice.set_index('Date')
    powerPrice['Price']=pd.to_numeric(powerPrice['Price'], errors='coerce')
    powerPrice['diffScaledPrice']=differenceData(powerPrice['Price'],powerScaler)

    data=powerPrice.join(weatherFrame, how='inner')
    data['scaledTemp']= tempScaler.fit_transform(np.array(data['TT_TU']).reshape(-1,1))
    data.drop('TT_TU',axis=1,inplace=True)
    data['Weekend'] = (pd.DatetimeIndex(data.index).dayofweek>5).astype(int)
    data['Hour']=hourScaler.fit_transform(np.array(data.index.hour).reshape(-1,1))

    # holidaysGer=holidays.Germany()
    # data["Holiday"]=(pd.DatetimeIndex(data.index).date)
    # data["Holiday"]=data["Holiday"].apply(lambda dateToCheck :dateToCheck in holidaysGer).astype(float)
    data=data.interpolate()
    data=data.fillna(value={"SD_SO":0," V_N":-1})
    return data

