import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

power_scaler = MinMaxScaler(feature_range=(-1, 1))
temp_scaler = MinMaxScaler(feature_range=(0, 1))
hour_scaler = MinMaxScaler(feature_range=(0, 1))


def difference_data(data, scaler):
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


def get_data(start='2016-1-1', end='2019-12-16', weatherparameter=["air_temperature", "cloudiness", "sun", "wind"]):
    weather_frame = pd.DataFrame(pd.date_range(start=start, end=end, freq="H"), columns=["MESS_DATUM"])
    weather_frame.set_index("MESS_DATUM", inplace=True)
    for param in weatherparameter:
        param_frame = pd.read_csv("Data/{}_historical.csv".format(param), index_col="MESS_DATUM")
        param_frame2 = pd.read_csv("Data/{}_recent.csv".format(param), index_col="MESS_DATUM")
        if len(param_frame.index) > 2:
            param_frame = param_frame.append(param_frame2)
        else:
            param_frame = param_frame2
        weather_frame = weather_frame.join(param_frame)

    weather_frame.columns = ["TT_TU", "V_N", "SD_SO", "F"]
    # forecast_frame=pd.read_csv("Data/forecast.csv")
    # forecast_frame["Date"]=pd.to_datetime(forecast_frame['Date'])
    # forecast_frame.columns=["MESS_DATUM","TT_TU","V_N","SD_SO","F"]
    # forecast_frame.set_index("MESS_DATUM",inplace=True)
    # forecast_frame=forecast_frame.tz_localize(None)
    # weather_frame=weather_frame.append(forecast_frame)

    power_price = pd.read_csv('Data/powerpriceData.csv')
    power_price['Date'] = pd.to_datetime(power_price['Date'], unit='ms')
    power_price = power_price.set_index('Date')
    power_price['Price'] = pd.to_numeric(power_price['Price'], errors='coerce')
    # power_price['diffScaledPrice']=differenceData(power_price['Price'],power_scaler)

    data = power_price.join(weather_frame, how='outer')
    # data['scaledTemp']= temp_scaler.fit_transform(np.array(data['TT_TU']).reshape(-1,1))
    data['Temp'] = data['TT_TU']
    data.drop('TT_TU', axis=1, inplace=True)
    data['Weekend'] = (pd.DatetimeIndex(data.index).dayofweek > 5).astype(int)
    # data['Hour']=hour_scaler.fit_transform(np.array(data.index.hour).reshape(-1,1))
    data["Hour"] = data.index.hour

    holidays_ger = holidays.Germany()
    data["Holiday"] = pd.DatetimeIndex(data.index).date
    data["Holiday"] = data["Holiday"].apply(lambda date_to_check: date_to_check in holidays_ger).astype(float)

    data = data.fillna(value={"SD_SO": 0, "V_N": -1, "F": 0, "scaledTemp": 0, "Temp": 0})
    data.dropna(inplace=True)
    return data  # , forecast_frame.index[0]
