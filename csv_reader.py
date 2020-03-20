import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import data_downloader

power_scaler = MinMaxScaler(feature_range=(-1, 1))
temp_scaler = MinMaxScaler(feature_range=(0, 1))
hour_scaler = MinMaxScaler(feature_range=(0, 1))


def decompose_data(data):
    #TODO naive decomposition - andere mothode finden
    series = data["Price"]
    components = STL(series,seasonal=13).fit()
    # estimated trend, seasonal and remainder components
    data["Remainder"] = components.resid  # the estimated remainder
    data["Seasonal"] = components.seasonal  # The estimated seasonal component
    data["Trend"] = components.trend  # The estimated trend component
    return data

def get_data(update_weather_data,update_price_data,test_length):
    weather_frame = data_downloader.updateWeatherHistory() if update_weather_data else read_weather_data()
    power_price_frame = data_downloader.update_power_price() if update_price_data else read_power_data()
    weather_frame.index=pd.to_datetime(weather_frame.index)
    power_price_frame.index = pd.to_datetime(power_price_frame.index)
    data = power_price_frame.join(weather_frame, how='inner')
    # data['scaledTemp']= temp_scaler.fit_transform(np.array(train_data['TT_TU']).reshape(-1,1))
    data['Weekend'] = (pd.DatetimeIndex(data.index).dayofweek > 5).astype(int)
    data["Hour"] = data.index.hour
    read_holidays(data)
    sum = data.isna().sum()
    data.interpolate(method='bfill',inplace=True)
    sum= data.isna().sum()
    #data = decompose_data(data)
    test_data = data.iloc[-test_length:]  # Part of data the network wont see during Training and validation
    train_data = data.iloc[:-test_length]
    return train_data,test_data  # , forecast_frame.index[0]


def read_holidays(data):
    holidays_ger = holidays.Germany()
    data["Holiday"] = pd.DatetimeIndex(data.index).date
    data["Holiday"] = data["Holiday"].apply(lambda date_to_check: date_to_check in holidays_ger).astype(float)


def read_power_data():
    power_price = pd.read_csv("Data/price.csv", index_col="MESS_DATUM",decimal=",")
    # power_price['diffScaledPrice']=differenceData(power_price['Price'],power_scaler)
    return power_price


def read_weather_data():
    weather_frame = pd.read_csv("Data/weather.csv", index_col="MESS_DATUM")
    weather_frame.columns = ["Temperature", "Wind", "Sun", "Clouds"]
    return weather_frame

def plot_decomposed_data(data):
    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].title.set_text("ORIGINAL")
    ax[0].plot(data["Price"])
    ax[1].title.set_text("RESIDUAL")
    ax[1].plot(data["Remainder"])
    ax[2].title.set_text("SEASONAL")
    ax[2].plot(data["Seasonal"])
    ax[3].title.set_text("TREND")
    ax[3].plot(data["Trend"])

