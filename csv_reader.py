import holidays
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import data_downloader

power_scaler = MinMaxScaler(feature_range=(-1, 1))
temp_scaler = MinMaxScaler(feature_range=(0, 1))
hour_scaler = MinMaxScaler(feature_range=(0, 1))


def get_data(update_weather_data, update_price_data, test_length):
    weather_frame = data_downloader.updateWeatherHistory() if update_weather_data else read_weather_data()
    power_price_frame = data_downloader.update_power_price() if update_price_data else read_power_data()

    data = power_price_frame.join(weather_frame, how='inner')
    data['Weekend'] = (pd.DatetimeIndex(data.index).dayofweek > 5).astype(int)
    data["Hour"] = data.index.hour
    read_holidays(data)
    data["Price"].plot()
    plt.ylabel("Strompreis [€/MWh]")
    plt.xlabel("")
    plt.show()
    test_data = data.iloc[-test_length:]  # Part of data the network wont see during Training and validation
    train_data = data.iloc[:-test_length]
    return train_data, test_data  # , forecast_frame.index[0]


def read_holidays(data):
    holidays_ger = holidays.Germany()
    data["Holiday"] = pd.DatetimeIndex(data.index).date
    data["Holiday"] = data["Holiday"].apply(lambda date_to_check: date_to_check in holidays_ger).astype(float)


def read_power_data():
    power_price = pd.read_csv("Data/price.csv", index_col="MESS_DATUM")
    power_price.index = pd.to_datetime(power_price.index)
    # power_price['diffScaledPrice']=differenceData(power_price['Price'],power_scaler)
    return power_price


def read_weather_data():
    weather_frame = pd.read_csv("Data/weather.csv", index_col="MESS_DATUM")
    weather_frame.index = pd.to_datetime(weather_frame.index)
    return weather_frame




