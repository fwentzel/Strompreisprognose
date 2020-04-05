from datetime import datetime
import holidays
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import data_downloader

power_scaler = MinMaxScaler(feature_range=(-1, 1))
temp_scaler = MinMaxScaler(feature_range=(0, 1))
hour_scaler = MinMaxScaler(feature_range=(0, 1))


def get_data(test_length,
             test_pred_start_hour, past_history):
    # Only download new Data when the Data is 1 day old
    power_last_index = read_power_data().index[-1]
    weather_last_index = read_weather_data().index[-1]
    yesterday = datetime.today().day - 1
    get_new_power_data = power_last_index.day < yesterday or \
                         power_last_index.month < datetime.today().month
    get_new_weather_data = weather_last_index.day < yesterday or \
                           weather_last_index.month < datetime.today().month
    #override for testing
    get_new_power_data=False
    get_new_weather_data=False

    weather_frame = data_downloader.updateWeatherHistory() if get_new_weather_data else read_weather_data()
    power_price_frame = data_downloader.update_power_price() if get_new_power_data else read_power_data()

    data = power_price_frame.join(weather_frame, how='inner')
    data['Weekend'] = (
            pd.DatetimeIndex(data.index).dayofweek > 5).astype(int)
    data["Hour"] = data.index.hour
    read_holidays(data)
    test_split_at_hour = data.index[
                             -test_length + past_history].hour - test_pred_start_hour
    test_split_at_hour += test_length
    test_data = data.iloc[-test_split_at_hour-past_history:]
    train_data = data.iloc[:-test_split_at_hour]
    return train_data, test_data


def read_holidays(data):
    holidays_ger = holidays.Germany()
    data["Holiday"] = pd.DatetimeIndex(data.index).date
    data["Holiday"] = data["Holiday"].apply(
        lambda date_to_check: date_to_check in holidays_ger).astype(
        float)


def read_power_data():
    power_price = pd.read_csv("Data/price.csv", index_col="MESS_DATUM")
    power_price.index = pd.to_datetime(power_price.index)
    # power_price['diffScaledPrice']=differenceData(power_price['Price'],power_scaler)
    return power_price


def read_weather_data():
    weather_frame = pd.read_csv("Data/weather.csv",
                                index_col="MESS_DATUM")
    weather_frame.index = pd.to_datetime(weather_frame.index)
    # weather_frame["sun"].fillna(0, inplace=True)
    return weather_frame
