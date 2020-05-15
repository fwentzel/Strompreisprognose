import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from zipfile import ZipFile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from statsmodels.tsa.seasonal import STL,seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import sys
import os.path
import time
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import csv_reader
import numpy as np


def fill_power_na(series):
    series_copy = series.copy()
    shifted_series = series_copy.iloc[168:]
    lagged_series = series_copy  # index  0 will be the value lagged by 168 hours (1 week)
    mask = shifted_series.isna()
    values = []
    for i in range(len(mask) - 1):
        if mask[i + 1] and mask[
            i] == False:  # look for begining of missing day
            j = i + 1
            if mask[i + 2] and mask[i + 3] and mask[
                i + 4]:  # up to 4 nan can be interpolated. everything else needs better fill methods
                values.append(
                    [shifted_series.iloc[i], lagged_series.iloc[i]])
                while mask[j]:
                    values.append(
                        [shifted_series.iloc[j], lagged_series.iloc[j]])
                    j += 1
                values2 = np.array(values)
                diff = (values2[0][0] - values2[0][1])
                values2[:][1] = values2[:][1] + diff
                set = values2[1:, 1]
                value_series = pd.Series(set, index=mask.index[i + 1:j])
                series_copy = series_copy.fillna(value_series)

        # plt.plot(range(len(nanSeries)), laggedSeries, label="24.7.2019")
        # plt.plot(range(len(nanSeries)), nanSeries, label="31.7.2019")
        # plt.legend()
        # plt.show()
    series_copy = series_copy.interpolate()  # interpolate small missing nans (1-4 hours of missing data)
    sum2 = series_copy.isna().sum()
    return series_copy


def plot_decomposed_data(components):
    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].title.set_text("DIFFERENZIERTER PREIS")
    ax[0].plot(components.observed)
    ax[1].title.set_text("REMAINDER")
    ax[1].plot(components.resid)
    ax[2].title.set_text("SEASONAL")
    ax[2].plot(components.seasonal)
    ax[3].title.set_text("TREND")
    ax[3].plot(components.trend)
    # plt.savefig("{}.png".format(i))
    plt.show()


# LatexDecomposeMarkerStart
def decompose_data(price_series):
    i = 0
    while price_series.index[i].hour != 0:
        i += 1
    new_frame = pd.DataFrame(price_series.iloc[i:])

    new_frame["diff_price"]=new_frame["Price"].diff()
    components_day = STL(new_frame["diff_price"].iloc[1:],
                         seasonal=101).fit()
    new_frame["Remainder"] = components_day.resid
    new_frame["Seasonal"] = components_day.seasonal
    new_frame["Trend"] = components_day.trend
    #drop differenced Price Column since its not needed anymore
    #drop differenced Price Column since its not needed anymore
    new_frame=new_frame.drop("diff_price",axis=1)
    return new_frame
# LatexDecomposeMarkerEnd


def update_power_price():
    path = sys.path[0]
    data_path = "{}\\Data".format(path)
    # unix_timestamp = datetime.datetime.now().timestamp() * 1000
    unix_timestamp = 1585951199000

    existing_data = csv_reader.read_power_data()
    start_for_new_data = existing_data.index[-1].timestamp() * 1000
    #start_for_new_data = 1420502400000

    print("opening URL")
    URL = "https://www.smard.de/home/downloadcenter/download_marktdaten/726#!?" \
          "downloadAttributes=%7B%22" \
          "selectedCategory%22:3,%22" \
          "selectedSubCategory%22:8,%22" \
          "selectedRegion%22:%22DE%22,%22" \
          "from%22:{},%22" \
          "to%22:{},%22" \
          "selectedFileType%22:%22CSV%22%7D".format(
        start_for_new_data,
        unix_timestamp)  # 1420502400000 für 6.1.2015; 1546382759999 für 3.4.2020
    options = Options()
    options.add_argument('headless')
    options.add_experimental_option('prefs', {
        "download": {
            "default_directory": data_path
        }
    })
    driver = webdriver.Chrome(
        "./selenium_web_driver/chromedriver.exe",
        options=options)  #
    driver.get(URL)
    button = driver.find_element_by_xpath(
       "//button[@name='button'][@type='button'][contains(text(), 'Datei herunterladen')]")
    print("downloading")

    button.click()

    while not [filename for filename in os.listdir(data_path) if
               filename.startswith("Tabellen_Daten")]:
        print("downloading")
        time.sleep(1)
    print("finished download")

    driver.quit()
    print("reading ZIP")
    for filename in os.listdir(data_path):
        if filename.startswith("Tabellen_Daten"):
            zip_filename = "{}\\{}".format(data_path, filename)
            with ZipFile(zip_filename) as zipFile:
                new_frame = pd.read_csv(
                    zipFile.open(zipFile.namelist()[-1]), sep=';')
                new_frame['MESS_DATUM'] = new_frame[
                    'Datum'].str.cat(new_frame['Uhrzeit'], sep=" ")
                new_frame.rename(
                    columns={
                        "Deutschland/Luxemburg[Euro/MWh]": "Price",
                        "Deutschland/Österreich/Luxemburg[Euro/MWh]": "Price2"},
                    inplace=True)
                new_frame = pd.DataFrame(
                    new_frame[["Price", "Price2", "MESS_DATUM"]])
                new_frame["MESS_DATUM"] = pd.to_datetime(
                    new_frame["MESS_DATUM"],
                    format="%d.%m.%Y %H:%M")
                new_frame.set_index("MESS_DATUM", inplace=True)
                new_frame["Price"] = new_frame["Price"].apply(
                    lambda x: x.replace(",", "."))
                new_frame["Price2"] = new_frame["Price2"].apply(
                    lambda x: x.replace(",", "."))

                localized_frame = new_frame.tz_localize(
                    tz='Europe/Berlin', ambiguous="infer").asfreq(
                    freq='H')
                localized_frame["Price"] = pd.to_numeric(
                    localized_frame["Price"], errors="coerce")
                localized_frame["Price2"] = pd.to_numeric(
                    localized_frame["Price2"], errors="coerce")
                localized_frame["Price"] = localized_frame.mean(axis=1)
                localized_frame.drop("Price2", axis=1, inplace=True)

                # sum of missing new_config
                sum = localized_frame["Price"].isna().sum()
                if sum > 0:
                    print(
                        "Missing new_config after Timezone Conversion detected. "
                        "Automatically filling these new_config, but consider redownloading.")
                    fill_power_na(localized_frame["Price"])
            # os.remove(zip_filename)
    existing_data = existing_data.append(localized_frame)
    existing_data.index = pd.to_datetime(existing_data.index,utc=True)
    existing_data.dropna(inplace=True, how='all')
    existing_data = decompose_data(existing_data)
    existing_data.to_csv("Data/price.csv")
    print("finished")
    return existing_data


def updateWeatherHistory():
    existing_data = csv_reader.read_weather_data().asfreq("H")
    start = existing_data.index[-1] + datetime.timedelta(hours=1)
    end = datetime.date.today() - datetime.timedelta(days=1, hours=0)
    i = 0
    new_data = pd.DataFrame(
        pd.date_range(start=start, end=end, freq="H"),
        columns=["MESS_DATUM"])
    new_data.set_index("MESS_DATUM", inplace=True)

    params = {"wind": "   F", "sun": "SD_SO", "cloudiness": " V_N",
              "air_temperature": "TT_TU"}

    for val in params:
        temp_frame = pd.DataFrame(
            pd.date_range(start=start, end=end, freq="H"),
            columns=["MESS_DATUM"]).set_index("MESS_DATUM")
        for timeMode in ["recent"]:  # "historical",
            _URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/{}/{}/".format(
                val, timeMode)
            r = urlopen(_URL)
            soup = bs(r.read(), features="html.parser")
            links = soup.findAll('a')[4:]
            maximum = len(links)
            for j, link in enumerate(links):
                print("\r{} :{}/{}".format(val, j + 1, maximum),
                      sep=' ', end='', flush=True)
                _FULLURL = _URL + link.get('href')
                resp = urlopen(_FULLURL)
                zipfile = ZipFile(BytesIO(resp.read()))
                file = zipfile.namelist()[-1]
                downloaded_frame = pd.read_csv(zipfile.open(file),
                                               sep=';',
                                               index_col="MESS_DATUM",
                                               na_values="-999")
                downloaded_frame.index = pd.to_datetime(
                    downloaded_frame.index,
                    format='%Y%m%d%H',utc=True)
                downloaded_frame = downloaded_frame.loc[
                    downloaded_frame.index >= start]
                column = "{}_{}".format(params[val], timeMode)
                temp_frame[column] = pd.concat(
                    [temp_frame, downloaded_frame[params[val]]],
                    axis=1).mean(axis=1)
                sum = temp_frame.isna().sum()
        temp_frame[params[val]] = temp_frame.mean(axis=1)
        new_data = pd.concat(
            [new_data, temp_frame[params[val]]], axis=1)
        sum = new_data.isna().sum()
        i += 1

    new_data.columns = [x for x in params]
    existing_data = existing_data.append(new_data)
    existing_data["sun"].fillna(0, inplace=True)
    existing_data.to_csv("Data/weather.csv")
    return existing_data


def updateForecast(properties=["FF", "N", "SunD1", "TTT"],
                   updateGermanCities=False, ):
    print("downloading Forecast")
    resp = urllib.request.urlopen(
        "https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz")
    print("downloaded")
    kmz = ZipFile(BytesIO(resp.read()), 'r')
    kml = kmz.open(kmz.namelist()[0], 'r').read()
    root = ET.fromstring(kml)

    namespace = {"kml": "http://www.opengis.net/kml/2.2",
                 "dwd": "https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd"}
    cities = pd.read_csv("Data/germanCities.csv")
    cities = cities["Ort"].astype(str).values

    print("reading XML")
    last_frame = pd.DataFrame()

    index_list = []
    for time_step in root.iter(
            '{https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd}TimeStep'):
        index_list.append(time_step.text)
    for prop in properties:
        for placemark in root.iter(
                '{http://www.opengis.net/kml/2.2}Placemark'):
            city = placemark.find("kml:description", namespace).text
            if city in cities:  # stadt in DE
                df = pd.DataFrame()
                forecast = placemark.find(
                    "./kml:ExtendedData/dwd:Forecast[@dwd:elementName='{}']".format(
                        prop),
                    namespace)
                df[city] = list(map(float, forecast[0].text.replace("-",
                                                                    "-999").split()))
                last_frame[prop] = df.mean(axis=1)
    last_frame["Date"] = index_list
    last_frame["Date"] = pd.to_datetime(last_frame['Date'])
    last_frame.set_index("Date", inplace=True)
    print("writing to forecast.csv")
    last_frame.to_csv("Data/forecast.csv")

    if (updateGermanCities):
        print("updating germanCities.csv")
        elemts = root[0].findall("./kml:Placemark/kml:description",
                                 namespace)
        cityList = []
        for element in elemts:
            cityList.append(element.text)
        citiesDWD = pd.DataFrame(cityList, columns=["Ort"])
        citiesDWD['Ort'] = citiesDWD["Ort"].apply(lambda x: x.upper())

        cities = pd.DataFrame(
            pd.read_csv("deCitiescompare.csv", delimiter=";")["Ort"],
            columns=["Ort"])
        cities['Ort'] = cities["Ort"].apply(lambda x: x.upper())

        mergedStuff = pd.merge(cities, citiesDWD, on=['Ort'],
                               how='inner')
        mergedStuff = mergedStuff.drop_duplicates()
        mergedStuff.to_csv("Data/germanCities.csv", index=False)
