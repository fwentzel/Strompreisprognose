import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from zipfile import ZipFile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from statsmodels.tsa.seasonal import STL
import sys
import os.path
import time
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def fill_power_na(series):
    mask = series.isna()
    newMask = mask.copy()
    for i in range(3, len(mask) - 3):
        if mask[i + 3] or mask[i - 3]:
            newMask[i] = True
    for i in range(168, len(newMask) - 168):
        if newMask[i + 168]:
            newMask[i] = True
    nanFrame = series[newMask]
    split = int(len(nanFrame) / 4)
    for i in range(2):
        if i == 0:
            laggedSeries = nanFrame.iloc[:split]
            nanSeries = nanFrame.iloc[split:split * 2]
        else:
            laggedSeries = nanFrame.iloc[split * 2:split * 3]
            nanSeries = nanFrame.iloc[split * 3:]

        diff = laggedSeries[2] - nanSeries[2]
        laggedSeries = laggedSeries.apply(lambda x: x - diff)
        laggedSeries.index = laggedSeries.index.shift(7, freq='D')
        series.loc[laggedSeries.index] = laggedSeries

        # plt.plot(range(len(nanSeries)), laggedSeries, label="24.7.2019")
        # plt.plot(range(len(nanSeries)), nanSeries, label="31.7.2019")
        # plt.legend()
        # plt.show()


def plot_decomposed_data(data):
    fig, ax = plt.subplots(4, 1, sharex=True)
    data = data.iloc[1000:]
    ax[0].title.set_text("ORIGINAL")
    ax[0].plot(data["Price"])
    ax[1].title.set_text("RESIDUAL")
    ax[1].plot(data["Remainder"])
    ax[2].title.set_text("SEASONAL")
    ax[2].plot(data["Seasonal"])
    ax[3].title.set_text("TREND")
    ax[3].plot(data["Trend"])
    # plt.savefig("{}.png".format(i))
    plt.show()


# LatexDecomposeMarkerStart
def decompose_data(data_frame):
    i = 0
    while data_frame.index[i].hour != 0:
        i += 1

    if i > 0:
        new_frame = data_frame.iloc[i:]
    else:
        new_frame = data_frame

    components = STL(new_frame["Price"], seasonal=13).fit()
    new_frame["Remainder"] = components.resid
    new_frame["Seasonal"] = components.seasonal
    new_frame["Trend"] = components.trend
    #plot_decomposed_data(new_frame)
    return new_frame


# LatexDecomposeMarkerEnd


def update_power_price():
    path = sys.path[0]
    data_path = "{}\\Data".format(path)
    getNewData = False

    milliseconds_since_epoch = datetime.datetime.now().timestamp() * 1000

    if getNewData:
        print("opening URL")
        URL = "https://www.smard.de/home/downloadcenter/download_marktdaten/726#!?" \
              "downloadAttributes=%7B%22" \
              "selectedCategory%22:3,%22" \
              "selectedSubCategory%22:8,%22" \
              "selectedRegion%22:%22DE%22,%22" \
              "from%22:1420502400000,%22" \
              "to%22:{},%22" \
              "selectedFileType%22:%22CSV%22%7D".format(
            milliseconds_since_epoch)  # 1420502400000 für 6.1.2015; 1538352000000 für 1.10.2018
        options = Options()
        # options.add_argument('headless')
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
            print("not there yet")
            time.sleep(2)
        print("finished")

        driver.quit()
    print("reading ZIP")
    for filename in os.listdir(data_path):
        if filename.startswith("Tabellen_Daten"):
            zip_filename = "{}\\{}".format(data_path, filename)
            with ZipFile(zip_filename) as zipFile:
                power_frame = pd.read_csv(
                    zipFile.open(zipFile.namelist()[-1]), sep=';')
                power_frame['MESS_DATUM'] = power_frame[
                    'Datum'].str.cat(power_frame['Uhrzeit'], sep=" ")
                power_frame.rename(
                    columns={"Deutschland/Luxemburg[Euro/MWh]": "Price",
                             "Deutschland/Österreich/Luxemburg[Euro/MWh]": "Price2"},
                    inplace=True)
                power_frame = pd.DataFrame(
                    power_frame[["Price", "Price2", "MESS_DATUM"]])
                power_frame["MESS_DATUM"] = pd.to_datetime(
                    power_frame["MESS_DATUM"], format="%d.%m.%Y %H:%M")
                power_frame.set_index("MESS_DATUM", inplace=True)
                power_frame["Price"] = power_frame["Price"].apply(
                    lambda x: x.replace(",", "."))
                power_frame["Price2"] = power_frame["Price2"].apply(
                    lambda x: x.replace(",", "."))
                power_frame = power_frame[
                    ~power_frame.index.duplicated()].asfreq(
                    freq='H')  # remove duplicate entries (2 faulty values from database) and set frequency to Hourly
                power_frame["Price"] = pd.to_numeric(
                    power_frame["Price"], errors="coerce")
                power_frame["Price2"] = pd.to_numeric(
                    power_frame["Price2"], errors="coerce")
                power_frame["Price"] = power_frame.mean(axis=1)
                power_frame.drop("Price2", axis=1, inplace=True)
                sum = power_frame["Price"].isna().sum()
                power_frame = power_frame.interpolate(
                    limit=1)  # interpolate to fill missing value 21.3.2019 2:00
                sum = power_frame["Price"].isna().sum()
                fill_power_na(power_frame["Price"])

    power_frame = decompose_data(power_frame)
    print("finished")
    remove_path = zip_filename
    # os.remove(remove_path)
    power_frame.to_csv("Data/price.csv")
    return power_frame

    # print("downloading power prices")
    # value_frame = pd.DataFrame()
    # for year in range(16, 20):
    #     for month in range(1, 13):
    #         if (month < 10):
    #             month = "0" + str(month)
    #         url = "https://energy-charts.de/price/month_20{}_{}.json".format(year, month)
    #         json = urllib.request.urlopen(url)
    #         data = pd.read_json(json)
    #         value_series = pd.Series(data["values"].iloc[5])
    #         value_frame = value_frame.append(pd.DataFrame(value_series.values.tolist()))
    # print("Writing to powerpriceData.csv")
    # value_frame.columns = ["Date", "Price"]
    # value_frame.to_csv('Data/powerpriceData.csv', index=False)


def fill_weather_na(frame):
    # #
    for param in ["sun", "wind", "cloudiness", "air_temperature"]:
        hours = [0 for x in range(24)]
        days = [0 for x in range(7)]
        series = frame[param]
        mask = series.isna()
        p = 0
        for i in range(len(series) - 1):
            if mask[i]:
                if mask[i + 1] == False:
                    p += 1
                hours[series.index[i].hour] += 1
                days[series.index[i].dayofweek] += 1

    # plt.bar([x for x in range(24)], hours)
    # plt.xlabel("Tageszeit")
    # plt.ylabel("Anzahl fehlende Datenpunkte")
    # plt.title("Tageszeit der fehlenden Datenpunkten zu den Sonnenstunden")
    # plt.xticks([x for x in range(0, 24, 2)])
    # plt.show()
    sum = frame.isna().sum()
    # print(hours)
    frame["sun"].fillna(0, inplace=True)

    # frame["wind"].fillna(0, inplace=True)
    sum = frame.isna().sum()

    print("x")


def updateWeatherHistory():
    start = datetime.datetime(2015, 1, 6, 0)
    end = datetime.date.today() - datetime.timedelta(days=1, hours=0)
    i = 0
    weather_frame = pd.DataFrame(
        pd.date_range(start=start, end=end, freq="H"),
        columns=["MESS_DATUM"])
    weather_frame.set_index("MESS_DATUM", inplace=True)

    parameters = {"wind": "   F", "sun": "SD_SO", "cloudiness": " V_N",
                  "air_temperature": "TT_TU"}

    for longform in parameters:
        temp_frame = pd.DataFrame(
            pd.date_range(start=start, end=end, freq="H"),
            columns=["MESS_DATUM"]).set_index("MESS_DATUM")
        for timeMode in ["recent"]:  # "historical",
            _URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/{}/{}/".format(
                longform, timeMode)
            r = urlopen(_URL)
            soup = bs(r.read(), features="html.parser")
            links = soup.findAll('a')[4:]
            maximum = len(links)
            for j, link in enumerate(links):
                print("\r{} :{}/{}".format(longform, j + 1, maximum),
                      sep=' ', end='', flush=True)
                _FULLURL = _URL + link.get('href')
                resp = urlopen(_FULLURL)
                zipfile = ZipFile(BytesIO(resp.read()))
                file = zipfile.namelist()[-1]
                tempdf = pd.read_csv(zipfile.open(file), sep=';',
                                     index_col="MESS_DATUM",
                                     na_values="-999")
                tempdf.index = pd.to_datetime(tempdf.index,
                                              format='%Y%m%d%H')
                tempdf = tempdf.loc[tempdf.index >= start]
                test = pd.concat(
                    [temp_frame, tempdf[parameters[longform]]], axis=1)
                test = test.mean(axis=1)
                temp_frame["{}_{}".format(parameters[longform],
                                          timeMode)] = test
                sum = temp_frame.isna().sum()
        temp_frame[parameters[longform]] = temp_frame.mean(axis=1)
        weather_frame = pd.concat(
            [weather_frame, temp_frame[parameters[longform]]], axis=1)
        sum = weather_frame.isna().sum()
        i += 1

    weather_frame.columns = [x for x in parameters]
    # weather_frame["sun"].fillna(0, inplace=True)
    weather_frame.to_csv("Data/weather.csv")
    return weather_frame


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
