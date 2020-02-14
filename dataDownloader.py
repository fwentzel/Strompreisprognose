import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from bs4 import BeautifulSoup as bs


##############################################################################################################
###############################################Strompreis#####################################################
##############################################################################################################
def update_power_price():
    print("downloading power prices")
    value_frame = pd.DataFrame()
    for year in range(16, 20):
        for month in range(1, 13):
            if (month < 10):
                month = "0" + str(month)
            url = "https://energy-charts.de/price/month_20{}_{}.json".format(year, month)
            json = urllib.request.urlopen(url)
            data = pd.read_json(json)
            value_series = pd.Series(data["values"].iloc[5])
            value_frame = value_frame.append(pd.DataFrame(value_series.values.tolist()))
    print("Writing to powerpriceData.csv")
    value_frame.columns = ["Date", "Price"]
    value_frame.to_csv('Data/powerpriceData.csv', index=False)


##############################################################################################################
###############################################Wetterhistory##################################################
##############################################################################################################
def updateWeatherHistory(parameter=["air_temperature", "cloudiness", "sun", "wind"],
                         shortform=["TT_TU", " V_N", "SD_SO", "   F"], times=["recent", "historical"],
                         start='2016-1-1', end='2019-12-16'):
    i = 0
    final_frame = pd.DataFrame(pd.date_range(start=start, end=end, freq="H"), columns=["MESS_DATUM"])
    final_frame.set_index("MESS_DATUM", inplace=True)
    for param in parameter:
        timesCombinedFrame = pd.DataFrame(columns=["MESS_DATUM", shortform[i]])
        timesCombinedFrame.set_index("MESS_DATUM", inplace=True)
        for timeMode in times:
            df = pd.DataFrame(pd.date_range(start=start, end=end, freq="H"), columns=["MESS_DATUM"])
            df.set_index("MESS_DATUM", inplace=True)
            print("")
            _URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/{}/{}/".format(
                param, timeMode)
            r = urlopen(_URL)
            soup = bs(r.read(), features="html.parser")
            links = soup.findAll('a')
            maximum = len(links)
            for j, link in enumerate(links[20:40]):
                print("\r{} {}:{}/{}".format(param, timeMode, j, maximum), sep=' ', end='', flush=True)
                if link.get('href').endswith('.zip'):
                    _FULLURL = _URL + link.get('href')
                    resp = urlopen(_FULLURL)
                    zipfile = ZipFile(BytesIO(resp.read()))
                    file = zipfile.namelist()[-1]
                    tempdf = pd.read_csv(zipfile.open(file), sep=';', index_col="MESS_DATUM")
                    tempdf.index = pd.to_datetime(tempdf.index, format='%Y%m%d%H')
                    df[shortform[i].strip()] = pd.concat([df, tempdf[shortform[i]]], axis=1).mean(axis=1)
            df.dropna(inplace=True)
            if timeMode == "recent":
                recent_first_date = df.index[0]
            if timeMode == "historical":
                df = df.loc[df.index < recent_first_date]
            df = df.loc[df.index >= start]
            df.to_csv("Data/{}_{}.csv".format(param, timeMode))
        i += 1


##############################################################################################################
###############################################Vorhersage#####################################################
##############################################################################################################
def updateForecast(properties=["FF", "N", "SunD1", "TTT"], WriteXML=False, updateGermanCities=False, ):
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
    for time_step in root.iter('{https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd}TimeStep'):
        index_list.append(time_step.text)
    for prop in properties:
        for placemark in root.iter('{http://www.opengis.net/kml/2.2}Placemark'):
            city = placemark.find("kml:description", namespace).text
            if city in cities:  # stadt in DE
                df = pd.DataFrame()
                forecast = placemark.find("./kml:ExtendedData/dwd:Forecast[@dwd:elementName='{}']".format(prop),
                                          namespace)
                df[city] = list(map(float, forecast[0].text.replace("-", "-999").split()))
                last_frame[prop] = df.mean(axis=1)
    last_frame["Date"] = index_list
    last_frame["Date"] = pd.to_datetime(last_frame['Date'])
    last_frame.set_index("Date", inplace=True)
    print("writing to forecast.csv")
    last_frame.to_csv("Data/forecast.csv")

    # if(updateGermanCities):
    #     print("updating germanCities.csv")
    #     elemts=root[0].findall("./kml:Placemark/kml:description",namespace)
    #     cityList=[]
    #     for element in elemts:
    #         cityList.append(element.text)
    #     citiesDWD=pd.DataFrame(cityList,columns=["Ort"])
    #     citiesDWD['Ort']=citiesDWD["Ort"].apply(lambda x: x.upper())

    #     cities=pd.DataFrame(pd.read_csv("deCitiescompare.csv",delimiter=";")["Ort"],columns=["Ort"])
    #     cities['Ort']=cities["Ort"].apply(lambda x: x.upper())

    #     mergedStuff = pd.merge(cities, citiesDWD, on=['Ort'], how='inner')
    #     mergedStuff=mergedStuff.drop_duplicates()
    #     mergedStuff.to_csv("Data/germanCities.csv",index=False)
