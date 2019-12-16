import fnmatch
from io import BytesIO
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import time

##############################################################################################################
###############################################Strompreis#####################################################
##############################################################################################################
def updatePowerprice():
    print("downloading Powerprice")
    valueFrame = pd.DataFrame()
    for year in range(16,20):
    	for month in range(1,13):
    		if(month<10):
    			month="0"+ str(month)
    		url="https://energy-charts.de/price/month_20{}_{}.json".format(year,month)
    		json=urllib.request.urlopen(url)
    		data= pd.read_json(json)
    		valueSeries=pd.Series(data["values"].iloc[5])
    		valueFrame = valueFrame.append(pd.DataFrame(valueSeries.values.tolist()))
    print("Writing to powerpriceData.csv")
    valueFrame.columns=["Date","Price"]
    valueFrame.to_csv('Data/powerpriceData.csv',index=False)

##############################################################################################################
###############################################Wetterhistory##################################################
##############################################################################################################
def updateWeatherHistory(parameter=["air_temperature","cloudiness","sun","wind"],shortform=["TT_TU"," V_N","SD_SO","   F"],times=["recent","historical"],
                            start='1/1/2016', end='11/12/2019'):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d%H')
    i=0
    finalFrame=pd.DataFrame(pd.date_range(start=start, end=end,freq ="H"),columns=["MESS_DATUM"])
    finalFrame.set_index("MESS_DATUM",inplace=True)
    testFrame=pd.DataFrame()
    for param in parameter:
        timesCombinedFrame=pd.DataFrame(columns=["MESS_DATUM",shortform[i]])
        timesCombinedFrame.set_index("MESS_DATUM",inplace=True)
        for timeMode in times:
            df=pd.DataFrame(pd.date_range(start=start, end=end,freq ="H"),columns=["MESS_DATUM"])
            df.set_index("MESS_DATUM",inplace=True)
            print("")
            _URL="https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/{}/{}/".format(param,timeMode)
            r = urlopen(_URL)
            soup = bs(r.read(),features="html.parser")
            links=soup.findAll('a')
            maximum=len(links)
            for j, link in enumerate(links[1:10]) :
                print("\r{} {}:{}/{}".format(param,timeMode,j,maximum), sep=' ', end='', flush=True)
                if link.get('href').endswith('.zip'):
                    _FULLURL = _URL + link.get('href')
                    resp=urlopen(_FULLURL)
                    zipfile = ZipFile(BytesIO(resp.read()))
                    file=zipfile.namelist()[-1]
                    tempdf=pd.read_csv(zipfile.open(file),sep=';').set_index("MESS_DATUM")
                    tempdf.index = pd.to_datetime(tempdf.index,format='%Y%m%d%H')
                    df[shortform[i]]=pd.concat([df, tempdf[shortform[i]]], axis=1).mean(axis=1)
            df.dropna(inplace=True)
            if timeMode=="recent":
                df=df.loc[df.index>start]
                recentFirstDate=df.index[0]
            if timeMode=="historical":
                mask = (df.index > start) & (df.index <= recentFirstDate)
                df=df.loc[mask]
                df=df.loc[df.index>start]
            df.to_csv("Data/{}_{}.csv".format(param,timeMode))

        # finalFrame = finalFrame.join(timesCombinedFrame,how="left")
        
        i+=1
    # finalFrame = finalFrame.loc[finalFrame.index.drop_duplicates()]
    # finalFrame.index = pd.to_datetime(finalFrame.index,format='%Y%m%d%H')
    # print("Writing to weatherHistory.csv")
    # finalFrame.to_csv("Data/weatherHistory2.csv")
##############################################################################################################
###############################################Vorhersage#####################################################
##############################################################################################################
def updateForecast(properties=["FF","N","SunD1","TTT"],WriteXML=False, updateGermanCities=False,):
    print("downloading Forecast")
    resp=urllib.request.urlopen("https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz")
    print("downloaded")
    kmz = ZipFile(BytesIO(resp.read()), 'r')
    kml = kmz.open(kmz.namelist()[0], 'r').read()
    root = ET.fromstring(kml)

    namespace={"kml":"http://www.opengis.net/kml/2.2",
                "dwd":"https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd"}
    cities=pd.read_csv("Data/germanCities.csv")
    cities=cities["Ort"].astype(str).values


    print("reading XML")
    lastFrame=pd.DataFrame()

    indexList=[]
    for TimeStep in root.iter('{https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd}TimeStep'):
        indexList.append(TimeStep.text)
    for prop in properties:
        for Placemark in root.iter('{http://www.opengis.net/kml/2.2}Placemark'):
            city=Placemark.find("kml:description",namespace).text
            if city in cities:#stadt in DE
                df=pd.DataFrame()
                forecast=Placemark.find("./kml:ExtendedData/dwd:Forecast[@dwd:elementName='{}']".format(prop),namespace)
                df[city]=list(map(float,forecast[0].text.replace("-","-999").split()))
                lastFrame[prop]=df.mean(axis=1)
    lastFrame["Date"]=indexList
    lastFrame["Date"]=pd.to_datetime(lastFrame['Date'])
    lastFrame.set_index("Date",inplace=True)
    lastFrame.columns=["Windgeschwindigkeit","Bewölkung", "Sonnensekunden","Temperatur 2m"]
    print("writing to forecast.csv")
    lastFrame.to_csv("Data/forecast.csv")

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
