import pandas as pd
import matplotlib.pyplot as plt
import dataCsvReader as dr
from statsmodels.tsa.seasonal import seasonal_decompose
import os
os.system('cls')
powerPrice=dr.getData()["Price"]
# powerPrice = powerPrice[~powerPrice.index.duplicated()]
result = seasonal_decompose(powerPrice, model='additive',freq=24*7)

result.plot()
plt.show()