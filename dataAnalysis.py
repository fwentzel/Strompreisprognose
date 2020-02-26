import matplotlib.pyplot as plt
import csvReader as dr
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import os

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

import csvReader as dr

os.system('cls')
powerPrice=dr.get_data()["Price"]
# powerPrice = powerPrice[~powerPrice.index.duplicated()]
result = seasonal_decompose(powerPrice, model='additive',freq=24*7)

result.plot()
plt.show()