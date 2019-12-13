import holidays
from datetime import date
holidays = holidays.Germany()
print(date(2019, 12, 25) in holidays)