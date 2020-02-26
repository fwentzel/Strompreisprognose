from datetime import date

import holidays

holidays = holidays.Germany()
print(date(2019, 12, 25) in holidays)