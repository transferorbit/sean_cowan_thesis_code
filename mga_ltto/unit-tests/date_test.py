import tudatpy
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel import constants
import datetime

julian_day = constants.JULIAN_DAY
calendar_date = datetime.datetime(2022, 5, 21, 13, 52, 41)
# jd = 8000# * julian_day
#
# print(jd)
# jd_to_mjd = 2400000.5
# mjd = jd - jd_to_mjd
# print(mjd)
print('Please enter conversion type [date-to-mjd/mjd-to-date]')
option = input()


if option == 'mjd-to-date':
    print('Please enter a number in [MJD2000] : ')
    number = float(input()) #MJD2000
    print(time_conversion.julian_day_to_calendar_date(
        time_conversion.modified_julian_day_to_julian_day(number + 51544.5)))
elif option == 'date-to-mjd':
    print('Please enter a date [year, month, day, hour, minute, second]: ')
    date = input().split(',')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    hour = int(date[3])
    minute = int(date[4])
    second = int(date[5])
    date_formatted = datetime.datetime(year, month, day, hour, minute, second)
    print(time_conversion.calendar_date_to_julian_day_since_epoch(date_formatted))
else:
    print('Something went wrong')


