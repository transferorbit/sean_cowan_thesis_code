'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: Unknown

This file implements some conversion functions, using tudatpy conversion functions and datetime.
'''

# General
import datetime

# Tudatpy
import tudatpy
from tudatpy.kernel.astro import time_conversion

class dateConversion:

    def __init__(self, mjd2000 = None, calendar_date = None) -> None:
        self.mjd2000 = mjd2000
        self.calendar_date = calendar_date

    def mjd_to_date(self):
        number = float(self.mjd2000)
        return time_conversion.julian_day_to_calendar_date(
                time_conversion.modified_julian_day_to_julian_day(number + 51544.5))

    def date_to_mjd(self):
        date = self.calendar_date.split(',')
        year = int(date[0])
        month = int(date[1])
        day = int(date[2])
        date_formatted = datetime.date(year, month, day)

        return time_conversion.calendar_date_to_julian_day_since_epoch(date_formatted)

