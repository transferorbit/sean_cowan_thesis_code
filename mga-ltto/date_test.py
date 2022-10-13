import tudatpy
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel import constants
julian_day = constants.JULIAN_DAY
print(julian_day)

# jd = 8000# * julian_day
#
# print(jd)
# jd_to_mjd = 2400000.5
# mjd = jd - jd_to_mjd
# print(mjd)

print(time_conversion.julian_day_to_calendar_date(time_conversion.modified_julian_day_to_julian_day(59000)))
print(time_conversion.julian_day_to_seconds_since_epoch(time_conversion.modified_julian_day_to_julian_day(59000)))
