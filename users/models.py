from django.db import models
from django.contrib.auth.models import User

import datetime
# Create your models here.


class Attendance(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date = models.DateField(default=datetime.date.today)
	present=models.BooleanField(default=False)
	time_in=models.DateTimeField(null=True,blank=True)
	time_out=models.DateTimeField(null=True,blank=True)
	#default=datetime.datetime.now or timezone.now or datetime.time.now for time
	#timezone.now  or datetime.date.today for date


'''
# Create datetime objects for each time (a and b)
dateTimeA = datetime.datetime.combine(datetime.date.today(), a)
dateTimeB = datetime.datetime.combine(datetime.date.today(), b)
# Get the difference between datetimes (as timedelta)
dateTimeDifference = dateTimeA - dateTimeB
# Divide difference in seconds by number of seconds in hour (3600)  
dateTimeDifferenceInHours = dateTimeDifference.total_seconds() / 3600
'''


'''at the time of marking attendance- put datetime.time(datetime.datetime.now)'''




'''
date-> datetime.date.today
time->datetime.datetime.now
'''