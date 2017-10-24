#!/usr/bin/env python 

from __future__ import print_function
import psutil
import time
import os
from subprocess import Popen
import sys


PROCNAME = "/home/houjebek/tmp.py"
INTERVAL = 0.5

started = False
while True:
	print("Loop") 
	if psutil.cpu_percent(INTERVAL) > 5 and not started:	
		started = True	
		print("Starting") 
		pid = Popen([sys.executable, PROCNAME])	
	elif psutil.cpu_percent(INTERVAL) < 10 and started:
		started = False
		print("Stopping") 


		for proc in psutil.process_iter():
		    # check whether the process name matches
		    if proc.name() == PROCNAME:
		        proc.kill()
			time.sleep(INTERVAL)
