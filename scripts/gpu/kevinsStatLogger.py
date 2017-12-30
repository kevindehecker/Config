#!/usr/bin/env python

from __future__ import print_function
import psutil
import time
import os
from subprocess import Popen
import sys
import signal

PROCNAME = "/home/kevin/claymore/ethdcrminer64 -epool eu1.ethermine.org:4444 -ewal 0x56A3c9144B5F23A6075EF95751c840F836c9a325.MV -epsw x"
INTERVAL = 10
NGPUS = 4

state = 0

gpu_mem_usages = [NGPUS]

for i in range (0:3)
	gpu_mem_usages[i] = []

#filters output
import subprocess
proc = subprocess.Popen(['python','fake_utility.py'],stdout=subprocess.PIPE)
while True:
  line = proc.stdout.readline()
  if line != '':
    #the real code does filtering here
    
		if line.startswith("+-----"): #start and end of the output, reset state
			state=0

		if state==0:
			if line.startswith("| Processes:"):
				state =1
		elif state==1:
			if line.startswith("|========"):
				state =2
		elif state==2:
			gpu_id = line[1:].strip()
			gpu_id = gpu_id[0:1]
			print ("gpu id:" , gpu_id)
			



  else:
    break

exit()

#started = False
#while True:
        #v = psutil.cpu_percent(INTERVAL)
        #print("Deamon loop " , str(v))
        #if v < 5 and not started:
                #started = True
                #print("Starting")
                #pro = Popen(PROCNAME,shell=True,preexec_fn=os.setsid)
                #print("STARTED")
        #elif v > 10 and started:
                #started = False
                #print("Stopping")
                #os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


        #time.sleep(INTERVAL)
