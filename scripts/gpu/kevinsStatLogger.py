#!/usr/bin/env python2

from __future__ import print_function
import psutil
import time
import os
from subprocess import Popen
import sys
import signal
import subprocess

PROCNAME = "/home/kevin/ccminer/ccminer -a x17 -o stratum+tcp://yiimp.eu:3737 -u D9uarTgV9p7cywXfv8eMjnxtqxbCMMDbrq -d"
PROCNAME_SHORT = "/home/kevin/ccminer/ccminer"
INTERVAL = 60
NGPUS = 2


users = []

mining_pids = [psutil.Process] * NGPUS
mining_started = [False] * NGPUS

while True:
	proc = subprocess.Popen(['nvidia-smi'],stdout=subprocess.PIPE) #TODO: change to nvidia-smi util

	state = 0
	gpu_mem_usages = [0] * NGPUS
	gpu_is_used = [False] * NGPUS
	gpu_is_mining = [False] * NGPUS

	for i in range (0,NGPUS):
		gpu_mem_usages[i] = []

	while True:
		line = proc.stdout.readline()
		if line != '':    	    
			if line.startswith("+-----"): #start and end of the output, reset state
				state=0
			if state==0:
				if line.startswith("| Processes:"):
					state =1
			elif state==1:
				if line.startswith("|========"):
					state =2
			elif state==2:
				ls = line.split()
				
				gpu_id = int(ls[1])
				process_id = int(ls[2])
				p = psutil.Process(process_id)
				process_type = ls[3]
				process_name = ls[4]
				process_user = p.username()
				process_cpu = p.cpu_percent()
				mem_text = ls[5]
				process_mem = mem_text[0:len(mem_text)-3] #currently assuming it is always in MiB
				print (gpu_id,": ", process_name,process_user,process_cpu, process_mem)
				
				if process_user not in users:
					users.append(process_user)

				if process_name != "/usr/lib/xorg/Xorg" and process_name != PROCNAME_SHORT:
					gpu_is_used[gpu_id] = True
				if process_name == PROCNAME_SHORT:
					gpu_is_mining[gpu_id] = True
		else: #end of smi output
			break

	print("used: ", gpu_is_used)
	print("mining: ", gpu_is_mining)
	print(users)

	for i in range(0,NGPUS):
		if not gpu_is_used[i]and not gpu_is_mining[i]:
			#start miner on this gpu
			print("Starting miner on GPU", i)
			pro = Popen(PROCNAME + str(gpu_id),shell=True,preexec_fn=os.setsid)
			mining_pids[i] = pro
			mining_started[i] = True
		elif gpu_is_used[i] and gpu_is_mining[i]:
			#stop miner on this gpu
			print("Stopping miner on GPU", i)
			os.killpg(os.getpgid(mining_pids[i].pid), signal.SIGTERM)
			mining_started[i] = False
		else:
			print("Nothing to do for GPU", i)

	time.sleep(INTERVAL)
