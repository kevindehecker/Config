#!/usr/bin/env python2

#this daemon script logs usage statistics and starts a coin miner if gpu's are not used
#made by Kevin

from __future__ import print_function
import psutil
import time
import os
from subprocess import Popen
import sys
import signal
import subprocess

MINER_PROC_NAME = "/home/kevin/ccminer/ccminer -a x17 -o stratum+tcp://yiimp.eu:3737 -u D9uarTgV9p7cywXfv8eMjnxtqxbCMMDbrq -d"
MINER_PROC_NAME_SHORT = "/home/kevin/ccminer/ccminer"
INTERVAL = 60
NGPUS = 2


users = []
gpu_users = [0] * NGPUS

mining_pids = [psutil.Process] * NGPUS
mining_started = [False] * NGPUS

gpu_was_used = [False] * NGPUS

while True: # loops INTERVAL
	time_str =time.strftime("%I:%M:%S") + ' ' + time.strftime("%d/%m/%Y")
	proc = subprocess.Popen(['nvidia-smi'],stdout=subprocess.PIPE) 
	#proc = subprocess.Popen(['python', 'fake_utility.py'],stdout=subprocess.PIPE) #TMP

	state = 0
	gpu_mem_usages = [0] * NGPUS
	gpu_is_used = [False] * NGPUS
	gpu_is_mining = [False] * NGPUS

	for i in range (0,NGPUS):
		gpu_mem_usages[i] = []
		gpu_users[i] = []

	gpu_overview_id = 0
	while True: # read SMI output
		line = proc.stdout.readline()
		if line != '':

			if '%' in line:
				#log this line to appropiate file, contains overall GPU usage statistics
				with open('gpu' + str(gpu_overview_id) + '.txt', 'a') as f:
					f.write(time_str + ' ' + line )
				gpu_overview_id = gpu_overview_id +1


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
				process_cpu = p.cpu_percent(interval=0.01)
				mem_text = ls[5]
				process_mem = mem_text[0:len(mem_text)-3] #currently assuming this is always in MiB
#				print (gpu_id,": ", process_name,process_user,process_cpu, process_mem)
				
				if process_user not in users:
					users.append(process_user)
				if process_user not in gpu_users[gpu_id]:
					gpu_users[gpu_id].append(process_user)

				if process_name != "/usr/lib/xorg/Xorg" and process_name != MINER_PROC_NAME_SHORT:
					gpu_is_used[gpu_id] = True
				elif process_name == MINER_PROC_NAME_SHORT and gpu_is_mining[gpu_id]:
					#multiple miner instances... Weird!?!? Kill it immidiately.
					print("Failure state: multiple miners detected on the same GPU" + str(gpu_id) + ". Attempting to kill pid: " + str(process_id))
					os.killpg(os.getpgid(process_id), signal.SIGTERM)
				elif process_name == MINER_PROC_NAME_SHORT:
					gpu_is_mining[gpu_id] = True
					if mining_pids[gpu_id].pid != process_id:
						print("Unrecognized miner detected on gpu" + str(gpu_id) + ". Known PID: " + str(mining_pids[0].pid) + " & " + str(mining_pids[1].pid)  + ". Attempting to kill pid: " + str(process_id))
						os.killpg(os.getpgid(process_id), signal.SIGTERM)
		else: #end of smi output
			break

	print("GPU is used:   ", gpu_is_used, ", was used: ", gpu_was_used)
	print("GPU mining: ", gpu_is_mining)
	print("GPU users: ",  users)

	with open('general.txt', 'a') as f:
		f.write(time_str + ';' + str(gpu_is_used) + ';' + str(gpu_is_mining) + ';' + str(gpu_users) + '\n')

	for i in range(0,NGPUS):
		if not gpu_is_used[i]and not gpu_was_used[i] and not gpu_is_mining[i]:
			#start miner on this gpu			
			pro = Popen("exec " + MINER_PROC_NAME + str(i),shell=True,preexec_fn=os.setsid)
			mining_pids[i] = pro
			print(time_str + ": started miner " + str(pro.pid) + " on GPU", i)
			mining_started[i] = True
		elif gpu_is_used[i] and gpu_is_mining[i]:
			#stop miner on this gpu
			print(time_str + ": stopping miner " + str(mining_pids[i].pid) + " on GPU", i)
			try:
				os.killpg(os.getpgid(mining_pids[i].pid), signal.SIGTERM)
				mining_started[i] = False
			except:
				print(time_str + ": Failed to stop miner " + str(mining_pids[i].pid) + " on GPU", i + ". Killing all ccminer.")
				Popen("killall ccminer",shell=True,preexec_fn=os.setsid)
		else:
			print(time_str + ": nothing to do for GPU", i)


	gpu_was_used = gpu_is_used

	#log CPU usage:
	mem =psutil.virtual_memory()
	cpu = str(psutil.cpu_percent(INTERVAL))
	with open('cpu.txt', 'a') as f:
		f.write(time_str + ': ' + cpu + ', ' + str(mem.percent) + ', ' + str(mem.available/1024/1024) + '\n')
	print("CPU: ", cpu)
	#time.sleep(INTERVAL)
