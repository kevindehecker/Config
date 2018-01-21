import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser(description='Extracts data from caffe log.')
parser.add_argument("folder")
args = parser.parse_args()

plt.close("all")
plt.figure()

for x in range(0, 4):

	logfilename = os.path.join(args.folder , str(x+1)  +  "/snapshots/logfile.log")

	print("Reading log file: " + logfilename)

	TA = [] #test score Accuracy
	TL = [] #test score Loss

	TrL = [] # train loss

	fp = open(logfilename)
	last_iter= 0
	for i, line in enumerate(fp):    
		if "Iteration" in line:
			s = line.split(']')[1].rstrip().split(' ')[2].split(',')[0]			
			last_iter = int(s)		
		if  "Test score #0" in line: #accuracy
			s = line.split('#')[1].rstrip().split(':')[1]
			TA.append([last_iter, float(s)])		
		elif "Test score #1" in line: #loss
			s = line.split('#')[1].rstrip().split(':')[1]		
			TL.append([last_iter, float(s)])
		elif "loss =" in line:
			s = line.split('=')[1].rstrip()
			TrL.append([last_iter, float(s)])
					
	fp.close()
	TrLa = np.array(TrL)
	plt.subplot(3,1,1)
	plt.plot(TrLa[:,0],TrLa[:,1])

	TAa = np.array(TA)
	plt.subplot(3,1,2)
	plt.plot(TAa[:,0],TAa[:,1])

	TLa = np.array(TL)
	plt.subplot(3,1,3)
	plt.plot(TLa[:,0],TLa[:,1])

	
	
plt.subplot(3,1,1)
plt.xlabel("Iterations")
plt.ylabel("Training Loss")

plt.subplot(3,1,2)
plt.xlabel("Iterations")
plt.ylabel("Test Accuracy")

plt.subplot(3,1,3)
plt.xlabel("Iterations")
plt.ylabel("Test Loss")

plt.show()