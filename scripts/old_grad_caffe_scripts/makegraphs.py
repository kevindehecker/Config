#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Creates graphs from caffe log csv data')
parser.add_argument("folder")
args = parser.parse_args()

folder = args.folder
print(folder)

testscores = np.array([[]])
trainlosses = np.array([[]])
for x in range(0, 4):
	print(x)
	testscore = np.genfromtxt(os.path.join(folder , str(x+1)  +  "/snapshots/logdata_testscores.csv"), delimiter=";")
	testscore=testscore[:,0]
	testscores= np.append(testscores,[testscore],axis=1)
	
	trainloss = np.genfromtxt(os.path.join(folder , str(x+1)  +  "/snapshots/logdata_trainlosses.csv"), delimiter=";")	
	trainlosses= np.append(trainlosses,[trainloss],axis=1)
	
	plt.subplot(211)
	plt.plot(testscore)	
	plt.subplot(212)
	plt.plot(trainloss) 
	
plt.subplot(211)
plt.title("Test scores")
plt.subplot(212)
plt.title("Test scores")

	
	
plt.show()