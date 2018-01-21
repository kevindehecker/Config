
import argparse

parser = argparse.ArgumentParser(description='Extracts data from caffe log.')
parser.add_argument("logfilename")
parser.add_argument("outputfile")
args = parser.parse_args()

logfilename = args.logfilename
outputfile = args.outputfile
print("Opening log file: " + logfilename)

T1 = list() #test score Accuracy
T2 = list() #test score Loss

Tr = list() # train loss

fp = open(logfilename)
for i, line in enumerate(fp):    
	if  "Test score #0" in line:
		s = line.split('#')[1].rstrip().split(':')[1]
		T1.append(s)
	#	print (s)
	elif "Test score #1" in line:
		s = line.split('#')[1].rstrip().split(':')[1]
		T2.append(s)
	elif "loss =" in line:
		s = line.split('=')[1].rstrip()
		Tr.append(s)		
fp.close()

print("Writing test set scores:" + outputfile + "_testScores.csv")
fp = open(outputfile + "_testscores.csv", 'w')
for j, line in enumerate(T1):
	line = T1[j] + '; ' + T2[j]
	#print(line)
	fp.write(line + '\n')
	
fp.close()

print("Writing training losses:" + outputfile + "_trainlosses.csv");
fp = open(outputfile + "_trainlosses.csv", 'w')
for j, line in enumerate(Tr):
	line = Tr[j] 
	#print(line)
	fp.write(line + '\n')
	
fp.close()
print("Done")
