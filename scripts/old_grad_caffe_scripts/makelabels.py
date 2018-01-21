'''
Created on Jul 5, 2014

@author: houjebek
'''

def run():
    #import os
    #import glob
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Creates labeled data for caffe.')
    parser.add_argument("datatxtPath")
    parser.add_argument("dataPath")
    parser.add_argument("batchid")
    parser.add_argument("targetpath")
    args = parser.parse_args()
    
    datatxt = args.datatxtPath
    target = args.targetpath
    path = args.dataPath
    kk_id = int(args.batchid)
    
    fp = open(datatxt)
    entries=[]
    for i, line in enumerate(fp):    
        entries.append(line)
    fp.close()
    
    
    tot = len(entries) / 2
    
    
    partitions = np.round(np.linspace(0,tot,5, 'int'))
    
    print ("Writing entries")    
    
    fp_trn = open( target + "train" + ".txt", 'w')
    fp_tst = open( target + "test" + ".txt", 'w')
    for kk in range(1,5):     
        if kk ==kk_id:        
            for i in range(int(partitions[kk-1]),int(partitions[kk])):                          
                fp_tst.write(path + adaptlabel(entries[2*i]))  #left
                fp_tst.write(path + adaptlabel(entries[2*i+1])) #right
        else:
            for i in range(int(partitions[kk-1]),int(partitions[kk])):       
                fp_trn.write(path + adaptlabel(entries[2*i]))  #left
                fp_trn.write(path + adaptlabel(entries[2*i+1])) #right
                
        
    fp_trn.close()
    fp_tst.close()
    
    print ("done")
    
    


def adaptlabel(entry): 
    bla = entry.split(" ");
    f = bla[0]
    label = int(bla[1])
    
    
    return f + " " + str(label) + "\n"
    
    
   
   
run() 
