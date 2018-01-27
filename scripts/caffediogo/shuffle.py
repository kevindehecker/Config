#!/usr/bin/env python2

import glob
import numpy as np
import cv2
import pdb
import os
import sys
import random


def shuffle_set(prefix):
 fname2_in =  sys.argv[1] + prefix + "_images2_labeled.txt"
 fname3_in =  sys.argv[1] + prefix + "_images3_labeled.txt"

 with open(fname2_in) as f:
  images2 = f.readlines()

 with open(fname3_in) as f:
  images3 = f.readlines()

 #pdb.set_trace()

 fname2_out =  sys.argv[1] + prefix + "_images2_shuffled.txt"
 fname3_out =  sys.argv[1] + prefix + "_images3_shuffled.txt"

 with open(fname2_out,'w') as f2, open(fname3_out,'w') as f3:

  while (len(images2) > 0):
   r = random.randint(0,len(images2)-1)
   print("{} --> {}".format(len(images2),r))
   #print(images2[r])
   f2.write(images2.pop(r))
   f3.write(images3.pop(r))


shuffle_set("val")
shuffle_set("train")
