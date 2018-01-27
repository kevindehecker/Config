#!/usr/bin/env python2

import glob
import numpy as np
import cv2
import pdb
import os
import sys
import random

#shuffles the data set, but keeps the image pairs together (a pair consists out of: image 2, image 3, disp, conf)
#shuffles both the train and the val set.

def shuffle_set(prefix):
 fname2_in =  sys.argv[1] + prefix + "_images2_labeled.txt"
 fname3_in =  sys.argv[1] + prefix + "_images3_labeled.txt"
 fnamed_in =  sys.argv[1] + prefix + "_disp_labeled.txt"
 fnamec_in =  sys.argv[1] + prefix + "_conf_labeled.txt"

 with open(fname2_in) as f:
  images2 = f.readlines()
 with open(fname3_in) as f:
  images3 = f.readlines()
 with open(fnamed_in) as f:
  imagesd = f.readlines()
 with open(fnamec_in) as f:
  imagesc = f.readlines()

 fname2_out =  sys.argv[1] + prefix + "_images2_shuffled.txt"
 fname3_out =  sys.argv[1] + prefix + "_images3_shuffled.txt"
 fnamed_out =  sys.argv[1] + prefix + "_disp_shuffled.txt"
 fnamec_out =  sys.argv[1] + prefix + "_conf_shuffled.txt"



 with open(fname2_out,'w') as f2, open(fname3_out,'w') as f3, open(fnamed_out,'w') as fd, open(fnamec_out,'w') as fc:

  while (len(images2) > 0):
   r = random.randint(0,len(images2)-1)
   print("{} --> {}".format(len(images2),r))
   f2.write(images2.pop(r))
   f3.write(images3.pop(r))
   fd.write(imagesd.pop(r))
   fc.write(imagesc.pop(r))


shuffle_set("val")
shuffle_set("train")
