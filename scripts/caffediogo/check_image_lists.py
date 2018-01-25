import glob
import numpy as np
import cv2
import pdb
import os

def diff_letters(a,b):
    return sum ( a[i] != b[i] for i in range(len(a)) )

fname_left = "val_images2.txt"; #"train_images2.txt"; 
fname_right = "val_images3.txt"; #"train_images3.txt";

with open(fname_left) as f:
  images_left = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  images_left = [x.strip() for x in images_left] 

with open(fname_right) as f:
  images_right = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  images_right = [x.strip() for x in images_right] 


for idx in np.arange(0, len(images_left), 1):
  num_diff = diff_letters(images_left[idx], images_right[idx]);
  if(num_diff != 1):
    print('Element in the list: {}. Names: {} <=> {}'.format(idx, images_left[idx], images_right[idx]));
  else:
    print('.');




