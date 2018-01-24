# show generated disparity images
import glob
import numpy as np
import cv2
import pdb
import os

def show_disparity_images():

  ONLY_DISP = True;
  min_disp = 1;
  num_disp = 64;

  fname_left = "images2.txt"; #"/data/kevin/kitti/raw_data/2011_09_26/images3.txt";
  with open(fname_left) as f:
    images_left = f.readlines()
  
  # you may also want to remove whitespace characters like `\n` at the end of each line
  images_left = [x.strip() for x in images_left] 

  pdb.set_trace();

  # iterate over the files:
  for idx in np.arange(0, len(images_left), 10):

    print("{} / {}".format(idx, len(images_left)));

    im = images_left[idx];

    imgL = cv2.imread(im);

    base_name = os.path.basename(im);
    file_name, ext = os.path.splitext(base_name);
    dir_name = os.path.dirname(im);
    dir_name = os.path.dirname(dir_name);
    dir_name = os.path.dirname(dir_name);
    dir_name = dir_name;

    disp = cv2.imread(dir_name + "/disp/" + file_name + "_disparity.png");

    if(not ONLY_DISP):
      cv2.imshow('left', imgL);

    cv2.imshow('disparity', (disp.astype('float')-min_disp)/num_disp);
    		
    # wait for a key to be pressed before moving on:
    cv2.waitKey();
    cv2.destroyAllWindows();

    
show_disparity_images();
