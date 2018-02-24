#!/usr/bin/env python2
 

import glob
import numpy as np
import cv2
import pdb
import os
import sys
import subprocess
import performance
import argparse
from matplotlib.pylab import cm
import pickle
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import merge
import evaluation_utils
import copy
import pdb
from PIL import Image

def colorize(im, scale):
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imt = im.astype(np.float32)/255.0 # convert to float


    imt = Image.fromarray(np.uint8(scale*cm.viridis(imt)),'RGBA')    
    return imt

def applyPlotStyle(plt,title):
    plt.title(title)
    plt.axis('off')
  
    

fileids = []
fileids.append('0000000008')
fileids.append('0000000043')
fileids.append('0000000080')
fileids.append('0000000039')
fileids.append('0000000057')
fileids.append('0000000044')
fileids.append('0000000056')
fileids.append('0000000014')

syncids = []
syncids.append('0005')
syncids.append('0013')
syncids.append('0020')
syncids.append('0023')
syncids.append('0036')
syncids.append('0079')
syncids.append('0095')
syncids.append('0113')

fig, axes = plt.subplots(nrows=len(fileids), ncols=6);
for i in np.arange(0, len(fileids)):
    filename = fileids[i]
    dirdate_name = '2011_09_26_drive_' + syncids[i] + '_sync'
    dir_name = '/data/kevin/kitti/raw_data/2011_09_26/' + dirdate_name
    rgb_path  = dir_name + '/image_02/data/' + filename + '.png'
    stereo_path  = dir_name + '/disp/' + filename + '_disparity.png'
    conf_path    = dir_name + '/conf/' + filename + '_conf.png'
    gt_path      = dir_name + "/../../../data_depth_annotated/val/" + dirdate_name + "/proj_depth/groundtruth/image_02/" + filename + ".png"        
    mix_fcn_path = dir_name  + "/mix_fcn/" + filename + "_mix_fcn.png"
    sperzi_path  = dir_name + "/sperzi/" + filename + "_sperziboon.png"
    mancini_path = dir_name  + "/mancini/" + filename + "_mancini.png";
    merged_sperzi_path = dir_name + "/merged/" + filename + "_merged_" + "sperzi" + ".png"
    merged_mix_fcn_path = dir_name + "/merged/" + filename + "_merged_" + "mix_fcn" + ".png"
    merged_mancini_path = dir_name + "/merged/" + filename + "_merged_" + "manchini" + ".png"
    fusionconf_sperzi_path = dir_name + "/fusionconf/" + filename + "_fusionconf_" + "sperzi" + ".png"
    fusionconf_mix_fcn_path = dir_name + "/fusionconf/" + filename + "_fusionconf_" + "mix_fcn" + ".png"
    fusionconf_mancini_path = dir_name + "/fusionconf/" + filename + "_fusionconf_" + "manchini" + ".png"

    im_rgb = cv2.imread(rgb_path)
    im_stereo = cv2.imread(stereo_path)
    im_conf = cv2.imread(conf_path)
    im_gt = cv2.imread(gt_path)
    im_sperzi = cv2.imread(sperzi_path)
    im_mix_fcn = cv2.imread(mix_fcn_path)
    im_mancini = cv2.imread(mancini_path)
    im_merged_sperzi = cv2.imread(merged_sperzi_path)
    #im_merged_mix_fcn = cv2.imread(merged_mix_fcn_path)
    im_merged_mancini = cv2.imread(merged_mancini_path)
    
    im_fusionconf_sperzi = cv2.imread(fusionconf_sperzi_path)
    #im_fusionconf_mix_fcn = cv2.imread(fusionconf_mix_fcn_path)
    im_fusionconf_mancini = cv2.imread(fusionconf_mancini_path)
    print merged_mancini_path
    
    im_stereo = colorize(im_stereo,255.0)
    im_gt = colorize(im_gt,255.0)
    im_sperzi = colorize(im_sperzi,255.0)
    im_mix_fcn = colorize(im_mix_fcn,255.0)
    im_mancini = colorize(im_mancini,255.0)
    im_merged_sperzi = colorize(im_merged_sperzi,255.0)
    #im_merged_mix_fcn = colorize(im_merged_mix_fcn,255.0)
    im_merged_mancini = colorize(im_merged_mancini,255.0)
    

    
    
    cf = axes[i,0].imshow(im_rgb)   
    cf = axes[i,1].imshow(im_stereo)
    cf = axes[i,2].imshow(im_mancini)
    cf = axes[i,3].imshow(im_merged_mancini)
    cf = axes[i,4].imshow(im_fusionconf_mancini)
    cf = axes[i,5].imshow(im_gt)


    axes[i,0].axis('off')
    axes[i,1].axis('off')
    axes[i,2].axis('off')
    axes[i,3].axis('off')
    axes[i,4].axis('off')
    axes[i,5].axis('off')
    

plt.show()


# im = cv2.imread(im_rgb_path)




    

def createOverlay(im_background,im_overlay):
    if im_background.shape[0] != im_overlay.shape[0] or im_background.shape[1] != im_overlay.shape[1]:
        im_overlay = cv2.resize(im_overlay,(im_background.shape[1], im_background.shape[0]), interpolation = cv2.INTER_CUBIC)
    imt = im_overlay.astype(np.float32) # convert to float
    imt = Image.fromarray(np.uint8(255*cm.RdBu(imt)))
    imt2 = Image.fromarray(cv2.cvtColor(im_background,cv2.COLOR_GRAY2RGBA))
    imt = Image.blend(imt2, imt, 0.5)
    return imt