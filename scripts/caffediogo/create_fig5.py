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
    if len(im.shape) == 3:
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im[0,0] = 0
    im[0,1] = 255
    imt = im.astype(np.float32)/scale # convert to float
    
    imt = Image.fromarray(np.uint8(255.0*cm.viridis(imt)),'RGBA')    
    return imt
 
def do_merge(mono_path,stereo_path,gt_path,rgb_path,method):
    perf_result, im_merged,im_fusionconf,im_errormap,bla = performance.merge_depth_maps(mono_path,stereo_path,gt_path,rgb_path,graphics=False,verbose=False, method=method, non_occluded=True) 
    im_merged = cv2.resize(im_merged,(im_rgb.shape[1], im_rgb.shape[0]), interpolation = cv2.INTER_CUBIC)
    im_merged -=np.min(im_merged)
    return im_merged,im_fusionconf,im_errormap

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

print("\tgt [m],\tstereo [m],\tmancini [m],\tsperzi [m]")


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
    im_rgb = cv2.cvtColor(im_rgb,cv2.COLOR_BGR2RGBA)
    im_stereo = cv2.imread(stereo_path)
    im_conf = cv2.imread(conf_path)
    im_gt = cv2.imread(gt_path)
    im_sperzi = cv2.imread(sperzi_path)
    #im_mix_fcn = cv2.imread(mix_fcn_path)
    im_mancini = cv2.imread(mancini_path)
    #im_merged_sperzi = cv2.imread(merged_sperzi_path)
    #im_merged_mix_fcn = cv2.imread(merged_mix_fcn_path)    
    #im_merged_mancini = cv2.imread(merged_mancini_path)

    im_stereo = evaluation_utils.convert_disps_to_depths_kitti(im_stereo,target_width = 1242, target_height = 375, mask = True, limit_depth = 80.0)
    im_mancini = evaluation_utils.convert_disps_to_depths_kitti(im_mancini,target_width = 1242, target_height = 375, mask = False, limit_depth = 80.0)
    im_sperzi = evaluation_utils.convert_disps_to_depths_kitti(im_sperzi,target_width = 1242, target_height = 375, mask = False, limit_depth = 80.0)
     
    im_merged_mancini,im_fusionconf_mancini,im_mancini_errormap = do_merge(mancini_path,stereo_path,gt_path,rgb_path,"mancini")
    im_merged_sperzi,im_fusionconf_sperzi,im_sperzi_errormap = do_merge(sperzi_path,stereo_path,gt_path,rgb_path,"sperzi")

    #im_fusionconf_sperzi = cv2.imread(fusionconf_sperzi_path)
    #im_fusionconf_mix_fcn = cv2.imread(fusionconf_mix_fcn_path)
    #im_fusionconf_mancini = cv2.imread(fusionconf_mancini_path)

    #im_fusionconf_mancini = cv2.resize(im_fusionconf_mancini,(im_rgb.shape[1], im_rgb.shape[0]), interpolation = cv2.INTER_CUBIC)
    #print np.min(im_gt) , np.max(im_gt),np.min(im_stereo), np.max(im_stereo),np.min(im_mancini), np.max(im_mancini),np.min(im_merged_mancini), np.max(im_merged_mancini)


    print ("Min:\t" + str(np.min(im_gt)) + "\t\t" +  str(np.min(im_stereo)) + "\t\t" +  str(np.min(im_mancini)) + "\t\t" +  str(np.min(im_sperzi))  ) #+ "   " +  str(np.min(im_merged_mancini))
    print ("Max:\t" + str(np.max(im_gt)) + "\t\t" +  str(np.max(im_stereo)) + "\t\t" +  str(np.max(im_mancini)) + "\t\t" +  str(np.max(im_sperzi))  )
    print "--------------------------------------------"
    #print rgb_path
    im_stereo = colorize(im_stereo,80.0)
    im_gt = colorize(im_gt,80.0)
    im_sperzi = colorize(im_sperzi,80.0)
    #im_mix_fcn = colorize(im_mix_fcn,80.0)
    im_mancini = colorize(im_mancini,80.0)
    #im_merged_sperzi = colorize(im_merged_sperzi,255.0)
    #im_merged_mix_fcn = colorize(im_merged_mix_fcn,255.0)
    #pdb.set_trace()
    im_merged_mancini = colorize(im_merged_mancini,80.0)

    # ii = i
    # if i==0:
    #     fig, axes = plt.subplots(nrows=7, ncols=len(fileids)/2);
    # elif i ==4:
    #     fig, axes = plt.subplots(nrows=7, ncols=len(fileids)/2);

    # if i > 3:
    #     ii = i-4

    # cf = axes[0,ii].imshow(im_rgb)   
    # cf = axes[1,ii].imshow(im_stereo)
    # cf = axes[2,ii].imshow(im_mancini)
    # cf = axes[3,ii].imshow(im_merged_mancini)
    # cf = axes[4,ii].imshow(im_fusionconf_mancini)
    # cf = axes[5,ii].imshow(im_mancini_errormap)
    # cf = axes[6,ii].imshow(im_gt)

    # axes[0,ii].axis('off')
    # axes[1,ii].axis('off')
    # axes[2,ii].axis('off')
    # axes[3,ii].axis('off')
    # axes[4,ii].axis('off')
    # axes[5,ii].axis('off')
    # axes[6,ii].axis('off')

    border  = Image.new('RGBA', (1242, 30))

    im_mancini = im_mancini.resize((1242,375), Image.ANTIALIAS)

    colm = np.concatenate((im_rgb,border), axis=0)
    colm = np.concatenate((colm, np.array(im_stereo)), axis=0)
    colm = np.concatenate((colm, border), axis=0)
    colm = np.concatenate((colm, np.array(im_mancini)), axis=0)
    colm = np.concatenate((colm, border), axis=0)
    colm = np.concatenate((colm, np.array(im_merged_mancini)), axis=0)
    colm = np.concatenate((colm, border), axis=0)
    colm = np.concatenate((colm, np.array(im_fusionconf_mancini)), axis=0)
    colm = np.concatenate((colm, border), axis=0)
    colm = np.concatenate((colm, np.array(im_mancini_errormap)), axis=0)
    colm = np.concatenate((colm, border), axis=0)
    colm = np.concatenate((colm, np.array(im_gt)), axis=0)

    if i ==0:
        total = colm
    elif i ==4:
        total1 = total
        total = colm
    else:
        # pdb.set_trace()
        border  = Image.new('RGBA', (30, 7*375+6*30))
        total = np.concatenate((total, border), axis=1)  
        total = np.concatenate((total, colm), axis=1)        




Image.fromarray(total1).save('~/hv/Cloud/Google/Guido/fig5a.png')
Image.fromarray(total).save('~/hv/Cloud/Google/Guido/fig5b.png')
plt.figure()
plt.imshow(total1)
plt.axis('off')
plt.figure()
plt.imshow(total)
plt.axis('off')
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