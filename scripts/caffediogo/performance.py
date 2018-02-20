# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:47:54 2018

Scripts that determines the performance of the various algorithms.

@author: guido
"""

# type:
# %matplotlib qt 
# when you want windows instead of inline plots

import cv2
import numpy as np
import matplotlib.pyplot as plt
import merge
import evaluation_utils

MAX_DISP = 64.0;

def merge_depth_maps(mono_name = "/home/guido/cnn_depth_tensorflow/tmp/00002.png", 
                     stereo_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_disparity.png",
                     GT_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_GT.png",
                     graphics = False, image_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_org.png"):

    if(graphics):
        image = cv2.imread(image_name);
    
    mono = cv2.imread(mono_name);
    mono = cv2.cvtColor(mono, cv2.COLOR_RGB2GRAY);
    mono = mono.astype(float);
    mono /= 255.0 / MAX_DISP;

    stereo = cv2.imread(stereo_name);
    stereo = cv2.cvtColor(stereo, cv2.COLOR_RGB2GRAY);

    GT = cv2.imread(GT_name);
    GT = cv2.cvtColor(GT, cv2.COLOR_RGB2GRAY);
    
    if(graphics):
        fig, axes = plt.subplots(nrows=2, ncols=2);
        cf = axes[0,0].imshow(mono);
        axes[0,0].set_title('Mono disp');
        fig.colorbar(cf, ax=axes[0,0])
        cf = axes[0,1].imshow(stereo);
        axes[0,1].set_title('Stereo disp');
        fig.colorbar(cf, ax=axes[0,1])
        cf = axes[1,0].imshow(GT);
        axes[1,0].set_title('GT disp');
        fig.colorbar(cf, ax=axes[1,0])
        axes[1,1].imshow(image);
        axes[1,1].set_title('Image');
    
    depth_stereo = evaluation_utils.convert_disps_to_depths_kitti(stereo);
    depth_mono = evaluation_utils.convert_disps_to_depths_kitti(mono);
    depth_GT = evaluation_utils.convert_disps_to_depths_kitti(GT);
    depth_fusion = merge.merge_Diogo(depth_stereo, depth_mono, image, graphics = False);
    
    if(graphics):
        fig, axes = plt.subplots(nrows=2, ncols=2);
        cf = axes[0,0].imshow(depth_mono);
        axes[0,0].set_title('Mono depth');
        fig.colorbar(cf, ax=axes[0,0])
        cf = axes[0,1].imshow(depth_stereo);
        axes[0,1].set_title('Stereo depth');
        fig.colorbar(cf, ax=axes[0,1])
        cf = axes[1,0].imshow(depth_GT);
        axes[1,0].set_title('GT depth');
        fig.colorbar(cf, ax=axes[1,0])
        axes[1,1].imshow(depth_fusion);
        axes[1,1].set_title('Fusion depth');
        fig.colorbar(cf, ax=axes[1,1])
    
    performance = np.zeros([3, 7]);
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluation_utils.compute_errors(depth_GT[:], depth_stereo[:]);
    performance[0,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];
    print('Performance:\tabs_rel,sq_rel,rmse,rmse_log,a1,a2,a3.');
    print('Stereo:\t{},{},{},{},{},{},{}.'.format(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3));
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluation_utils.compute_errors(depth_GT[:], depth_mono[:]);
    performance[1,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];
    print('Mono:\t{},{},{},{},{},{},{}.'.format(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3));
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluation_utils.compute_errors(depth_GT[:], depth_fusion[:]);
    print('Fusion:\t{},{},{},{},{},{},{}.'.format(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3));
    performance[2,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];
    
    return performance, depth_fusion;

#merge_depth_maps(graphics=True);