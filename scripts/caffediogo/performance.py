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

def print_performance(performance_matrix, name='Performance'):
    print('%s' % name);
    prefix = ['Stereo', 'Mono:', 'Fusion']
    for r in range(3):
        print(prefix[r] + ': %f, %f, %f, %f, %f, %f, %f' % tuple(performance_matrix[r, :]));

def merge_depth_maps(mono_name = "/home/guido/cnn_depth_tensorflow/tmp/00002.png", 
                     stereo_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_disparity.png",
                     GT_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_GT.png",
                     image_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_org.png",
                     graphics = False, verbose = True):

    #if(graphics):
    image = cv2.imread(image_name);
    
    mono = cv2.imread(mono_name);
    mono = cv2.cvtColor(mono, cv2.COLOR_RGB2GRAY);
    mono = mono.astype(float);
    mono /= 255.0 / MAX_DISP; # TODO: is this correct for mancini?
    # mono = cv2.resize(mono, (64, 20), interpolation=cv2.INTER_NEAREST);

    stereo = cv2.imread(stereo_name);
    if(len(stereo.shape) == 3 and stereo.shape[2] > 1):
        stereo = cv2.cvtColor(stereo, cv2.COLOR_RGB2GRAY);

    GT = cv2.imread(GT_name);
    if(len(GT.shape) == 3 and GT.shape[2] > 1):
        GT = cv2.cvtColor(GT, cv2.COLOR_RGB2GRAY);
    kernel_dilate = np.ones([5,5]);
    GT = cv2.dilate(GT, kernel_dilate);
    
    if(mono.shape[0] != stereo.shape[0] or mono.shape[0] != GT.shape[0]):
        H = np.asarray([mono.shape[0]] * 3);
        W = np.asarray([mono.shape[1]] * 3);
        H[1] = stereo.shape[0];
        W[1] = stereo.shape[1];
        H[2] = GT.shape[0];
        W[2] = GT.shape[1];
        TARGET_H = np.min(H);
        TARGET_W = np.min(W);
        if(mono.shape[0] != TARGET_H):
            mono = cv2.resize(mono, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST);
        if(stereo.shape[0] != TARGET_H):
            stereo = cv2.resize(stereo, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST);
        if(GT.shape[0] != TARGET_H):
            GT = cv2.resize(GT, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST);        
        
        
    
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
    depth_GT = GT;
    # depth_GT = evaluation_utils.convert_disps_to_depths_kitti(GT);
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
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluation_utils.compute_errors(depth_GT[:], depth_stereo[:], name_fig = 'stereo error map');
    performance[0,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluation_utils.compute_errors(depth_GT[:], depth_mono[:], name_fig = 'mono error map');
    performance[1,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluation_utils.compute_errors(depth_GT[:], depth_fusion[:], name_fig = 'fusion error map');
    performance[2,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];
    
    if(verbose):
        print_performance(performance);

    # scaled mono
#    depth_mono = merge.scale_mono_map(depth_stereo, depth_mono);    
#    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluation_utils.compute_errors(depth_GT[:], depth_mono[:]);
#    performance[1,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];    
#    
#    if(verbose):
#        print('Scaled mono:')
#        print_performance(performance);
#        
        
    return performance, depth_fusion;

# merge_depth_maps(graphics=True);
#merge_depth_maps(mono_name = "./tmp/0000000013_sperziboon.png", 
#                     stereo_name = "./tmp/0000000013_disparity.png",
#                     GT_name = "./tmp/0000000013_GT.png",
#                     image_name = "./tmp/0000000013_image.png",
#                     graphics = False, verbose = True)