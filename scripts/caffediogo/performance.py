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
import matplotlib
import merge
import evaluation_utils

MAX_DISP = 64.0;

def print_performance(performance_matrix, name='Performance'):
    print('%s' % name);
    prefix = ['Stereo', 'Mono:', 'Fusion']
    print('\t\tabs_rel, sq_rel, rmse, rmse_log, a1, a2, a3');
    for r in range(3):
        print(prefix[r] + ':\t %f, %f, %f, %f, %f, %f, %f' % tuple(performance_matrix[r, :]));

def merge_depth_maps(mono_name = "/home/guido/cnn_depth_tensorflow/tmp/00002.png", 
                     stereo_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_disparity.png",
                     GT_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_GT.png",
                     image_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_org.png",
                     graphics = True, verbose = True, method='sperzi'):

    #if(graphics):
    image = cv2.imread(image_name);
    if(len(image.shape) == 3 and image.shape[2] > 1):
        gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);    
    
    mono = cv2.imread(mono_name);
    if(len(mono.shape) == 3 and mono.shape[2] > 1):
        mono = cv2.cvtColor(mono, cv2.COLOR_RGB2GRAY);
    mono = mono.astype(float);
    if(method == 'sperzi'):
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
        
        
    
    if(graphics and False):
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
    depth_fusion, stereo_confidence = merge.merge_Diogo(depth_stereo, depth_mono, image, graphics = False);
    
    if(graphics):
        fig, ax = plt.subplots()
        ax.imshow(gray_scale, cmap='gray');
        ax.imshow(1.0 - stereo_confidence, norm = matplotlib.colors.Normalize(vmin= 0.0,vmax=1.0), cmap=plt.cm.RdBu, alpha=0.5);    
        PCM=ax.get_children()[3] #get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(PCM, ax=ax)
        #cb.set_lim(-MAX_DEPTH, MAX_DEPTH);
        plt.title('Mono confidence');
        
        
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
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, error_map_stereo = evaluation_utils.compute_errors(depth_GT[:], depth_stereo[:], name_fig = 'stereo error map');
    performance[0,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, error_map_mono = evaluation_utils.compute_errors(depth_GT[:], depth_mono[:], name_fig = 'mono error map');
    performance[1,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, error_map_fusion = evaluation_utils.compute_errors(depth_GT[:], depth_fusion[:], name_fig = 'fusion error map');
    performance[2,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];
    
    if(graphics):
        error_comparison = error_map_stereo - error_map_mono;
        #plt.figure();
        MAX_DEPTH = 40;
        fig, ax = plt.subplots()
        ax.imshow(gray_scale, cmap='gray');
        ax.imshow(error_comparison, norm = matplotlib.colors.Normalize(vmin= -MAX_DEPTH,vmax=MAX_DEPTH), cmap=plt.cm.RdBu, alpha=0.5);    
        PCM=ax.get_children()[3] #get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(PCM, ax=ax)
        #cb.set_lim(-MAX_DEPTH, MAX_DEPTH);
        plt.title('abs error stereo - abs error mono');
        
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
#                     graphics = True, verbose = True)#, method = 'mancini')
