# -*- coding: utf-8 -*- 
"""
Created on Mon Feb 12 15:47:54 2018

Script that determines the performance of the various algorithms.

@author: guido
"""

# type:
# %matplotlib qt 
# when you want windows instead of inline plots

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
#from collections import namedtuple

#distance_versus_error_info = namedtuple("distance_versus_error_info", "err_mono gt_mono err_stereo gt_stereo err_fusion gt_fusion")

MAX_DISP = 64.0;

def print_performance(performance_matrix, name='Performance'):
    print('%s' % name);
    prefix = ['Stereo', 'Mono:', 'Fusion']
    print('\t\tabs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, lsi_err');
    for r in range(3):
        print(prefix[r] + ':\t %f, %f, %f, %f, %f, %f, %f, %f' % tuple(performance_matrix[r, :]));

def createOverlay(im_background,im_overlay, v_min=0.0, v_max=1.0):
    if im_background.shape[0] != im_overlay.shape[0] or im_background.shape[1] != im_overlay.shape[1]:
        im_overlay = cv2.resize(im_overlay,(im_background.shape[1], im_background.shape[0]), interpolation = cv2.INTER_CUBIC)
    imt = im_overlay.astype(np.float32) # convert to float
    norm = matplotlib.colors.Normalize(vmin=v_min,vmax=v_max)
    imt = Image.fromarray(np.uint8(255*cm.RdBu(norm(imt))))
    imt2 = Image.fromarray(cv2.cvtColor(im_background,cv2.COLOR_GRAY2RGBA))
    imt = Image.blend(imt2, imt, 0.5)
    return imt

def merge_depth_maps(mono_name = "/home/guido/cnn_depth_tensorflow/tmp/00002.png", 
                     stereo_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_disparity.png",
                     GT_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_GT.png",
                     image_name = "/home/guido/cnn_depth_tensorflow/tmp/00002_org.png",
                     graphics = True, verbose = True, method='sperzi', non_occluded=True,
                     Diogo_weighting=True, scaling=True):

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
    depth_fusion, stereo_confidence = merge.merge_Diogo(depth_stereo, depth_mono, image, graphics = False, Diogo_weighting=Diogo_weighting, scaling=scaling);
    im_mono_conf = createOverlay(gray_scale,1-stereo_confidence)

    if(graphics):        
        fig, ax = plt.subplots()
        plt.imshow(im_mono_conf);
        #ax.imshow(gray_scale, cmap='gray');
        #ax.imshow(1.0 - stereo_confidence, norm = matplotlib.colors.Normalize(vmin= 0.0,vmax=1.0), cmap=plt.cm.RdBu, alpha=0.5);    
        #PCM=ax.get_children()[3] #get the mappable, the 1st and the 2nd are the x and y axes
        #plt.colorbar(PCM, ax=ax)
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
    
    performance = np.zeros([3, 8]);
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, lsi_err, error_map_stereo  = evaluation_utils.compute_errors(depth_GT[:], depth_stereo[:], graphics, name_fig = 'stereo error map', non_occluded=non_occluded);
    performance[0,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, lsi_err];
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, lsi_err, error_map_mono = evaluation_utils.compute_errors(depth_GT[:], depth_mono[:], graphics, name_fig = 'mono error map', non_occluded=non_occluded);
    performance[1,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, lsi_err];
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, lsi_err, error_map_fusion = evaluation_utils.compute_errors(depth_GT[:], depth_fusion[:], graphics, name_fig = 'fusion error map', non_occluded=non_occluded);
    performance[2,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, lsi_err];


    if(graphics):
        ig, axes = plt.subplots(nrows=2, ncols=1);
        cf = axes[0].imshow(depth_fusion);
        axes[0].set_title('depth_fusion');
        cf = axes[1].imshow(depth_GT);
        axes[1].set_title('depth_GT');
#        cf = axes[2,0].imshow(tmp);
#        axes[2,0].set_title('tmp');
#        cf = axes[3,0].imshow(tmp2);
#        axes[3,0].set_title('tmp2');
        #plt.title('Pixel depth error')  
    error_comparison = error_map_stereo - error_map_mono;
    MAX_DEPTH_PLOT = 40;
    im_errormap = createOverlay(gray_scale,error_comparison, v_min=-MAX_DEPTH_PLOT, v_max=MAX_DEPTH_PLOT)
    
    if(graphics):
        
        #plt.figure() 
        #        fig, ax = plt.subplots()
        #        ax.imshow(gray_scale, cmap='gray');
        #        ax.imshow(error_comparison, norm = matplotlib.colors.Normalize(vmin= -MAX_DEPTH_PLOT,vmax=MAX_DEPTH_PLOT), cmap=plt.cm.RdBu, alpha=0.5);    
        #        PCM=ax.get_children()[3] #get the mappable, the 1st and the 2nd are the x and y axes
        #        plt.colorbar(PCM, ax=ax)
        #        #cb.set_lim(-MAX_DEPTH, MAX_DEPTH);
        #        plt.title('abs error stereo - abs error mono');
        
        plt.figure()
        plt.imshow(im_errormap)
        plt.title('abs error stereo - abs error mono');
        
    if(verbose):
        print_performance(performance);


    gt_mono, error_measure_mono = evaluation_utils.compute_error_vs_distance(depth_GT[:], depth_mono[:], error_type = 'raw', graphics = graphics, name_fig='Mono: error vs distance', non_occluded=True, marker='rx');
    gt_stereo, error_measure_stereo = evaluation_utils.compute_error_vs_distance(depth_GT[:], depth_stereo[:], error_type = 'raw', graphics = graphics, name_fig='Stereo: error vs distance', non_occluded=True, marker='bo');
    gt_fusion, error_measure_fusion = evaluation_utils.compute_error_vs_distance(depth_GT[:], depth_fusion[:], error_type = 'raw', graphics = graphics, name_fig='Fusion: error vs distance', non_occluded=True, marker='g*');    
    # dve_info = distance_versus_error_info(error_measure_mono, gt_mono, error_measure_stereo, gt_stereo, error_measure_fusion, gt_fusion);
    dve_info = [error_measure_mono, gt_mono, error_measure_stereo, gt_stereo, error_measure_fusion, gt_fusion];
    
    max_samples = 1000;
    for i in np.arange(0, 6, 2):    
        n_samples = len(dve_info[i]);
        if(n_samples < max_samples):
            print('N samples < max samples for %s' % (image_name));
        inds = np.random.choice(n_samples, size=np.min([max_samples, n_samples]), replace=False);
        dve_info[i] = dve_info[i][inds];
        dve_info[i+1] = dve_info[i+1][inds];
        
    if(graphics):
        plt.figure();
        evaluation_utils.plot_error_vs_distance(gt_mono, error_measure_mono, bin_size_depth_meters=5, color='r', alpha_fill=[0.1, 0.3], label_name='Mono');
        evaluation_utils.plot_error_vs_distance(gt_stereo, error_measure_stereo, bin_size_depth_meters=5, color='b', alpha_fill=[0.1, 0.3], label_name='Stereo');
        plt.legend()
        plt.figure();
        evaluation_utils.plot_error_vs_distance(gt_fusion, error_measure_fusion, bin_size_depth_meters=5, color='g', alpha_fill=[0.1, 0.3], label_name='Fusion');
        plt.legend();
    
    
    # scaled mono
#    depth_mono = merge.scale_mono_map(depth_stereo, depth_mono);    
#    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluation_utils.compute_errors(depth_GT[:], depth_mono[:]);
#    performance[1,:] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3];    
#    
#    if(verbose):
#        print('Scaled mono:')
#        print_performance(performance);
#        
        
    return performance, depth_fusion,im_mono_conf, im_errormap, dve_info;

# merge_depth_maps(graphics=True);
#merge_depth_maps(mono_name = "./tmp/0000000013_sperziboon.png", 
#                     stereo_name = "./tmp/0000000013_disparity.png",
#                     GT_name = "./tmp/0000000013_GT.png",
#                     image_name = "./tmp/0000000013_image.png",
#                     graphics = False, verbose = True)#, method = 'mancini')
