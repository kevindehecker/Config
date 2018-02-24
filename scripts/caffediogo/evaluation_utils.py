# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:59:23 2018

For a large part from: evaluation utils from https://raw.githubusercontent.com/mrharicot/monodepth/master/utils/evaluation_utils.py

@author: guido
"""

import numpy as np
import pandas as pd
import os
import cv2
from collections import Counter
import pdb
import matplotlib.pyplot as plt

MAX_DEPTH = 80;

# Based on Tony S Yu code:
# https://tonysyu.github.io/plotting-error-bars.html#.WpFOVeZG3CI
def errorfill(x, y, yerr, ymin = None, ymax = None, color=None, alpha_fill=0.3, ax=None):

    ax = ax if ax is not None else plt.gca()
    
    if color is None:
        color = ax._get_lines.color_cycle.next()

    if(ymin is None or ymax is None):
        if np.isscalar(yerr) or len(yerr) == len(y):
            ymin = y - yerr
            ymax = y + yerr
        elif len(yerr) == 2:
            ymin, ymax = yerr
        
        ax.plot(x, y, color=color)
        ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, label='_nolegend_')
    else:
        ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, label='_nolegend_')        
            

#errorfill(x, y_sin, 0.2)
#errorfill(x, y_cos, 0.2)
#plt.show()
    
def plot_error_vs_distance(gt, error_measure, bin_size_depth_meters=1, color=None, alpha_fill=[0.1, 0.3], label_name='graph'):
    # create the bin limits
    bins = np.arange(0.0, MAX_DEPTH, bin_size_depth_meters);
    n_bins = len(bins);
    # get the bin indices of all samples:
    bin_inds = np.digitize(gt, bins, right=False) - 1;
    
    Values = [];
    for b in range(n_bins):
        Values.append([]);
        
    n_points = len(gt);
    for p in range(n_points):
        Values[bin_inds[p]].append(error_measure[p]);
        
    Stats = np.zeros([5, n_bins]);
    for b in range(n_bins):
        err_values = np.asarray(Values[b]);
        if(len(err_values) > 0):
            Stats[0,b] = np.median(err_values);
            Stats[1,b] = np.percentile(err_values, 25);
            Stats[2,b] = np.percentile(err_values, 75);
            Stats[3,b] = np.percentile(err_values, 5);
            Stats[4,b] = np.percentile(err_values, 95);

    ax = plt.gca();
    errorfill(bins, Stats[0,:], None, ymin=Stats[3,:], ymax=Stats[4,:], color=color, alpha_fill=alpha_fill[0], ax=plt.gca());
    errorfill(bins, Stats[0,:], None, ymin=Stats[1,:], ymax=Stats[2,:], color=color, alpha_fill=alpha_fill[1], ax=plt.gca());    
    ax.plot(bins, Stats[0,:], color=color, label=label_name)

    
def plot_dve_info(DVE_info):
    
    plt.figure();
    plt.plot(DVE_info[3], DVE_info[2], 'x') 
    plt.title('stereo');
    
    plt.figure();
    plot_error_vs_distance(DVE_info[1], DVE_info[0], bin_size_depth_meters=1, color='r', alpha_fill=[0.1, 0.3], label_name='Mono');
    plot_error_vs_distance(DVE_info[3], DVE_info[2], bin_size_depth_meters=1, color='b', alpha_fill=[0.1, 0.3], label_name='Stereo');
    plt.xlabel('Ground-truth depth [m]');
    plt.ylabel('Absolute depth error [m]');
    plt.legend()
    plt.figure();
    plot_error_vs_distance(DVE_info[5], DVE_info[4], bin_size_depth_meters=1, color='g', alpha_fill=[0.1, 0.3], label_name='Fusion');    
    plt.xlabel('Ground-truth depth [m]');
    plt.ylabel('Absolute depth error [m]');
    plt.legend();
 
    

def compute_error_vs_distance(gt, pred, error_type = 'rmse', graphics = False, name_fig='Error vs distance', non_occluded=True, marker='bo'):
    
    if(non_occluded):
        inds1 = gt > 0;
        inds2 = pred > 0;
        inds = np.logical_and(inds1, inds2);
        gt = gt[inds];
        pred = pred[inds];
    else:
        inds = gt > 0;
        gt = gt[inds];
        pred = pred[inds];
        inds = pred == 0;
        pred[inds] += 1E-4;

    sorted_inds = np.argsort(gt);
    if(error_type == 'rmse'):
        error_measure = np.sqrt((gt - pred) ** 2);
    elif(error_type == 'raw'):
        error_measure = gt - pred;        
    
    if(graphics):
        plt.figure();
        plt.plot(gt[sorted_inds], error_measure[sorted_inds], marker);
        plt.title(name_fig);
    
    # return the raw lists for accumulation over multiple images:
    return gt, error_measure;
        

def compute_errors(gt, pred, graphics = False, name_fig='error map', non_occluded=True):

    Mask = gt == 0;
    Mask = 1.0 - Mask;
    AbsErr = np.abs(gt - pred);
    AbsErr = np.multiply(Mask, AbsErr);
    
    if(graphics):
        plt.figure();
        plt.imshow(AbsErr);
        cb = plt.colorbar();
        cb.set_clim(0, MAX_DEPTH);
        plt.title(name_fig);
        
    
    if(non_occluded):
        inds1 = gt > 0;
        inds2 = pred > 0;
        inds = np.logical_and(inds1, inds2);
        gt = gt[inds];
        pred = pred[inds];
    else:
        inds = gt > 0;
        gt = gt[inds];
        pred = pred[inds];
        inds = pred == 0;
        pred[inds] += 1E-4;

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    # TODO: should this be log10?
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    delta = np.maximum((gt / pred), (pred / gt))
    a1 = (delta < 1.25   ).mean()
    a2 = (delta < 1.25 ** 2).mean()
    a3 = (delta < 1.25 ** 3).mean()
    
    alpha = np.mean(np.log(gt) - np.log(pred));
    lsi_err = 0.5 * np.mean((np.log(pred) - np.log(gt) + alpha) ** 2)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, lsi_err, AbsErr;

# KITTI

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

def convert_disps_to_depths_kitti(disparity_map, target_width = 1242, target_height = 375, mask = True, limit_depth = 80.0):
    if len(disparity_map.shape) == 3:
        disparity_map = cv2.cvtColor(disparity_map,cv2.COLOR_BGR2GRAY)

    # this way of converting only works when the stereo vision was performed on the original image size:
    disparity_map = disparity_map.astype(float);
    if(mask):
        M = disparity_map == 0;
        
    disparity_map[disparity_map == 0] = 1E-5;
    #factor = (width_to_focal[target_width]/target_width) * 0.54 ;
    #depth_map = factor / disparity_map;
    
    # this formula comes from the run_demoVelodyne.m function in the raw KITTI development kit.
    depth_map = (64.0*5.0) / disparity_map; 

    if(mask):
        depth_map = np.multiply(depth_map, 1-M);
    
    depth_map[depth_map > limit_depth] = limit_depth;
    return depth_map;

def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(map(float, value.split(' ')))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def get_focal_length_baseline(calib_dir, cam):
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    P2_rect = cam2cam['P_rect_02'].reshape(3,4)
    P3_rect = cam2cam['P_rect_03'].reshape(3,4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0,3] / -P2_rect[0,0]
    b3 = P3_rect[0,3] / -P3_rect[0,0]
    baseline = b3-b2

    if cam==2:
        focal_length = P2_rect[0,0]
    elif cam==3:
        focal_length = P3_rect[0,0]

    return focal_length, baseline


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False):
    # load calibration files
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir + 'calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).iteritems() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds==dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth<0] = 0

    if interp:
        # interpolate the depth map to fill in holes
        depth_interp = lin_interp(im_shape, velo_pts_im)
        return depth, depth_interp
    else:
        return depth



