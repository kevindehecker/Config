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

def compute_errors(gt, pred, graphics = True, name_fig='error map', non_occluded=True):

    if(graphics):
        Mask = gt == 0;
        Mask = 1.0 - Mask;
        AbsErr = np.abs(gt - pred);
        AbsErr = np.multiply(Mask, AbsErr);
        plt.figure();
        plt.imshow(AbsErr);
        plt.colorbar();
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

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# KITTI

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

def convert_disps_to_depths_kitti(disparity_map, target_width = 1242, target_height = 375, mask = True, limit_depth = 80.0):
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



