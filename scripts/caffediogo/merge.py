# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:45:01 2018

Merge a monocular and stereo map

@author: guido
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d


def merge_Diogo(stereo_map, mono_map, image, graphics = False):
    
    TARGET_H = stereo_map.shape[0];
    TARGET_W = stereo_map.shape[1];
    
    if(image.shape[0] != stereo_map.shape[0]):
        # first resize the image:
        image = cv2.resize(image, (TARGET_W, TARGET_H));
        

    # 1. Make the stereo confidence map:
    stereo_confidence = determine_stereo_confidence(stereo_map, image);
    if(graphics):
        plt.figure();
        plt.imshow(stereo_confidence);
        plt.title('Stereo confidence');
        plt.colorbar();
    
    # 2. Determine the scale of the monocular map:
    mono_map = scale_mono_map(stereo_map, mono_map);
    
    # 3. Get the weighing matrix:
    weight_map = get_Diogo_weight_map(stereo_map, mono_map);
    if(graphics):
        plt.figure();
        plt.imshow(weight_map);
        plt.title('Weight map');
        plt.colorbar();
    
    # 4. Do the fusion:
#    fusion = np.multiply(stereo_confidence, stereo_map);
#    pre_fusion = np.multiply(weight_map, stereo_map) + np.multiply(1-weight_map, mono_map);
#    fusion += np.multiply(1 - stereo_confidence, pre_fusion);
    
    # equivalent:
    stereo_confidence = stereo_confidence + np.multiply(1 - stereo_confidence, 1-weight_map);
    if(graphics):
        plt.figure();
        plt.imshow(stereo_confidence);
        plt.title('Updated stereo confidence');
        plt.colorbar();
        
    mono_confidence = 1 - stereo_confidence;
    if(graphics):
        plt.figure();
        plt.imshow(mono_confidence);
        plt.title('Mono confidence');
        plt.colorbar();
        
    fusion = np.multiply(stereo_confidence, stereo_map) + np.multiply(mono_confidence, mono_map);
    
    # 5. post-processing with median filter:
    # fusion = cv2.medianBlur(fusion.astype(np.uint8), 3);
    fusion = medfilt2d(fusion, 3);
    
    if(graphics):
        fig, axes = plt.subplots(nrows=1, ncols=3)
        axes[0].imshow(mono_map);
        axes[0].set_title('Mono');
        axes[1].imshow(stereo_map);
        axes[1].set_title('Stereo');
        axes[2].imshow(fusion);
        axes[2].set_title('Fusion');
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(image);
        axes[0].set_title('Image');
        axes[1].imshow(stereo_confidence);    
        axes[1].set_title('Stereo confidence');        
    
    return fusion;
    
    
def determine_stereo_confidence(stereo_map, image, blur_window = 3, gradient_threshold = 175, graphics = False):

    # TODO: use matching cost / uncertainty of stereo

    # always when there is no stereo output, low conf
    ret,thresh0 = cv2.threshold(stereo_map, 0, 255, cv2.THRESH_BINARY); 
    # find edges:
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
    blur = cv2.GaussianBlur(grey,(blur_window,blur_window),0);

    if(graphics):
        plt.figure()
        plt.imshow(blur);
        plt.title('initial blur grayscale for conf stereo')
    
    #blur = grey;
    sobelX = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5);
    ret,thresh1 = cv2.threshold(sobelX,gradient_threshold,255,cv2.THRESH_BINARY);
    
    if(graphics):
        plt.figure();
        plt.imshow(thresh1);
        plt.title('thresh1: enough vertical contrast');
    
    # combine the confidence of thresh 0 (invalid matches) and thresh 1 (edges)
    thresh1 = cv2.bitwise_and(thresh0.astype(np.uint8),thresh1.astype(np.uint8));  
    thresh1 = thresh1.astype(float);
    
    if(graphics):
        plt.figure()
        plt.imshow(thresh1);
        plt.title('thresh2: vertical contrast and valid match')    
    
    
    # blur the threshold image
    blur = cv2.GaussianBlur(thresh1,(blur_window*2+1,blur_window*2+1),0);
    blur /= np.max(blur[:]);
    
#    if(graphics):
#        plt.figure()
#        plt.imshow(blur);
#        plt.title('second blur stereo')    
    
    # on the edges, the confidence should be 1
    stereo_confidence = blur;
    # stereo_confidence = np.maximum(blur, thresh1);
    
    if(graphics):
        plt.figure()
        plt.imshow(stereo_confidence);
        plt.title('stereo confidence')    
    
    return stereo_confidence;

def scale_mono_map(stereo_map, mono_map):
    
    # Diogo's original method (TODO: make more robust to outliers):
    # TODO: what is the best scaling for performance?
    max_stereo = np.max(stereo_map[stereo_map != 0]);
    min_stereo = np.min(stereo_map[stereo_map != 0]);
    min_mono = np.min(mono_map[:]);
    mono_map -= min_mono;
    max_mono = np.max(mono_map[:]);
    mono_map /= max_mono;
    mono_map *= max_stereo - min_stereo;
    mono_map += min_stereo;
    
    return mono_map;

def get_Diogo_weight_map(stereo_map, mono_map, graphics = False):
    
    # get the weight map:    
    # first normalize both maps:
    Ns = stereo_map / np.max(stereo_map[:]);
    Nm = mono_map / np.max(mono_map[:]);
    
    Ns[Ns==0] = 1E-3;
    Nm[Nm==0] = 1E-3;
    
    # Then the weight map for when stereo is bigger, b, or smaller, s
    Wsb = np.divide(Nm, Ns);
    Wss = np.divide(Ns, Nm);

    # Eq 4 in the article:    
    weight_map = np.multiply(Ns >= Nm, Wsb) + np.multiply(Ns < Nm, Wss);
    weight_map[stereo_map == 0] = 1;
    
    if(graphics):
        plt.figure()
        plt.imshow(weight_map);
        plt.title('weight map');
        plt.colorbar();
    
        
    
    return weight_map;