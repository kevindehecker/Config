# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:45:01 2018

Merge a monocular and stereo map

@author: guido
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def merge_Diogo(stereo_map, mono_map, image, graphics = False):
    
    TARGET_H = stereo_map.shape[0];
    TARGET_W = stereo_map.shape[1];
    
    if(image.shape[0] != stereo_map.shape[0]):
        # first resize the image:
        image = cv2.resize(image, (TARGET_W, TARGET_H));
        

    # 1. Make the stereo confidence map:
    stereo_confidence = determine_stereo_confidence(stereo_map, image);
    
    # 2. Determine the scale of the monocular map:
    mono_scale = determine_mono_scale(stereo_map, mono_map);
    # from here on work with the scaled mono outputs:
    mono_map *= mono_scale;    
    
    # 3. Get the weighing matrix:
    weight_map = get_Diogo_weight_map(stereo_map, mono_map);
    
    # 4. Do the fusion:
    fusion = np.multiply(stereo_confidence, stereo_map);
    pre_fusion = np.multiply(weight_map, stereo_map) + np.multiply(1-weight_map, mono_map);
    fusion += np.multiply(1 - stereo_confidence, pre_fusion);
    
    # equivalent:
    stereo_confidence = stereo_confidence + np.multiply(1 - stereo_confidence, weight_map);
    mono_confidence = 1 - stereo_confidence;
    fusion = np.multiply(stereo_confidence, stereo_map) + np.multiply(mono_confidence, mono_map);
    
    if(graphics):
        fig, axes = plt.subplots(nrows=1, ncols=3)
        axes[0].imshow(mono_map);
        axes[0].title('Mono');
        axes[0].colorbar();
        axes[1].imshow(stereo_map);
        axes[1].title('Stereo');
        axes[1].colorbar();
        axes[2].imshow(fusion);
        axes[2].title('Fusion');
        axes[2].colorbar();
        
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(image);
        axes[0].title('Image');
        axes[1].imshow(stereo_confidence);
        axes[1].colorbar();
    
    return fusion;
    
    
def determine_stereo_confidence(stereo_map, image, blur_window = 11, gradient_threshold = 175):

    # always when there is no stereo output, low conf
    ret,thresh0 = cv2.threshold(stereo_map, 0, 255, cv2.THRESH_BINARY); 
    # find edges:
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
    blur = cv2.GaussianBlur(grey,(blur_window,blur_window),0);
    sobelX = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5);
    ret,thresh1 = cv2.threshold(sobelX,gradient_threshold,255,cv2.THRESH_BINARY);
    # combine the confidence of thresh 0 (invalid matches) and thresh 1 (edges)
    thresh1 = cv2.bitwise_and(thresh0.astype(np.uint8),thresh1.astype(np.uint8));    
    # blur the threshold image
    blur = cv2.GaussianBlur(thresh1,(blur_window,blur_window),0);
    blur /= np.max(blur[:]);
    # on the edges, the confidence should be 1
    stereo_confidence = np.maximum(blur, thresh1);
    return stereo_confidence;

def determine_mono_scale(stereo_map, mono_map):
    
    # Diogo's original method (TODO: make more robust to outliers):
    max_stereo = max(stereo_map[:]);
    max_mono = max(mono_map[:]);
    mono_scale = max_stereo / max_mono;
    return mono_scale;

def get_Diogo_weight_map(stereo_map, mono_map):
    
    # get the weight map:    
    # first normalize both maps:
    Ns = stereo_map / max(stereo_map[:]);
    Nm = mono_map / max(mono_map[:]);
    
    # Then the weight map for when stereo is bigger, b, or smaller, s
    Wsb = np.divide(Nm, Ns);
    Wss = np.divide(Ns, Nm);

    # Eq 4 in the article:    
    weight_map = np.multiply(Ns >= Nm, Wsb) + np.multiply(Ns < Nm, Wss);
    
    return weight_map;