#!/usr/bin/env python2
 
#usage:
#python2 ./process_images.py /data/kevin/kitti/raw_data/2011_09_26/val_images2.txt /data/kevin/kitti/raw_data/2011_09_26/val_images3.txt mrharicot False

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
import evaluation_utils
from random import *

regen_combined = False
regen_merged = False
regen_stereo = True
regen_sperzi = False
regen_mancini = False

#parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
#parser.add_argument('--encoder',          type=str,   help='type of encoder, mrharicot or mancini', default='mrharicot')

# whether to use mrharicot or mancini
CNN = sys.argv[3] #args.encoder #'mancini';

if(CNN == 'mrharicot'):
    sperzi_dir = '/data/kevin/Config/scripts/caffediogo/monodepth/'
    sys.path.insert(0, sperzi_dir)
    import monodepth_kevin
else:
    # Mancini's network:
    mancini_dir = '/home/guido/trained_mancini/'
    # '/home/guido/SSL_OF/keras/';
    sys.path.insert(0, mancini_dir)
#    # sys.path.append('/home/SSL_OF/keras/')
    import upsample_vgg16 


def add_dve_info(DVE_info, dve_info):    
    for i in range(6):
        DVE_info[i] = np.concatenate((dve_info[i], DVE_info[i]));
    return DVE_info;

def diff_letters(a,b):
    return sum ( a[i] != b[i] for i in range(len(a)) )

def generate_maps():
    im_step = 1;
    fname_left =  sys.argv[1]
    fname_right = sys.argv[2]
    with open(fname_left) as f:
        images_left = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    images_left = [x.strip() for x in images_left] 

    with open(fname_right) as f:
        images_right = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    images_right = [x.strip() for x in images_right] 

    if(CNN == 'mancini'):
        model = upsample_vgg16.load_model(dir_name=mancini_dir);
    else:
        model = 'hack'

    DVE_info1 = [];
    DVE_info2 = [];
    for i in range(6):
        DVE_info1.append([]);
        DVE_info2.append([]);
    Performance1 = np.zeros([3, 8]);
    Performance2 = np.zeros([3, 8]);
    n_perfs = 0;
    non_occluded = True
    # for the merging process:
    Diogo_weighting = True;
    mono_scaling = True;

    # iterate over the files:
    # for idx, im in enumerate(images_left):
    for idx in np.arange(0, len(images_left), im_step):
        # read the image (consisting of a right image and left image)
        im = images_left[idx];        
        imgL = cv2.imread(im);
        imgR = cv2.imread(images_right[idx]);
        print("{} / {} {}".format(idx, len(images_left),im));        
        # check if the two images correspond:
        num_diff = diff_letters(im, images_right[idx]);
        if(num_diff != 1):
          print('Element in the list: {}. Names: {} <=> {}'.format(idx, im, images_right[idx]));
          pdb.set_trace();
        
        base_name = os.path.basename(im);
        file_name, ext = os.path.splitext(base_name);
        dir_name = os.path.dirname(im);
        dir_name = os.path.dirname(dir_name);
        dir_name = os.path.dirname(dir_name);
  
        #sperzi_path = do_sperziboon(dir_name, file_name, im);        
        sperzi_path = do_mancini_original(dir_name, file_name, im)
        mancini_path = do_mancini(dir_name, file_name, im,model)            
        
        dirdate_name = os.path.basename(dir_name)
        gt_path = dir_name + "/../../../data_depth_annotated/val/" + dirdate_name + "/proj_depth/groundtruth/image_02/" + file_name + ".png"        
        stereo_path,conf_path = do_stereo(dir_name, file_name, imgL, imgR)       

        #   merged_sperzi_path,fusionconf_sperzi_path,perf_result1, dve_info1 = do_merge(dir_name, file_name, sperzi_path,stereo_path,gt_path,im,"sperzi",non_occluded, Diogo_weighting =Diogo_weighting, mono_scaling=mono_scaling)
        # DVE_info1 = add_dve_info(DVE_info1, dve_info1);
        # Performance1 += perf_result1;
        # if(np.mod(n_perfs, 10) == 0):
        #     print('nocc: ' + str(non_occluded) + ", Diogo_weighting: " +  str(Diogo_weighting) + ", mono_scaling: " + str(mono_scaling))
        #     performance.print_performance(Performance1 / n_perfs, name = 'Performance mix_fcn');
        # plt.show()
        # merged_mancini_path,fusionconf_mancini_path,perf_result2, dve_info2 = do_merge(dir_name, file_name, mancini_path,stereo_path,gt_path,im, "manchini",non_occluded, Diogo_weighting =Diogo_weighting, mono_scaling=mono_scaling)
        # DVE_info2 = add_dve_info(DVE_info2, dve_info2);
        # Performance2 += perf_result2;
        # if(np.mod(n_perfs, 10) == 0):
        #     performance.print_performance(Performance2 / n_perfs, name = 'Performance manchini');
            
        # n_perfs += 1;

        # do_combine(dir_name, file_name, sperzi_path,mancini_path,stereo_path,conf_path,gt_path, im,merged_sperzi_path,merged_mancini_path,fusionconf_sperzi_path,fusionconf_mancini_path)
        

    # make DVE plots:
    evaluation_utils.plot_dve_info(DVE_info1);
    evaluation_utils.plot_dve_info(DVE_info2);
    
    Performance1 = Performance1 / n_perfs
    filehandler = open("performance_1.pkl","wb")
    pickle.dump(Performance1, filehandler)
    filehandler.close()
    print('nocc: ' + str(non_occluded) + ", Diogo_weighting: " +  str(Diogo_weighting) + ", mono_scaling: " + str(mono_scaling))
    performance.print_performance(Performance1, name = 'Performance mix_fcn');

    Performance2 = Performance2 / n_perfs
    filehandler = open("performance_2.pkl","wb")
    pickle.dump(Performance2, filehandler)
    filehandler.close()
    performance.print_performance(Performance2, name = 'Performance manchini');
    
    filehandler = open("DVE_info_1.pkl","wb")
    pickle.dump(DVE_info1, filehandler)
    filehandler.close()

    filehandler = open("DVE_info_2.pkl","wb")
    pickle.dump(DVE_info2, filehandler)
    filehandler.close()
    
    plt.show()
    

def do_combine(dir_name, file_name, sperzi_path,mancini_path,stereo_path,conf_path,gt_path, im_rgb_path,merged_sperzi_path,merged_mancini_path,fusionconf_sperzi_path,fusionconf_mancini_path):
    if not os.path.exists(dir_name  + "/combined/"): 
        os.makedirs(dir_name  + "/combined/")
    combined_path = dir_name + "/combined/" + file_name + "_combined.jpg"
    if not os.path.isfile(combined_path) or sys.argv[4] == 'True' or regen_combined:
        im = cv2.imread(im_rgb_path)
        imd = cv2.imread(stereo_path)
        imc = cv2.imread(conf_path)
        img = cv2.imread(gt_path)
        ims = cv2.imread(sperzi_path)
        imm = cv2.imread(mancini_path)
        imms = cv2.imread(merged_sperzi_path)
        immm = cv2.imread(merged_mancini_path)
        imfcm = cv2.imread(fusionconf_mancini_path)
        imfcs = cv2.imread(fusionconf_sperzi_path)

        #pdb.set_trace()
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        imd = cv2.applyColorMap(imd, cv2.COLORMAP_JET)
        ims = cv2.applyColorMap(ims, cv2.COLORMAP_JET)
        ims = cv2.resize(ims,(img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
        imm = cv2.applyColorMap(imm, cv2.COLORMAP_JET)
        imm = cv2.resize(imm,(img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
        imms = cv2.applyColorMap(imms, cv2.COLORMAP_JET)
        imms = cv2.resize(imms,(img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
        immm = cv2.applyColorMap(immm, cv2.COLORMAP_JET)
        immm = cv2.resize(immm,(img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
        
        
        #rgb    | conf
        #laser  | stereo
        #sperzi | mancini
        #merged | merged
        
        visL = np.concatenate((im, img), axis=0)
        visL = np.concatenate((visL, ims), axis=0)
        visL = np.concatenate((visL, imms), axis=0)
        visL = np.concatenate((visL, imfcs), axis=0)
        
        visR = np.concatenate((imc, imd), axis=0)
        visR = np.concatenate((visR, imm), axis=0) 
        visR = np.concatenate((visR, immm), axis=0)
        visR = np.concatenate((visR, imfcm), axis=0)
        
        vis = np.concatenate((visL, visR), axis=1)
        cv2.imwrite(combined_path, vis);

    
def do_merge(dir_name, file_name, mono_path,stereo_path,gt_path, im_rgb_path, cnn,non_occluded, Diogo_weighting = True, mono_scaling=True):
    if not os.path.exists(dir_name  + "/merged/"): 
        os.makedirs(dir_name  + "/merged/")
    merged_path = dir_name + "/merged/" + file_name + "_merged_" + cnn + ".png"
    if not os.path.exists(dir_name  + "/fusionconf/"): 
        os.makedirs(dir_name  + "/fusionconf/")
    fusionconf_path = dir_name + "/fusionconf/" + file_name + "_fusionconf_" + cnn + ".png"
    perf_result, depth_fusion,im_mono_conf,im_error_map, dve_info = performance.merge_depth_maps(mono_path,stereo_path,gt_path,im_rgb_path,graphics=False,verbose=False, method=cnn, non_occluded=non_occluded, Diogo_weighting=Diogo_weighting, scaling=mono_scaling) 
    if not os.path.isfile(merged_path) or sys.argv[4] == 'True' or regen_merged:    
        cv2.imwrite(merged_path, depth_fusion);
    if not os.path.isfile(merged_path) or sys.argv[4] == 'True' or regen_merged:    
        cv2.imwrite(fusionconf_path, np.asarray(im_mono_conf));
    return merged_path,fusionconf_path,perf_result, dve_info
    
def do_mancini(dir_name, file_name, im_rgb_path,model):
    if not os.path.exists(dir_name  + "/mancini/"): 
        os.makedirs(dir_name  + "/mancini/")
    mono_path = dir_name  + "/mancini/" + file_name + "_mancini.png";
    if (not os.path.isfile(mono_path) and not CNN == 'mrharicot') or sys.argv[4] == 'True' or regen_mancini:
        prediction = upsample_vgg16.test_model_on_image(im_rgb_path, save_image_name = mono_path, model=model);
    return mono_path

def do_mancini_original(dir_name, file_name, im_rgb_path):
    if not os.path.exists(dir_name  + "/mix_fcn/"): 
        os.makedirs(dir_name  + "/mix_fcn/")
    mono_path = dir_name  + "/mix_fcn/" + file_name + "_mix_fcn.png";
    if (not os.path.isfile(mono_path) and not CNN == 'mrharicot') or sys.argv[4] == 'True' :
        ERROR
    return mono_path
    
def do_sperziboon(dir_name, file_name, im_rgb_path):    
    if not os.path.exists(dir_name  + "/sperzi/"): 
        os.makedirs(dir_name  + "/sperzi/")
    out_file = dir_name + "/sperzi/" + file_name + "_sperziboon.png"
    if (not os.path.isfile(out_file) and CNN == 'mrharicot') or sys.argv[4] == 'True' or regen_sperzi:
        monodepth_kevin.process_im_sperzi(im_rgb_path,'/data/kevin/Config/scripts/caffediogo/monodepth/models/model_kitti',out_file)
    return out_file

def do_stereo(dir_name, file_name, imgL, imgR):

    if not os.path.exists(dir_name  + "/disp/"): 
        os.makedirs(dir_name  + "/disp/")
    if not os.path.exists(dir_name  + "/conf/"): 
        os.makedirs(dir_name  + "/conf/")
    stereo_path = dir_name + "/disp/" + file_name + "_disparity.png"
    conf_path = dir_name + "/conf/" + file_name + "_confidence.png"
    if (not os.path.isfile(stereo_path) and not os.path.isfile(conf_path)) or sys.argv[4] == 'True' or regen_stereo:
        # parameters for the stereo matching:
        window_size = 9;
        min_disp = 1;
        num_disp = 64; # must be divisible by 16 (http://docs.opencv.org/java/org/opencv/calib3d/StereoSGBM.html)
        
        # calculate the disparities:
        disp = calculate_disparities(imgL, imgR, window_size, min_disp, num_disp);
        ret,thresh0 = cv2.threshold(disp,0,65535,cv2.THRESH_BINARY);

        # gradient for certainty:
        grey = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY);
        blur = cv2.GaussianBlur(grey,(11,11),0)
        sobelX = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5);
        ret,thresh1 = cv2.threshold(sobelX,175*256,65535,cv2.THRESH_BINARY);

        #mask out unknown pixels in disp map
        # thresh 0 is for uncertain disparities (e.g., left band in left image + bad matches)
        # thresh 1 is for detecting high texture regions
        confidence = cv2.bitwise_and(thresh0.astype(np.uint8),thresh1.astype(np.uint8))

        # write all images:
        disp = disp.astype(np.uint16)
        cv2.imwrite(stereo_path, disp);
        cv2.imwrite(conf_path, confidence);
        
    return stereo_path,conf_path

def calculate_disparities(imgL, imgR, window_size, min_disp, num_disp):

    # semi-global matching: https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html

    # minDisparity	Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
    # numDisparities	Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
    # blockSize	Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    # P1	The first parameter controlling the disparity smoothness. See below.
    # P2	The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*SADWindowSize*SADWindowSize and 32*number_of_image_channels*SADWindowSize*SADWindowSize , respectively).
    # disp12MaxDiff	Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
    # preFilterCap	Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
    # uniquenessRatio	Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
    # speckleWindowSize	Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    # speckleRange	Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
    # mode	Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .

    # stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities = num_disp, blockSize = window_size+1, preFilterCap=10, P1= 4 * 3 * window_size ** 2, P2 = 32 * 3 * window_size ** 2);
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities = num_disp, blockSize = window_size, preFilterCap=0, P1= 4 * 3 * window_size ** 2, P2 = 16 * 3 * window_size ** 2, disp12MaxDiff = 2)
    disp = stereo.compute(imgL, imgR) #.astype(np.float32) / 16.0
    return disp

generate_maps()
