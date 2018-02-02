#!/usr/bin/env python2

# partially based on stereo_match.py example in the OpenCV distribution.
import glob
import numpy as np
import cv2
import pdb
import os
import sys

def show_disparity_maps():

    # parameters for the stereo matching:
    window_size = 9;
    min_disp = 1;
    num_disp = 64; # must be divisible by 16 (http://docs.opencv.org/java/org/opencv/calib3d/StereoSGBM.html)

    TARGET_H = 40;
    TARGET_W = 128;


    WRITE = True;
    im_step = 1;

    fname_left =  sys.argv[1] #"/data/kevin/kitti/raw_data/2011_09_26/images2.txt";
    fname_right = sys.argv[2] #"/data/kevin/kitti/raw_data/2011_09_26/images3.txt";
    with open(fname_left) as f:
        images_left = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    images_left = [x.strip() for x in images_left] 

    with open(fname_right) as f:
        images_right = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    images_right = [x.strip() for x in images_right] 

    # iterate over the files:
    # for idx, im in enumerate(images_left):
    for idx in np.arange(0, len(images_left), im_step):

        print("{} / {}".format(idx, len(images_left)));

        # read the image (consisting of a right image and left image)
        im = images_left[idx];
        print(im)
        imgL = cv2.imread(im);
        #print(images_right[idx])
        imgR = cv2.imread(images_right[idx]);

        # check if the two images correspond:
        num_diff = diff_letters(im, images_right[idx]);
        if(num_diff != 1):
          print('Element in the list: {}. Names: {} <=> {}'.format(idx, im, images_right[idx]));
          pdb.set_trace();

        # calculate the disparities:
        disp = calculate_disparities(imgL, imgR, window_size, min_disp, num_disp);

        # gradient for certainty:
        grey = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY);
        blur = cv2.GaussianBlur(grey,(11,11),0)
        sobelX = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5);
        ret,thresh1 = cv2.threshold(sobelX,175,255,cv2.THRESH_BINARY);
        thresh1 = cv2.resize(thresh1, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST);

        base_name = os.path.basename(im);
        file_name, ext = os.path.splitext(base_name);
        dir_name = os.path.dirname(im);
        dir_name = os.path.dirname(dir_name);
        dir_name = os.path.dirname(dir_name);
        dir_name = dir_name;

        if(WRITE):
          # write the output:
          write_images(dir_name, file_name, imgL, imgR, disp, min_disp, num_disp, thresh1);

def write_images(dir_name, file_name, imgL, imgR, disp, min_disp, num_disp, confidence):
    # write all images:

    if not os.path.exists(dir_name  + "/disp/"): 
        os.makedirs(dir_name  + "/disp/")

    if not os.path.exists(dir_name  + "/conf/"): 
        os.makedirs(dir_name  + "/conf/")

    cv2.imwrite(dir_name + "/disp/" + file_name + "_disparity.png", disp);
    cv2.imwrite(dir_name + "/conf/" + file_name + "_confidence.png", confidence);

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
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disp


show_disparity_maps()
