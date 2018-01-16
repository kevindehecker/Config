# partially based on stereo_match.py example in the OpenCV distribution.
import glob
import numpy as np
import cv2
import pdb
import os

def show_disparity_maps():

    # parameters for the stereo matching:
    window_size = 7;
    min_disp = 0;
    num_disp = 256; # must be divisible by 16 (http://docs.opencv.org/java/org/opencv/calib3d/StereoSGBM.html)
    
    # get a list of all bmp-files in a directory:
    image_set = 'KITTI'; # 'race', 'pole', 'middlebury'    

    if(image_set == 'middlebury'):
        
        imgL = cv2.imread('./middlebury/im2.png');
        imgR = cv2.imread('./middlebury/im6.png');
        
         # calculate the disparities:
        num_disp = 64;
        disp = calculate_disparities(imgL, imgR, window_size, min_disp, num_disp);

        # gradient for certainty:
        grey = cv2.cvtColor( imgL, cv2.COLOR_RGB2GRAY )
        sobelX = cv2.Sobel(grey,cv2.CV_8U,1,0,ksize=5);
        ret,thresh1 = cv2.threshold(sobelX,70,255,cv2.THRESH_BINARY)

        # show the output
        # show_disparity_map(imgL, imgR, disp, min_disp, num_disp, thresh1);
        write_images(imgL, imgR, disp, min_disp, num_disp, thresh1);
    elif(image_set == 'KITTI'):

        fname_left = "images2.txt"; #"/data/kevin/kitti/raw_data/2011_09_26/images3.txt";
        fname_right = "images3.txt";
        with open(fname_left) as f:
            images_left = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        images_left = [x.strip() for x in images_left] 

        with open(fname_right) as f:
            images_right = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        images_right = [x.strip() for x in images_right] 

        # iterate over the files:
        for idx, im in enumerate(images_left):

            print("{} / {}".format(idx, len(images_left)));

            # read the image (consisting of a right image and left image)
            imgL = cv2.imread(im);
            imgR = cv2.imread(images_right[idx]);

            # calculate the disparities:
            disp = calculate_disparities(imgL, imgR, window_size, min_disp, num_disp);

            # gradient for certainty:
            grey = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY);
            sobelX = cv2.Sobel(grey,cv2.CV_64F,1,0,ksize=5);
            ret,thresh1 = cv2.threshold(sobelX,70,255,cv2.THRESH_BINARY);
            
            # show the output
            # show_disparity_map(imgL, imgR, disp, min_disp, num_disp, thresh1);

            base_name = os.path.basename(im);
            file_name, ext = os.path.splitext(base_name);
            dir_name = os.path.dirname(im);
            dir_name = os.path.dirname(dir_name);
            dir_name = os.path.dirname(dir_name);
            dir_name = dir_name;

            write_images(dir_name, file_name, imgL, imgR, disp, min_disp, num_disp, thresh1);
        

    else:
        im_files = glob.glob("./" + image_set + "/*.bmp");
        
        # iterate over the files:
        for im in im_files:
            # read the image (consisting of a right image and left image)
            cv_im = cv2.imread(im);
            imgL = cv_im[0:96, 126:252, :];
            imgR = cv_im[0:96, 0:126, :];
            # calculate the disparities:
            disp = calculate_disparities(imgL, imgR, window_size, min_disp, num_disp);
            # gradient for certainty:
            sobelX = cv2.Sobel(imgL,cv2.CV_64F,1,0,ksize=5);
            ret,thresh1 = cv2.threshold(sobelX,70,255,cv2.THRESH_BINARY)
            
            # show the output
            # show_disparity_map(imgL, imgR, disp, min_disp, num_disp, thresh1);
            write_images(imgL, imgR, disp, min_disp, num_disp, thresh1);
            
def show_disparity_map(imgL, imgR, disp, min_disp, num_disp, confidence):		
    # show the output
    cv2.imshow('left', imgL);
    cv2.imshow('right', imgR);
    cv2.imshow('disparity', (disp-min_disp)/num_disp);
    cv2.imshow('confidence', confidence);
    		
    # wait for a key to be pressed before moving on:
    cv2.waitKey();
    cv2.destroyAllWindows();

def write_images(dir_name, file_name, imgL, imgR, disp, min_disp, num_disp, confidence):
    # write all images:
    #cv2.imwrite(base_name + file_name + "_left.png", imgL);
    #cv2.imwrite(base_name + file_name + "_right.png", imgR);

    if not os.path.exists(dir_name  + "/disp/"): 
        os.makedirs(dir_name  + "/disp/")

    if not os.path.exists(dir_name  + "/conf/"): 
        os.makedirs(dir_name  + "/conf/")

    cv2.imwrite(dir_name + "/disp/" + file_name + "_disparity.png", disp);
    cv2.imwrite(dir_name + "/conf/" + file_name + "_confidence.png", confidence);

def calculate_disparities(imgL, imgR, window_size, min_disp, num_disp):
    # semi-global matching:
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities = num_disp, blockSize = window_size);
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0;
    return disp; 
	
show_disparity_maps();
