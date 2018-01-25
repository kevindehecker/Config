# partially based on stereo_match.py example in the OpenCV distribution.
import glob
import numpy as np
import cv2
import pdb
import os

def show_disparity_maps():

    # parameters for the stereo matching:
    window_size = 9;
    min_disp = 1;
    num_disp = 64; # must be divisible by 16 (http://docs.opencv.org/java/org/opencv/calib3d/StereoSGBM.html)
    
    # get a list of all bmp-files in a directory:
    image_set = 'KITTI'; #'tmp'; #'KITTI'; # 'race', 'pole', 'middlebury'    

    if(image_set == 'middlebury' or image_set == 'tmp'):
        
        imgL = cv2.imread('./tmp/0000000000_left.png');#('./middlebury/im2.png');
        imgR = cv2.imread('./tmp/0000000000_right.png'); #('./middlebury/im6.png');
        
        # calculate the disparities:
        disp = calculate_disparities(imgL, imgR, window_size, min_disp, num_disp);

        # gradient for certainty:
        grey = cv2.cvtColor( imgL, cv2.COLOR_RGB2GRAY )
        sobelX = cv2.Sobel(grey,cv2.CV_8U,1,0,ksize=5);
        ret,thresh1 = cv2.threshold(sobelX,70,255,cv2.THRESH_BINARY)

        # show the output
        show_disparity_map(imgL, imgR, disp, min_disp, num_disp, thresh1);
        # write_images(imgL, imgR, disp, min_disp, num_disp, thresh1);
    elif(image_set == 'KITTI'):

        SHOW = False;
        WRITE = True;
        im_step = 1;

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
        # for idx, im in enumerate(images_left):
        for idx in np.arange(0, len(images_left), im_step):

            print("{} / {}".format(idx, len(images_left)));

            # read the image (consisting of a right image and left image)
            im = images_left[idx];
            imgL = cv2.imread(im);
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
            sobelX = cv2.Sobel(grey,cv2.CV_64F,1,0,ksize=5);
            ret,thresh1 = cv2.threshold(sobelX,70,255,cv2.THRESH_BINARY);
            
            if(SHOW):
              # show the output
              show_disparity_map(imgL, imgR, disp, min_disp, num_disp, thresh1);

            base_name = os.path.basename(im);
            file_name, ext = os.path.splitext(base_name);
            dir_name = os.path.dirname(im);
            dir_name = os.path.dirname(dir_name);
            dir_name = os.path.dirname(dir_name);
            dir_name = dir_name;

            if(WRITE):
              # write the output:
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
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities = num_disp, blockSize = window_size, preFilterCap=0, P1= 4 * 3 * window_size ** 2, P2 = 16 * 3 * window_size ** 2, disp12MaxDiff = 2);
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0;
    return disp; 

def calculate_disparities_Eugene():
    # K_xx: 3x3 calibration matrix of camera xx before rectification
    K_L = np.matrix(
        [[9.597910e+02, 0.000000e+00, 6.960217e+02],
         [0.000000e+00, 9.569251e+02, 2.241806e+02],
         [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    K_R = np.matrix(
        [[9.037596e+02, 0.000000e+00, 6.957519e+02],
         [0.000000e+00, 9.019653e+02, 2.242509e+02],
         [0.000000e+00, 0.000000e+00, 1.000000e+00]])
     
    # D_xx: 1x5 distortion vector of camera xx before rectification
    D_L = np.matrix([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02])
    D_R = np.matrix([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])
     
    # R_xx: 3x3 rotation matrix of camera xx (extrinsic)
    R_L = np.transpose(np.matrix([[9.999758e-01, -5.267463e-03, -4.552439e-03],
                                  [5.251945e-03, 9.999804e-01, -3.413835e-03],
                                  [4.570332e-03, 3.389843e-03, 9.999838e-01]]))
    R_R = np.matrix([[9.995599e-01, 1.699522e-02, -2.431313e-02],
                     [-1.704422e-02, 9.998531e-01, -1.809756e-03],
                     [2.427880e-02, 2.223358e-03, 9.997028e-01]])
     
    # T_xx: 3x1 translation vector of camera xx (extrinsic)
    T_L = np.transpose(np.matrix([5.956621e-02, 2.900141e-04, 2.577209e-03]))
    T_R = np.transpose(np.matrix([-4.731050e-01, 5.551470e-03, -5.250882e-03]))
     
    # Guido: Is this not strange? it is 50 pixels less in height:
    IMG_SIZE = (1392, 512)
     
    rotation = R_L * R_R
    translation = T_L - T_R
     
    # output matrices from stereoRectify init
    R1 = np.zeros(shape=(3, 3))
    R2 = np.zeros(shape=(3, 3))
    P1 = np.zeros(shape=(3, 4))
    P2 = np.zeros(shape=(3, 4))
    Q = np.zeros(shape=(4, 4))
     
    # Guido: the images are not changed...
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K_L, D_L, K_R, D_R, IMG_SIZE, rotation, translation,
                                                                      R1, R2, P1, P2, Q,
                                                                      newImageSize=(1242, 375))
    
    window_size = 9
    minDisparity = 1
    stereo = cv2.StereoSGBM_create(
        blockSize=10,
        numDisparities=64,
        preFilterCap=10,
        minDisparity=minDisparity,
        P1=4 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2
    )
     
    disparity = stereo.compute(imgL, imgR)
    disp = disparity[validPixROI1[1]:validPixROI1[3], validPixROI1[0]:validPixROI1[2]]
     
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL[validPixROI1[1]:validPixROI1[3], validPixROI1[0]:validPixROI1[2]], cv2.COLOR_BGR2RGB)
     
    mask = disp > disp.min() + minDisparity
    out_points = points[mask]
    out_colors = colors[mask] 
     
    #write_ply("out.ply", out_points, out_colors)
    plt.imshow(disp)
    plt.show()

show_disparity_maps();
