#!/usr/bin/env python2

import sys
import caffe
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb
import os

caffe.set_mode_cpu()
#caffe.set_device(1)

print("Usage: prototxt, caffemodel, imagestxt")
net = caffe.Net(sys.argv[1],sys.argv[2],caffe.TEST)

with open(sys.argv[3]) as f:
	input_images = f.readlines()
input_images = [x.strip() for x in input_images] 

# iterate over the files:
for idx, imf in enumerate(input_images):
	print("{} / {}".format(idx, len(input_images)));
	im = Image.open(imf)
	im = im.resize((256, 78), Image.ANTIALIAS)
	#im.thumbnail(size,Image.ANTIALIAS)
	nim = np.array(im)
	nim = nim.reshape(1,3,78,256)
	net.blobs['data'].data[...] = nim
	out = net.forward()
	mat = out['depth_unnorm'][0]
	#a = np.asarray(mat)
	#print(mat)
	#mat = mat * 39.75
        #print(mat.shape)
	#print(mat[0,30:35,30:35])
	mat = (mat[0,:,:]).astype('uint8')

	base_name = os.path.basename(imf);
	file_name, ext = os.path.splitext(base_name);
	dir_name = os.path.dirname(imf);
	dir_name = os.path.dirname(dir_name);
	dir_name = os.path.dirname(dir_name);
	dir_name = dir_name;

	if not os.path.exists(dir_name  + "/mix_fcn/"): 
		os.makedirs(dir_name  + "/mix_fcn/")
	print(dir_name + "/mix_fcn/" + file_name + "_mix_fcn.png")

	#a = np.asarray(mat)
	#a = np.flipud(a)
	#plt.imsave(dir_name + "/mix_fcn/" + file_name + "_mix_fcn.png",a)

	#mat = cv2.applyColorMap(mat, cv2.COLORMAP_JET)
	cv2.imwrite(dir_name + "/mix_fcn/" + file_name + "_mix_fcn.png",mat)

	if False:
		im2 =  cv2.imread(dir_name + "/image_02/data/" + file_name + ".png")


		if not os.path.exists(dir_name  + "/combined/"):
			os.makedirs(dir_name  + "/combined/")
		#print(dir_name + "/disp/" + file_name + "_disparity.png")
		imd = cv2.imread(dir_name + "/disp/" + file_name + "_disparity.png")
		imd = cv2.applyColorMap(imd, cv2.COLORMAP_JET)
		imd = cv2.resize(imd,(im2.shape[1], im2.shape[0]), interpolation = cv2.INTER_CUBIC)

		vis = np.concatenate((im2, imd), axis=0)

		imm =  cv2.imread(dir_name + "/mix_fcn/" + file_name + "_mix_fcn.png")
		imm = cv2.resize(imm,(im2.shape[1], im2.shape[0]), interpolation = cv2.INTER_CUBIC)
		vis = np.concatenate((vis, imm), axis=0)

		cv2.imwrite(dir_name + "/combined/" + file_name + ".png",vis)
