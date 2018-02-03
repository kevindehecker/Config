#!/usr/bin/env python2

import sys
import caffe
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import os

caffe.set_mode_cpu()
#caffe.set_device(1)

print("Usage: prototxt, caffemodel, imagestxt")
net = caffe.Net(sys.argv[1],sys.argv[2],0)

with open(sys.argv[3]) as f:
	input_images = f.readlines()
input_images = [x.strip() for x in input_images] 

# iterate over the files:
for idx, imf in enumerate(input_images):
	print("{} / {}".format(idx, len(input_images)));
	im = Image.open(imf)
	im = im.resize((512, 160), Image.NEAREST)
	nim = np.array(im)
	nim = nim.reshape(1,3,512,160)
	net.blobs['inputData'].data[...] = nim
	out = net.forward()
	mat = out['coarse_depth'][0]
	a = np.asarray(mat)
	if np.mean(a) > 0:
		print(np.mean(a))
	#print(mat)
	mat = mat * 255
	#print(mat,mat.shape)
	mat = (mat[0,:,:]).astype('uint8')

	base_name = os.path.basename(imf);
	file_name, ext = os.path.splitext(base_name);
	dir_name = os.path.dirname(imf);
	dir_name = os.path.dirname(dir_name);
	dir_name = os.path.dirname(dir_name);
	dir_name = dir_name;

	if not os.path.exists(dir_name  + "/monocular/"): 
		os.makedirs(dir_name  + "/monocular/")
	print(dir_name + "/monocular/" + file_name + "_monocular.png")
	mat = cv2.applyColorMap(mat, cv2.COLORMAP_JET)
	cv2.imwrite(dir_name + "/monocular/" + file_name + "_monocular.png",mat)
	imm = mat

	#pdb.set_trace()
	if not os.path.exists(dir_name  + "/combined/"):
		os.makedirs(dir_name  + "/combined/")
	#print(dir_name + "/disp/" + file_name + "_disparity.png")
	imd = cv2.imread(dir_name + "/disp/" + file_name + "_disparity.png")
	imd = cv2.applyColorMap(imd, cv2.COLORMAP_JET)	
	im2 =  cv2.imread(dir_name + "/image_02/data/" + file_name + ".png")


	vis = np.concatenate((im2, imd), axis=0)


	img_w = 128
	img_h = 40
	background = Image.new('RGBA', (1242, 375), (0, 0, 0, 255))
	bg_w, bg_h = background.size
	offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)

	imm = Image.open(dir_name + "/monocular/" + file_name + "_monocular.png")
	background.paste(imm, offset)
	background.save("tmp.png")
	background =  cv2.imread("tmp.png")

	vis = np.concatenate((vis, background), axis=0)
	cv2.imwrite(dir_name + "/combined/" + file_name + ".png",vis)
