#!/usr/bin/env python2

import sys
import caffe
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net("/data/kevin/kitti/setup/net_runSparse10.prototxt", "/data/kevin/kitti/cheapSparse10/run_iter_50000.caffemodel",0)

im = Image.open("/data/kevin/kitti/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/0000000000.png")
im = im.resize((298, 218), Image.NEAREST)
nim = np.array(im)
nim = nim.reshape(1,3,218,298)

#imar = im[np.newaxis, np.newaxis, :, :]
#net.blobs['inputData'].reshape(*imar.shape)

net.blobs['inputData'].data[...] = nim
out = net.forward()
mat = out['fine_depthReLU'][0]
print(mat,mat.shape)
mat = (mat[0,:,:]).astype('uint8')
cv2.imwrite('test.png',mat)

