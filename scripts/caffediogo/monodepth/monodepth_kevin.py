# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import pdb
import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
import cv2

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

#parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

#parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
#parser.add_argument('--image_path',       type=str,   help='path to the input image', required=True)
#parser.add_argument('--output_path',       type=str,   help='path to the output image', required=True)
#parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
#parser.add_argument('--input_height',     type=int,   help='input height', default=256)
#parser.add_argument('--input_width',      type=int,   help='input width', default=512)

#args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def process_im_sperzi(image_path = '/data/kevin/kitti/raw_data/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000005.png',checkpoint_path = '/data/kevin/Config/scripts/caffediogo/monodepth/models/model_kitti',output_path = 'test.png' ):
    
    input_height = 256
    input_width = 512
    encoder = 'vgg'
    params = monodepth_parameters(
    encoder='vgg',
    height=256,
    width=512,
    batch_size=2,
    num_threads=1,
    num_epochs=1,
    do_stereo=False,
    wrap_mode="border",
    use_deconv=False,
    alpha_image_loss=0,
    disp_gradient_loss_weight=0,
    lr_loss_weight=0,
    full_summary=False)
    

    left  = tf.placeholder(tf.float32, [2, input_height, input_width, 3])
    model = MonodepthModel(params, "test", left, None)

    input_image = scipy.misc.imread(image_path, mode="RGB")
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [input_height, input_width], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

    output_directory = os.path.dirname(image_path)
    output_name = os.path.splitext(os.path.basename(image_path))[0]

    #np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    #plt.imsave(output_path, disp_to_img)
    cv2.imwrite(output_path, disp_to_img);
    

#process_im_sperzi()