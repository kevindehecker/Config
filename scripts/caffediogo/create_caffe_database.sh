#!/usr/bin/env bash

#cd /data/kevin/kitti/raw_data/2011_09_26
#find /data/kevin/kitti/raw_data/2011_09_26/ -iname *.png | grep image_03 > /data/kevin/kitti/raw_data/2011_09_26/images3.txt
#find /data/kevin/kitti/raw_data/2011_09_26/ -iname *.png | grep image_02 > /data/kevin/kitti/raw_data/2011_09_26/images2.txt
#find /data/kevin/kitti/raw_data/2011_09_26/ -iname *.png | grep disp> /data/kevin/kitti/raw_data/2011_09_26/disp.txt
#find /data/kevin/kitti/raw_data/2011_09_26/ -iname *.png | grep conf > /data/kevin/kitti/raw_data/2011_09_26/conf.txt


#cd /data/kevin/kitti/setup

#depth laser ground truth
find /data/kevin/kitti/data_depth_annotated/train/2011_09_26_drive_0*/proj_depth/groundtruth/image_03/  -iname *.png | grep image_03 > /data/kevin/kitti/data_depth_annotated/train/train_laser_images3.txt
find /data/kevin/kitti/data_depth_annotated/val/2011_09_26_drive_0*/proj_depth/groundtruth/image_03/ -iname *.png | grep image_03 > /data/kevin/kitti/data_depth_annotated/val/val_laser_images3.txt

#use these as templates, copy to all destinations
cp /data/kevin/kitti/data_depth_annotated/val/val_laser_images3.txt /data/kevin/kitti/raw_data/2011_09_26/val_images2.txt
cp /data/kevin/kitti/data_depth_annotated/val/val_laser_images3.txt /data/kevin/kitti/raw_data/2011_09_26/val_images3.txt
cp /data/kevin/kitti/data_depth_annotated/val/val_laser_images3.txt /data/kevin/kitti/raw_data/2011_09_26/val_conf.txt
cp /data/kevin/kitti/data_depth_annotated/val/val_laser_images3.txt /data/kevin/kitti/raw_data/2011_09_26/val_disp.txt
cp /data/kevin/kitti/data_depth_annotated/train/train_laser_images3.txt /data/kevin/kitti/raw_data/2011_09_26/train_images2.txt
cp /data/kevin/kitti/data_depth_annotated/train/train_laser_images3.txt /data/kevin/kitti/raw_data/2011_09_26/train_images3.txt
cp /data/kevin/kitti/data_depth_annotated/train/train_laser_images3.txt /data/kevin/kitti/raw_data/2011_09_26/train_conf.txt
cp /data/kevin/kitti/data_depth_annotated/train/train_laser_images3.txt /data/kevin/kitti/raw_data/2011_09_26/train_disp.txt

#search and replace the validation file paths 
sed -i 's/data_depth_annotated\/val/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/val_images2.txt
sed -i 's/proj_depth\/groundtruth\/image_03/image_02\/data/g' /data/kevin/kitti/raw_data/2011_09_26/val_images2.txt

sed -i 's/data_depth_annotated\/val/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/val_images3.txt
sed -i 's/proj_depth\/groundtruth\/image_03/image_03\/data/g' /data/kevin/kitti/raw_data/2011_09_26/val_images3.txt

sed -i 's/data_depth_annotated\/val/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/val_conf.txt
sed -i 's/proj_depth\/groundtruth\/image_03/conf/g' /data/kevin/kitti/raw_data/2011_09_26/val_conf.txt
sed -i 's/.png/_confidence.png/g' /data/kevin/kitti/raw_data/2011_09_26/val_conf.txt

sed -i 's/data_depth_annotated\/val/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/val_disp.txt
sed -i 's/proj_depth\/groundtruth\/image_03/disp/g' /data/kevin/kitti/raw_data/2011_09_26/val_disp.txt
sed -i 's/.png/_disparity.png/g' /data/kevin/kitti/raw_data/2011_09_26/val_disp.txt

#search and replace the validation train paths 
sed -i 's/data_depth_annotated\/train/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/train_images2.txt
sed -i 's/proj_depth\/groundtruth\/image_03/image_02\/data/g' /data/kevin/kitti/raw_data/2011_09_26/train_images2.txt

sed -i 's/data_depth_annotated\/train/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/train_images3.txt
sed -i 's/proj_depth\/groundtruth\/image_03/image_03\/data/g' /data/kevin/kitti/raw_data/2011_09_26/train_images3.txt

sed -i 's/data_depth_annotated\/train/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/train_conf.txt
sed -i 's/proj_depth\/groundtruth\/image_03/conf/g' /data/kevin/kitti/raw_data/2011_09_26/train_conf.txt
sed -i 's/.png/_confidence.png/g' /data/kevin/kitti/raw_data/2011_09_26/train_conf.txt

sed -i 's/data_depth_annotated\/train/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/train_disp.txt
sed -i 's/proj_depth\/groundtruth\/image_03/disp/g' /data/kevin/kitti/raw_data/2011_09_26/train_disp.txt
sed -i 's/.png/_disparity.png/g' /data/kevin/kitti/raw_data/2011_09_26/train_disp.txt

#add the silent label:
awk '{printf("%s %d\n",$0,0)}' /data/kevin/kitti/raw_data/2011_09_26/val_images3.txt > /data/kevin/kitti/raw_data/2011_09_26/val_images3_labeled.txt
awk '{printf("%s %d\n",$0,0)}' /data/kevin/kitti/raw_data/2011_09_26/val_images2.txt > /data/kevin/kitti/raw_data/2011_09_26/val_images2_labeled.txt
awk '{printf("%s %d\n",$0,0)}' /data/kevin/kitti/raw_data/2011_09_26/val_disp.txt > /data/kevin/kitti/raw_data/2011_09_26/val_disp_labeled.txt
awk '{printf("%s %d\n",$0,0)}' /data/kevin/kitti/raw_data/2011_09_26/val_conf.txt > /data/kevin/kitti/raw_data/2011_09_26/val_conf_labeled.txt

awk '{printf("%s %d\n",$0,0)}' /data/kevin/kitti/raw_data/2011_09_26/train_images3.txt > /data/kevin/kitti/raw_data/2011_09_26/train_images3_labeled.txt
awk '{printf("%s %d\n",$0,0)}' /data/kevin/kitti/raw_data/2011_09_26/train_images2.txt > /data/kevin/kitti/raw_data/2011_09_26/train_images2_labeled.txt
awk '{printf("%s %d\n",$0,0)}' /data/kevin/kitti/raw_data/2011_09_26/train_disp.txt > /data/kevin/kitti/raw_data/2011_09_26/train_disp_labeled.txt
awk '{printf("%s %d\n",$0,0)}' /data/kevin/kitti/raw_data/2011_09_26/train_conf.txt > /data/kevin/kitti/raw_data/2011_09_26/train_conf_labeled.txt
