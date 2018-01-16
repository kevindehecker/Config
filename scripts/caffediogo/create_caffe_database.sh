
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
sed -i 's/proj_depth\/groundtruth\/image_02/image_03\/data/g' /data/kevin/kitti/raw_data/2011_09_26/val_images3.txt

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
sed -i 's/proj_depth\/groundtruth\/image_02/image_03\/data/g' /data/kevin/kitti/raw_data/2011_09_26/train_images3.txt

sed -i 's/data_depth_annotated\/train/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/train_conf.txt
sed -i 's/proj_depth\/groundtruth\/image_03/conf/g' /data/kevin/kitti/raw_data/2011_09_26/train_conf.txt
sed -i 's/.png/_confidence.png/g' /data/kevin/kitti/raw_data/2011_09_26/train_conf.txt

sed -i 's/data_depth_annotated\/train/raw_data\/2011_09_26/g' /data/kevin/kitti/raw_data/2011_09_26/train_disp.txt
sed -i 's/proj_depth\/groundtruth\/image_03/disp/g' /data/kevin/kitti/raw_data/2011_09_26/train_disp.txt
sed -i 's/.png/_disparity.png/g' /data/kevin/kitti/raw_data/2011_09_26/train_disp.txt
