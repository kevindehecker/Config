#!/usr/bin/env sh
# Create the imagenet leveldb inputs
set -e -x
name=$1
resolution=$2
testbatchid=$3
lr=$4
maxiter=$5
weightDecay=$6
momentum=$7
stepiter=$8
camname=$9
stride1=${10}
stride2=${11}
stride3=${12}
num_output1=${13}
num_output2=${14}
num_output3=${15}
num_output4=${16}
stride0=${17}
finetune=${18}

if [ $HOSTNAME = fermi ] ; then
	echo "GPU ENABLED"
	gpu=1
else
	echo "GPU DISABLED"
	gpu=0
fi

TOOLS=../../build/tools
DATA=/data/tmp/kevin/data/$camname/$name/$resolution/
  
if $finetune ; then
	echo dummy
else
	rm -rf ./$name$resolution/$testbatchid/*
	rm -rf ./$name$resolution/$testbatchid
	mkdir -p $name$resolution/$testbatchid/snapshots
fi

echo "Copying proto files"	
cp *.prototxt $name$resolution/$testbatchid/
cp train.sh $name$resolution/$testbatchid/
cp create.sh $name$resolution/$testbatchid/

# if $finetune ; then
# 	net=dispcifar10_fine_
# else
	net=dispcifar10_quick_
# fi

echo "Adjusting parameters in prototxt files"
echo "input_dim: $resolution" 
perl -pi -w -e "s/input_dim: 32/input_dim: $resolution/g;" $name$resolution/$testbatchid/${net}deploy.prototxt
echo "solver_mode: $gpu" 
perl -pi -w -e "s/solver_mode: 0/solver_mode: $gpu/g;" $name$resolution/$testbatchid/${net}solver.prototxt
echo "base_lr: $lr" 
perl -pi -w -e "s/base_lr: 0.0001/base_lr: $lr/g;" $name$resolution/$testbatchid/${net}solver.prototxt
echo "max_iter: $maxiter" 
perl -pi -w -e "s/max_iter: 1000/max_iter: $maxiter/g;" $name$resolution/$testbatchid/${net}solver.prototxt
echo "weight decay: $weightDecay" 
perl -pi -w -e "s/weight_decay: 0.00004/weight_decay: $weightDecay/g;" $name$resolution/$testbatchid/${net}solver.prototxt
echo "test_interval: 500" 
perl -pi -w -e "s/test_interval: 10/test_interval: 500/g;" $name$resolution/$testbatchid/${net}solver.prototxt
echo "momentum: $momentum" 
perl -pi -w -e "s/momentum: 0.9/momentum: $momentum/g;" $name$resolution/$testbatchid/${net}solver.prototxt
echo "snapshot: $stepiter" 
perl -pi -w -e "s/snapshot: 500/snapshot: $stepiter/g;" $name$resolution/$testbatchid/${net}solver.prototxt

echo "stride 0: $stride0" 
perl -pi -w -e "s/stride0:/stride: $stride0/g;" $name$resolution/$testbatchid/${net}train.prototxt
perl -pi -w -e "s/stride0:/stride: $stride0/g;" $name$resolution/$testbatchid/${net}test.prototxt
perl -pi -w -e "s/stride0:/stride: $stride0/g;" $name$resolution/$testbatchid/${net}deploy.prototxt
echo "stride 1: $stride1" 
perl -pi -w -e "s/stride1:/stride: $stride1/g;" $name$resolution/$testbatchid/${net}train.prototxt
perl -pi -w -e "s/stride1:/stride: $stride1/g;" $name$resolution/$testbatchid/${net}test.prototxt
perl -pi -w -e "s/stride1:/stride: $stride1/g;" $name$resolution/$testbatchid/${net}deploy.prototxt
echo "stride 2: $stride2" 
perl -pi -w -e "s/stride2:/stride: $stride2/g;" $name$resolution/$testbatchid/${net}train.prototxt
perl -pi -w -e "s/stride2:/stride: $stride2/g;" $name$resolution/$testbatchid/${net}test.prototxt
perl -pi -w -e "s/stride2:/stride: $stride2/g;" $name$resolution/$testbatchid/${net}deploy.prototxt
echo "stride 3: $stride3" 
perl -pi -w -e "s/stride3:/stride: $stride3/g;" $name$resolution/$testbatchid/${net}train.prototxt
perl -pi -w -e "s/stride3:/stride: $stride3/g;" $name$resolution/$testbatchid/${net}test.prototxt
perl -pi -w -e "s/stride3:/stride: $stride3/g;" $name$resolution/$testbatchid/${net}deploy.prototxt

echo "num_output 1: $num_output1" 
perl -pi -w -e "s/num_output1:/num_output: $num_output1/g;" $name$resolution/$testbatchid/${net}train.prototxt
perl -pi -w -e "s/num_output1:/num_output: $num_output1/g;" $name$resolution/$testbatchid/${net}test.prototxt
perl -pi -w -e "s/num_output1:/num_output: $num_output1/g;" $name$resolution/$testbatchid/${net}deploy.prototxt
echo "num_output 2: $num_output2" 
perl -pi -w -e "s/num_output2:/num_output: $num_output2/g;" $name$resolution/$testbatchid/${net}train.prototxt
perl -pi -w -e "s/num_output2:/num_output: $num_output2/g;" $name$resolution/$testbatchid/${net}test.prototxt
perl -pi -w -e "s/num_output2:/num_output: $num_output2/g;" $name$resolution/$testbatchid/${net}deploy.prototxt
echo "num_output 3: $num_output3" 
perl -pi -w -e "s/num_output3:/num_output: $num_output3/g;" $name$resolution/$testbatchid/${net}train.prototxt
perl -pi -w -e "s/num_output3:/num_output: $num_output3/g;" $name$resolution/$testbatchid/${net}test.prototxt
perl -pi -w -e "s/num_output3:/num_output: $num_output3/g;" $name$resolution/$testbatchid/${net}deploy.prototxt
echo "num_output 4: $num_output4" 
perl -pi -w -e "s/num_output4:/num_output: $num_output4/g;" $name$resolution/$testbatchid/${net}train.prototxt
perl -pi -w -e "s/num_output4:/num_output: $num_output4/g;" $name$resolution/$testbatchid/${net}test.prototxt
perl -pi -w -e "s/num_output4:/num_output: $num_output4/g;" $name$resolution/$testbatchid/${net}deploy.prototxt



if  $finetune ;then
	echo dummy
else

	echo "Creating  labels..."
	python makelabels.py /data/tmp/kevin/data/$camname/$name/data.txt /data/tmp/kevin/data/$camname/$name/$resolution/ $testbatchid $name$resolution/$testbatchid/

	echo "Creating  train leveldb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset.bin / $name$resolution/$testbatchid/train.txt $name$resolution/$testbatchid/train_leveldb 1

	echo "Creating  test leveldb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset.bin / $name$resolution/$testbatchid/test.txt $name$resolution/$testbatchid/test_leveldb 1

	echo "Creating  image mean..."
	$TOOLS/compute_image_mean.bin $name$resolution/$testbatchid/train_leveldb $name$resolution/$testbatchid/mean.binaryproto	
fi	


echo "Done, created data for: $name$resolution/$testbatchid"
