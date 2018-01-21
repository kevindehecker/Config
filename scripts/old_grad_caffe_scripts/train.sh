#!/usr/bin/env sh
set -e -x
finetune=${1}

if [ ! -d "snapshots" ]; then
  echo "snapshots dir does not exist"
  return
fi

TOOLS=../../../../build/tools

echo "Training..."
now=$(date +"%T")
echo "Current time : $now"
pwd
hostname
users
pwd
net=dispcifar10_quick_
if $finetune ; then
	# net=dispcifar10_fine_
	GLOG_logtostderr=1 $TOOLS/finetune_net.bin ${net}solver.prototxt snapshots/dispcifar10_quick_iter_25000
else
	# net=dispcifar10_quick_
	GLOG_logtostderr=1 $TOOLS/train_net.bin ${net}solver.prototxt
fi

	
echo "Done."
