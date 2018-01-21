set -e -x
name=$1
resolution=$2
experimentname=$3
lr=$4
maxiter=$5
weightDecay=$6
momentum=$7
stepiter=$8
camname=$9
stride1=${10}
stride2=${11}
stride3=${12}
numout1=${13}
numout2=${14}
numout3=${15}
numout4=${16}
stride0=${17}

now=$(date +"%T")

set -e
echo "Current time : $now"
echo "Commands: $name, $resolution, $experimentname $lr"

#create + preprocess data from matlab
if false; then
	echo "Preprocssing data"
	matlab -r "addpath('/data/tmp/kevin/svn/Afstuderen/StereoVision/Matlab/'); \
	makeCaffeData('/data/tmp/kevin/data/$camname/$name/selected',[$resolution $resolution],1,1)"
fi 

#learn with caffe
if true; then
	echo "Train caffe on a 4-fold cross validation."
	now=$(date +"%T")
	echo "Current time : $now"	
	echo "training networks"
	pwd
	mkdir -p results
	#echo "rm ./$name$resolution -r"
	rm -rf ./$name$resolution
	rm -rf results/$name$resolution${experimentname}*
	mkdir -p $name$resolution	
	./a.sh $name $resolution 1 $experimentname $lr $maxiter $weightDecay $momentum $stepiter $camname $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
	./a.sh $name $resolution 2 $experimentname $lr $maxiter $weightDecay $momentum $stepiter $camname $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
	./a.sh $name $resolution 3 $experimentname $lr $maxiter $weightDecay $momentum $stepiter $camname $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
	./a.sh $name $resolution 4 $experimentname $lr $maxiter $weightDecay $momentum $stepiter $camname $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
elif false; then
	echo "Untar previously trained caffe results."
	mkdir -p $name$resolution/1/
	mkdir -p $name$resolution/2/
	mkdir -p $name$resolution/3/
	mkdir -p $name$resolution/4/
	tar -xzf results/$name/$name$resolution${experimentname}_1.tgz #-C $name$resolution/1/
	tar -xzf results/$name/$name$resolution${experimentname}_2.tgz #-C $name$resolution/2/
	tar -xzf results/$name/$name$resolution${experimentname}_3.tgz #-C $name$resolution/3/
	tar -xzf results/$name/$name$resolution${experimentname}_4.tgz #-C $name$resolution/4/	
else
	echo "nope"
fi 

#test with matlab
if false; then
	now=$(date +"%T")
	echo "Current time : $now"
	echo "Creating data visualisation"
	cp ../../matlab/caffe/testsaveNetwork._m ./testsaveNetwork.m
	perl -pi -w -e "s/maxjj = 10;/maxjj = $maxiter * 0.01;/g;" testsaveNetwork.m
	perl -pi -w -e "s/stepjj = 1;/stepjj = $stepiter * 0.01;/g;" testsaveNetwork.m
	
	if [ $HOSTNAME = fermi ] ; then
		gpu=1
	else
		gpu=0
	fi
	
	net=dispcifar10_quick_
	#net=finetune_
	
	matlab -r "addpath('/data/tmp/kevin/caffe/matlab/caffe/'); testsaveNetwork('$(pwd)/$name$resolution/' , '${net}', $gpu,1)"
	mkdir -p results/$name
	cp $name$resolution/results.mat results/$name/$name$resolution$experimentname.mat
	rm -rf testsaveNetwork.m
fi

#upload data (unnecessary with btsync)
if true; then
	now=$(date +"%T")
	echo "Current time : $now"
	echo "Uploading results" 
	echo "Execusting: scp -BpP100 results/$name$resolution$experimentname*.mat houjebek@dinstech.nl:/home/houjebek/shares/AfstudeerData/results/ &"
	#scp -BpP100 results/$name$resolution$experimentname* houjebek@dinstech.nl:/home/houjebek/shares/AfstudeerData/results/ &
	rsync -avzh --progress -e "ssh -p 22" ./results/ houjebek@dinstech.nl:/home/houjebek/shares/Terantisch/AfstudeerData/fermi/ &
fi
