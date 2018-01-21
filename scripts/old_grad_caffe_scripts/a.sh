set -e -x
name=$1
resolution=$2
testbatchid=$3
experimentname=$4
lr=$5
maxiter=$6
weightDecay=$7
momentum=$8
stepiter=$9
camname=${10}
stride1=${11}
stride2=${12}
stride3=${13}
numout1=${14}
numout2=${15}
numout3=${16}
numout4=${17}
stride0=${18}
finetune=${19}

echo "a.sh, arguments:"
echo "name: $name"
echo "resolution: $resolution"
echo "testbatchid: $testbatchid"
echo "experimentname: $experimentname"
echo "lr: $lr"
echo "maxiter: $maxiter"
echo "weightDecay: $weightDecay"
echo "momentum: $momentum"
echo "stepiter: $stepiter"

echo "Excecuting: ./create_dispcifar.sh $name $resolution $testbatchid $lr $maxiter $weightDecay $momentum"
./create.sh $name $resolution $testbatchid $lr $maxiter $weightDecay $momentum $stepiter $camname $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0 $finetune

echo "Excecuting: cd $name$resolution/$testbatchid"
cd $name$resolution/$testbatchid
./train.sh $finetune 2>&1 | tee snapshots/logfile.log
#python3 ../../getoutput.py snapshots/logfile.log snapshots/logdata
cd ../..


echo "Compressing results..."
mkdir -p results/$name
echo "Excecuting: tar -czvf results/$name$resolution${experimentname}_$testbatchid.tgz $name$resolution/$testbatchid/"
if $finetune; then
	tar -czvf results/$name/ft_$name$resolution${experimentname}_$testbatchid.tgz $name$resolution/$testbatchid/
else
	tar -czvf results/$name/$name$resolution${experimentname}_$testbatchid.tgz $name$resolution/$testbatchid/
fi
#mkdir -p ~/caffelogs
#cp $name$resolution/$testbatchid/snapshots/logfile.log ~/caffelogs/
