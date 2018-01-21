set -e -x
experimentname=$1
lr=$2
maxiter=$3
weightDecay=$4
momentum=$5
stepiter=$6
stride1=$7
stride2=$8
stride3=${9}
numout1=${10}
numout2=${11}
numout3=${12}
numout4=${13}
stride0=${14}
stepiter=2000
#./c.sh 32 $experimentname $lr $maxiter $weightDecay $momentum $stepiter $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0

#./c2.sh 128 $experimentname $lr $maxiter $weightDecay $momentum $stepiter $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
./c2.sh 256 $experimentname $lr $maxiter $weightDecay $momentum $stepiter $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
# ./c.sh 64 $experimentname $lr $maxiter $weightDecay $momentum $stepiter $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
