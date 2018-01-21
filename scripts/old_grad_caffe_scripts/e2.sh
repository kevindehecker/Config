set -e -x
maxiter=10000
weightDecay=$2
momentum=0.9
stepiter=100
experimentname=iets
lr=0.000001   
stride1=2
stride2=2
stride3=2
numout1=32
numout2=32
numout3=64
numout4=64
stride0=1

#./d.sh $experimentname $lr $maxiter $weightDecay $momentum $stepiter $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout$ $stride0


#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn6000.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_6000 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn5500.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_5500 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn5000.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_5000 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn4500.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_4500 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn4000.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_4000 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn3500.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_3500 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3


#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn500.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_500 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn1000.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_1000 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn1500.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_1500 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn2000.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_2000 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn2500.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_2500 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/secondgt/data_trn3000.txt /data/tmp/kevin/data/turnlogs/secondgt/data.txt
#./d.sh trainlr6_3000 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3




#cp /data/tmp/kevin/data/turnlogs/cublicle/data_trn500.txt /data/tmp/kevin/data/turnlogs/cublicle/data.txt
#./d2.sh trainlr6_500 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

#cp /data/tmp/kevin/data/turnlogs/cublicle/data_trn1000.txt /data/tmp/kevin/data/turnlogs/cublicle/data.txt
#./d2.sh trainlr6_1000 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

cp /data/tmp/kevin/data/turnlogs/cublicle/data_trn1500.txt /data/tmp/kevin/data/turnlogs/cublicle/data.txt
./d2.sh trainlr6_1500 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

cp /data/tmp/kevin/data/turnlogs/cublicle/data_trn2000.txt /data/tmp/kevin/data/turnlogs/cublicle/data.txt
./d2.sh trainlr6_2000 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3

cp /data/tmp/kevin/data/turnlogs/cublicle/data_trn2500.txt /data/tmp/kevin/data/turnlogs/cublicle/data.txt
./d2.sh trainlr6_2500 $lr $maxiter 100 0.9 200 4 2 2 32 32 128 128 3









