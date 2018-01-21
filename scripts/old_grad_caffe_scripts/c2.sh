set -e -x
resolution=$1
experimentname=$2
lr=$3
maxiter=$4
weightDecay=$5
momentum=$6
stepiter=$7
stride1=$8
stride2=$9
stride3=${10}
numout1=${11}
numout2=${12}
numout3=${13}
numout4=${14}
stride0=${15}


#parallel --max-procs 2 --ungroup  <<EOF

./b2.sh cublicle $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter turnlogs $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4  $stride0
    

#EOF


	#./b.sh keuken $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter DuoWebcam $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
       	#./b.sh labchairride $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter DuoWebcam $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
       	#./b.sh esa_noSphere $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter DuoWebcam $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
        #./b.sh esa_iss2 $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter DuoWebcam $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
        #./b.sh labwalkslow $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter DuoWebcam $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
        #./b.sh labchairnew $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter DuoWebcam $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
        #./b.sh esa_russia $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter DuoWebcam $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0

        #./b.sh TP_1106 $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter SPHERESfootage $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
        #./b.sh TS051 $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter SPHERESfootage $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0
        #./b.sh TS053 $resolution $experimentname $lr $maxiter $weightDecay $momentum $stepiter SPHERESfootage $stride1 $stride2 $stride3 $numout1 $numout2 $numout3 $numout4 $stride0

