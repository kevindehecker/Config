set -ex
snapdir=/data/kevin/kitti/coarse11/
#datadir=/data/kevin/kitti/raw_data/255
datadir=/data/kevin/kitti/raw_data/2011_09_26

mkdir -p $snapdir

cp solver.prototxt.template solver_autoconfigure.prototxt
echo snapshot_prefix: \"${snapdir}run\" >> solver_autoconfigure.prototxt
echo net: \"./net_autoconfigure.prototxt\" >> solver_autoconfigure.prototxt

cp net.prototxt.template net_autoconfigure.prototxt
sed -i "s~inputfolder~${datadir}~g" net_autoconfigure.prototxt


/home/kevin/caffe/build/tools/caffe train -solver solver_autoconfigure.prototxt -gpu 1 2>&1 | tee ${snapdir}logfile.log

