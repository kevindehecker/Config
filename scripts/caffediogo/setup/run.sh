set -ex
snapdir=/data/kevin/kitti/fine1/
#datadir=/data/kevin/kitti/raw_data/255
datadir=/data/kevin/kitti/raw_data/2011_09_26

mkdir -p $snapdir

cp solver.prototxt.template solver_autoconfigure.prototxt
echo snapshot_prefix: \"${snapdir}run\" >> solver_autoconfigure.prototxt
echo net: \"./net_autoconfigure.prototxt\" >> solver_autoconfigure.prototxt

cp net.prototxt.template net_autoconfigure.prototxt
sed -i "s~inputfolder~${datadir}~g" net_autoconfigure.prototxt


/home/kevin/caffe/build/tools/caffe train -solver solver_autoconfigure.prototxt -weights /data/kevin/kitti/coarse11/run_iter_140000.caffemodel -gpu 1 2>&1 | tee ${snapdir}logfile.log

