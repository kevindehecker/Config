set -ex
snapdir=/data/kevin/kitti/coarse2/

mkdir -p $snapdir

cp solver.prototxt.template solver_autoconfigure.prototxt
echo snapshot_prefix: \"${snapdir}run\" >> solver_autoconfigure.prototxt

/home/kevin/caffe/build/tools/caffe train -solver solver_autoconfigure.prototxt -gpu 1 2>&1 | tee ${snapdir}logfile.log

