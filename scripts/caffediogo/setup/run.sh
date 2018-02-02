set -ex
snapdir=/data/kevin/kitti/coarse3/

mkdir -p $snapdir

cp solver.prototxt.template solver_autoconfigure.prototxt
echo snapshot_prefix: \"${snapdir}run\" >> solver_autoconfigure.prototxt

/home/kevin/caffe/build/tools/caffe train -solver solver_autoconfigure.prototxt -weights /data/kevin/kitti/coarse2/run_iter_1400.caffemodel -gpu 0,1,2,3 2>&1 | tee ${snapdir}logfile.log

