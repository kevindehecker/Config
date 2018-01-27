set -ex
snapdir=/data/kevin/kitti/cheapSparse10_3/

mkdir -p $snapdir

cp solver.prototxt.template solver_autoconfigure.prototxt
echo snapshot_prefix: \"${snapdir}run\" >> solver_autoconfigure.prototxt

/home/kevin/caffe/build/tools/caffe train -solver solver_autoconfigure.prototxt -weights model_norm_abs_100k.caffemodel -gpu 0 2>&1 | tee ${snapdir}logfile.log

