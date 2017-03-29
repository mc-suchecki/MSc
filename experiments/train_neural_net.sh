#!/bin/bash

# log_file_name="./results/alexnet/" # AlexNet
log_file_name="./results/oxfordnet/" # OxfordNet
log_file_name+="`date +%Y-%d-%m-%H-%M-%S`";
log_file_name+="_log.txt"

# AlexNet
# fine-tuning
# /home/p307k07/Code/caffe/build/tools/caffe train -solver ./model/alexnet/solver.prototxt -weights /home/p307k07/Code/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel &> $log_file_name

# regular training
# /home/p307k07/Code/caffe/build/tools/caffe train -solver ./model/alexnet/solver.prototxt &> $log_file_name

# training from a snapshot
# /home/p307k07/Code/caffe/build/tools/caffe train -snapshot ./results/alexnet_model_iter_65000.solverstate -solver ./model/alexnet/solver.prototxt &> $log_file_name


# OxfordNet
# regular training
/home/p307k07/Code/caffe/build/tools/caffe train -solver ./model/oxfordnet/solver.prototxt &> $log_file_name
