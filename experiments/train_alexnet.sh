#!/bin/bash

log_file_name="./results/alexnet/"
log_file_name+="`date +%Y-%d-%m-%H-%M-%S`";
log_file_name+="_log.txt"

# fine-tuning
# /home/p307k07/Code/caffe/build/tools/caffe train -solver ./model/alexnet/solver.prototxt -weights /home/p307k07/Code/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

# without fine-tuning
/home/p307k07/Code/caffe/build/tools/caffe train -solver ./model/alexnet/solver.prototxt &> $log_file_name
