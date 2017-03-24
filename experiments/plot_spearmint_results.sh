#!/bin/bash

scp msucheck@mion.elka.pw.edu.pl:/home/mion/s/19/msucheck/*.txt /home/mc/Code/Python/MSc/experiments/results/spearmint/
python plot_caffe_learning_curves.py /home/mc/Code/Python/MSc/experiments/results/spearmint/*.txt
