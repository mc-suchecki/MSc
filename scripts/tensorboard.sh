#!/bin/bash

rm -r ../tmp/train/ 2> /dev/null
rm -r ../tmp/validation/ 2> /dev/null
rm -r ../tmp/test/ 2> /dev/null
tensorboard --logdir=~/Code/MSc/MSc/tmp --reload_interval 1
