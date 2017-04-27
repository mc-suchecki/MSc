#!/bin/bash

echo Waiting...; while ps -p 24427 > /dev/null; do sleep 2; done; kill -9 20622 && ./train_neural_net.sh
