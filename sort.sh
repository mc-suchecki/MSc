#!/bin/bash

# sorts the photos by resolution

for image in *.jpg;
    do res=$(identify -format %wx%h\\n $image);
    mkdir -p $res;
    mv $image $res;
done