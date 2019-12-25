#!/bin/bash

TARGET_PATH=$1

if [ "$TARGET_PATH" != "" ]; then
    echo 'Downloading to directory '.$TARGET_PATH
    mkdir -p $TARGET_PATH
    wget -O $TARGET_PATH/vgg7k.prototxt https://people.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/imagenet7k/imagenet7k_fc7.prototxt
    wget -O $TARGET_PATH/vgg7k.caffemodel https://people.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/imagenet7k/imagenet7k_fc7_train_iter_50000.caffemodel
else
    echo 'Syntax: '.$0.' <directory_to_download_to>'
fi
