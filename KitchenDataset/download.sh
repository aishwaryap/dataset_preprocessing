#!/bin/bash

TARGET_PATH=$1

if [ "$TARGET_PATH" != "" ]; then
    echo 'Downloading to directory '.$TARGET_PATH
    mkdir -p $TARGET_PATH
    wget -O $TARGET_PATH/Kitchen.tar.gz http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/Kitchen.tar.gz
    tar -xzvf $TARGET_PATH/Kitchen.tar.gz -C $TARGET_PATH
    rm $TARGET_PATH/Kitchen.tar.gz
else
    echo 'Syntax: '.$0.' <directory_to_download_to>'
fi
