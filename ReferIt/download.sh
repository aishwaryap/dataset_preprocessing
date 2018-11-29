#!/bin/bash

TARGET_PATH=$1

if [ "$TARGET_PATH" != "" ]; then
    echo 'Downloading to directory '.$TARGET_PATH
    mkdir -p $TARGET_PATH
    mkdir -p $TARGET_PATH/ReferitData
    wget -O $TARGET_PATH/ReferitData/ReferitData.zip http://tamaraberg.com/referitgame/ReferitData.zip
    unzip $TARGET_PATH/ReferitData/ReferitData.zip -d $TARGET_PATH/ReferitData/
    mkdir -p $TARGET_PATH/ImageCLEF/
    wget -O $TARGET_PATH/ImageCLEF/referitdata.tar.gz http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referitdata.tar.gz
    tar -xzvf $TARGET_PATH/ImageCLEF/referitdata.tar.gz -C $TARGET_PATH/ImageCLEF/

    wget -O $TARGET_PATH/referit_edgeboxes_top100.zip http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referit_edgeboxes_top100.zip
    unzip $TARGET_PATH/referit_edgeboxes_top100.zip -d $TARGET_PATH/
    rm $TARGET_PATH/referit_edgeboxes_top100.zip
else
    echo 'Syntax: '.$0.' <directory_to_download_to>'
fi
