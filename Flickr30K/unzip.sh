#!/bin/bash

TARGET_PATH=$1
DOWNLOADED_DATASET=$TARGET_PATH"/flickr30k_images.tar.gz"

if [ -f $DOWNLOADED_DATASET ]; then
    tar -xzvf $DOWNLOADED_DATASET
    rm $DOWNLOADED_DATASET
else
    echo 'Syntax: '.$0.' <path_to_directory_containing_downloaded_dataset>'
fi
