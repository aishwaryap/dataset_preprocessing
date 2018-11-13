#!/bin/bash

TARGET_PATH=$1

if [ "$TARGET_PATH" != "" ]; then
    echo 'Downloading to directory '.$TARGET_PATH
    mkdir -p $TARGET_PATH
    mkdir -p $TARGET_PATH/glove
    cd $TARGET_PATH/glove
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    rm glove.6B.zip
    cd ../word2vec
    gzip -d GoogleNews-vectors-negative300.bin.gz
else
    echo 'Syntax: '.$0.' <directory_to_download_to>'
fi
