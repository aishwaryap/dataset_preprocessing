#!/bin/bash

if [ "$1" != "" ]; then
    echo 'Entering directory '.$1
    cd $1

    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
    unzip images.zip
    rm images.zip

    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
    unzip images2.zip
    rm images2.zip

    wget http://visualgenome.org/static/data/dataset/image_data.json.zip
    unzip image_data.json.zip
    rm image_data.json.zip

    wget http://visualgenome.org/static/data/dataset/region_descriptions.json.zip
    unzip region_descriptions.json.zip
    rm region_descriptions.json.zip

    wget http://visualgenome.org/static/data/dataset/objects.json.zip
    unzip objects.json.zip
    rm objects.json.zip

    wget http://visualgenome.org/static/data/dataset/attributes.json.zip
    unzip attributes.json.zip
    rm attributes.json.zip

    wget http://visualgenome.org/static/data/dataset/synsets.json.zip
    unzip synsets.json.zip
    rm synsets.json.zip

    wget http://visualgenome.org/static/data/dataset/object_alias.txt
else
    echo 'Syntax: '.$0.' <directory_to_download_to>'
fi


