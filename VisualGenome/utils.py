#!/usr/bin/python

import os

__author__ = 'aishwarya'


# This dataset has 2 image folders. This is a util to search both and find the image
# verify=True does a more robust check that the image actually exists
def get_image_path(dataset_dir, image_id, verify=False):
    path1 = os.path.join(*[dataset_dir, 'VG_100K', str(image_id) + '.jpg'])
    path2 = os.path.join(*[dataset_dir, 'VG_100K_2', str(image_id) + '.jpg'])
    if os.path.isfile(path1):
        return path1
    else:
        if verify:
            if not os.path.isfile(path2):
                return None
        return path2

