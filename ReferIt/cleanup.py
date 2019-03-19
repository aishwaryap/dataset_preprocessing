#!/usr/bin/env python
# Moving resized_imcrop files into their directory
# These were created in the parent directory because I used + instead of os.path.join

__author__ = 'aishwarya'

import os
import re
import h5py
import traceback


def move():
    parent_dir = '/scratch/cluster/aish/ReferIt'
    target_dir = '/scratch/cluster/aish/ReferIt/resized_imcrop'
    match_regex = 'resized_imcrop[0-9]+_[0-9]+.png'
    for filename in os.listdir(parent_dir):
        if re.match(match_regex, filename):
            orig_file = os.path.join(parent_dir, filename)
            new_file = os.path.join(target_dir, filename)
            os.rename(orig_file, new_file)


def move_edgeboxes(image_list_file, orig_dir, target_dir):
    with open(image_list_file) as handle:
        image_list = handle.read().splitlines()
    for filename in os.listdir(orig_dir):
        if re.sub('.hdf5', '', filename) in image_list:
            orig_file = os.path.join(orig_dir, filename)
            new_file = os.path.join(target_dir, filename)
            os.rename(orig_file, new_file)


def rename():
    target_dir = '/scratch/cluster/aish/ReferIt/resized_imcrop'
    for filename in os.listdir(target_dir):
        orig_file = os.path.join(target_dir, filename)
        new_file = os.path.join(target_dir, re.sub('resized_imcrop', '', filename))
        os.rename(orig_file, new_file)


def fix_edgebox_hdf5():
    features_dir = '/u/aish/Documents/ReferIt_link/resnet_fcn_features/edgeboxes/'
    dataset_name_prefix = 'scratch/cluster/aish/ReferIt/image_lists/referit_edgeboxes/'
    bad_files_file = '/u/aish/Documents/temp/problematic_edgebox_files.txt'
    bad_files_handle = open(bad_files_file, 'w')
    files_to_clean = [f for f in os.listdir(features_dir)]

    for filename in files_to_clean:
        print('Processing file', filename)

        orig_file = os.path.join(features_dir, filename)
        try:
            orig_handle = h5py.File(orig_file, 'r')
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except:
            print(traceback.format_exc())
            bad_files_handle.write(filename + '\n')
            continue    # Problem opening file

        new_dataset_name = re.sub('.hdf5', '', filename)
        if new_dataset_name in orig_handle.keys():
            orig_handle.close()
            continue    # File is already fixed

        temp_file = os.path.join(features_dir, 'temp.hdf5')
        temp_handle = h5py.File(temp_file, 'w')
        orig_dataset_name = os.path.join(dataset_name_prefix, new_dataset_name)
        temp_handle.copy(source=orig_handle[orig_dataset_name], dest=new_dataset_name)
        temp_handle.close()
        os.remove(orig_file)
        os.rename(temp_file, orig_file)

    bad_files_handle.close()


if __name__ == '__main__':
    fix_edgebox_hdf5()

    # image_list_file = '/u/aish/Documents/ReferIt_link_T5/split/referit_test_imlist.txt'
    # orig_dir = '/u/aish/Documents/ReferIt_link/resnet_fcn_features/edgeboxes/'
    # target_dir = '/u/aish/Documents/ReferIt_link_T5/resnet_fcn_features/edgeboxes/'
    # move_edgeboxes(image_list_file, orig_dir, target_dir)
    # image_list_file = '/u/aish/Documents/ReferIt_link_T5/split/referit_train_imlist.txt'
    # orig_dir = '/u/aish/Documents/ReferIt_link_T5/resnet_fcn_features/edgeboxes/'
    # target_dir = '/u/aish/Documents/ReferIt_link/resnet_fcn_features/edgeboxes/'
    # move_edgeboxes(image_list_file, orig_dir, target_dir)
    # image_list_file = '/u/aish/Documents/ReferIt_link_T5/split/referit_val_imlist.txt'
    # orig_dir = '/u/aish/Documents/ReferIt_link/resnet_fcn_features/edgeboxes/'
    # target_dir = '/u/aish/Documents/ReferIt_link_T5/resnet_fcn_features/edgeboxes/'
    # move_edgeboxes(image_list_file, orig_dir, target_dir)
