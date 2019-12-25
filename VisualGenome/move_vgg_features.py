#!/usr/bin/python
# Reorganize VGG features and store as HDF5 files

__author__ = 'aishwarya'

from argparse import ArgumentParser
import numpy as np
import os
import csv
import h5py


def add_features(orig_region_list, num_features_files, orig_features_dir, target_region_lists, target_hdf5_files):
    region_idx = 0
    for features_file_idx in range(num_features_files):
        features_file = os.path.join(orig_features_dir, str(features_file_idx) + '.csv')
        file_handle = open(features_file, 'r')
        reader = csv.reader(file_handle, delimiter=',')

        for row in reader:
            feature = [float(x) for x in row]
            region_id = orig_region_list[region_idx]
            if region_id in target_region_lists['train']:
                target_hdf5_files['train'].create_dataset(region_id, (4096,), dtype=np.float32)
                target_hdf5_files['train'][region_id][:] = feature
            elif region_id in target_region_lists['val']:
                target_hdf5_files['val'].create_dataset(region_id, (4096,), dtype=np.float32)
                target_hdf5_files['val'][region_id][:] = feature
            elif region_id in target_region_lists['test']:
                target_hdf5_files['test'].create_dataset(region_id, (4096,), dtype=np.float32)
                target_hdf5_files['test'][region_id][:] = feature
            region_idx += 1

            if region_idx % 1000 == 0:
                print 'Processed region', region_idx

            if region_idx >= len(orig_region_list):
                # This will happen because the last CSV file ends with a bunch of extra rows of zeros
                # because of the way it was created
                break


def main(args):
    target_region_lists = dict()
    image_list_file = os.path.join(args.dataset_dir, 'split', args.split_dir, 'train_regions.txt')
    with open(image_list_file) as handle:
        target_region_lists['train'] = set(handle.read().splitlines())
    image_list_file = os.path.join(args.dataset_dir, 'split', args.split_dir, 'val_regions.txt')
    with open(image_list_file) as handle:
        target_region_lists['val'] = set(handle.read().splitlines())
    image_list_file = os.path.join(args.dataset_dir, 'split', args.split_dir, 'test_regions.txt')
    with open(image_list_file) as handle:
        target_region_lists['test'] = set(handle.read().splitlines())

    target_hdf5_files = dict()
    features_filename = os.path.join(args.dataset_dir, 'vgg_features', args.split_dir, 'train_features.hdf5')
    features_file = h5py.File(features_filename, 'w')
    target_hdf5_files['train'] = features_file
    features_filename = os.path.join(args.dataset_dir, 'vgg_features', args.split_dir, 'val_features.hdf5')
    features_file = h5py.File(features_filename, 'w')
    target_hdf5_files['val'] = features_file
    features_filename = os.path.join(args.dataset_dir, 'vgg_features', args.split_dir, 'test_features.hdf5')
    features_file = h5py.File(features_filename, 'w')
    target_hdf5_files['test'] = features_file

    orig_region_list_file = os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt')
    with open(orig_region_list_file) as handle:
        orig_region_list = handle.read().splitlines()

    orig_features_dir = os.path.join(args.dataset_dir, 'classifiers/data/features/train/')

    add_features(orig_region_list, 22, orig_features_dir, target_region_lists, target_hdf5_files)

    orig_region_list_file = os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt')
    with open(orig_region_list_file) as handle:
        orig_region_list = handle.read().splitlines()

    orig_features_dir = os.path.join(args.dataset_dir, 'classifiers/data/features/test/')

    add_features(orig_region_list, 8, orig_features_dir, target_region_lists, target_hdf5_files)
    
    for key, file_handle in target_hdf5_files.items():
        file_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--split-dir', type=str, required=True,
                            help='Subdirectory under split and vgg_features')
    args = arg_parser.parse_args()
    main(args)