#!/usr/bin/python

from argparse import ArgumentParser
import os
import numpy as np

__author__ = 'aishwarya'


def split_region_features(args):
    orig_features_dir = os.path.join(args.dataset_dir, 'classifiers/data/features/train/')
    target_features_dir = os.path.join(args.dataset_dir, 'indoor/region_features/train/')
    regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt')
    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')
    for batch_num in range(args.num_train_batches):
        print 'Reading batch', str(batch_num)
        orig_filename = orig_features_dir + str(batch_num) + '.csv'
        batch_features = np.loadtxt(orig_filename, dtype=np.float, delimiter=',')
        batch_regions = regions[batch_num * args.batch_size: (batch_num + 1) * args.batch_size]
        for (idx, region) in enumerate(batch_regions):
            region_features_file = target_features_dir + str(region)
            np.savetxt(region_features_file, batch_features[idx, :], delimiter=',')
            if idx % 100 == 0:
                print idx, 'regions processed'

    orig_features_dir = os.path.join(args.dataset_dir, 'classifiers/data/features/test/')
    target_features_dir = os.path.join(args.dataset_dir, 'indoor/region_features/test/')
    regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt')
    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')
    for batch_num in range(args.num_test_batches):
        print 'Reading batch', str(batch_num)
        orig_filename = orig_features_dir + str(batch_num) + '.csv'
        batch_features = np.loadtxt(orig_filename, dtype=np.float, delimiter=',')
        batch_regions = regions[batch_num * args.batch_size: (batch_num + 1) * args.batch_size]
        for (idx, region) in enumerate(batch_regions):
            region_features_file = target_features_dir + str(region)
            np.savetxt(region_features_file, batch_features[idx, :], delimiter=',')
            if idx % 100 == 0:
                print idx, 'regions processed'


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--num-train-batches', type=int, default=22,
                            help='Number of train batches')
    arg_parser.add_argument('--num-test-batches', type=int, default=8,
                            help='Number of test batches')
    arg_parser.add_argument('--batch-size', type=int, default=65536,
                            help='Batch size')
