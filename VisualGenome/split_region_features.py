#!/usr/bin/python

from argparse import ArgumentParser
import os
import numpy as np

__author__ = 'aishwarya'


def split_region_features(args):
    if args.in_train_set:
        orig_features_dir = os.path.join(args.dataset_dir, 'classifiers/data/features/train/')
        target_features_dir = os.path.join(args.dataset_dir, 'indoor/region_features/train/')
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt')
    else:
        orig_features_dir = os.path.join(args.dataset_dir, 'classifiers/data/features/test/')
        target_features_dir = os.path.join(args.dataset_dir, 'indoor/region_features/test/')
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt')

    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')

    print 'Reading batch', str(args.batch_num)
    orig_filename = orig_features_dir + str(args.batch_num) + '.csv'
    batch_features = np.loadtxt(orig_filename, dtype=np.float, delimiter=',')
    batch_regions = regions[args.batch_num * args.batch_size: min((args.batch_num + 1) * args.batch_size, len(regions))]
    for (idx, region) in enumerate(batch_regions):
        region_features_file = target_features_dir + str(region)
        np.savetxt(region_features_file, batch_features[idx, :], delimiter=',')
        if idx % 100 == 0:
            print idx, 'regions processed'


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--batch-num', type=int, required=True,
                            help='Batch num')
    arg_parser.add_argument('--batch-size', type=int, default=65536,
                            help='Batch size')
    arg_parser.add_argument('--in-train-set', action="store_true", default=False,
                            help='To distinguish between train and test set')

    args = arg_parser.parse_args()
    split_region_features(args)