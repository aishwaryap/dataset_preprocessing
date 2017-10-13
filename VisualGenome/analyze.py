#!/usr/bin/python

__author__='aishwarya'

from argparse import ArgumentParser
import os
import sys
sys.path.append('../utils')
from json_wrapper import *


def analyze(args):
    # Calculate number of regions with only 1 object
    region_graphs = load_json(os.path.join(args.dataset_dir, 'region_graphs.json'))
    all_region_ids = list()
    single_object_region_ids = list()
    num_images_processed = 0
    for image in region_graphs:
        for region in image['regions']:
            all_region_ids.append(region['region_id'])
            if len(region['objects']) == 1:
                single_object_region_ids.append(region['region_id'])
        num_images_processed += 1
        if num_images_processed % 1000 == 0:
            print num_images_processed, 'images processed'
    print '# regions =', len(all_region_ids)
    print '# regions with 1 object =', len(single_object_region_ids)
    print 'Fraction of regions with 1 object =', len(single_object_region_ids) / float(len(all_region_ids))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, default=None,
                            help='Path to dataset')
    args = arg_parser.parse_args()
    analyze(args)
