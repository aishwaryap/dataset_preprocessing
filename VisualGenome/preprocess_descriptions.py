#!/usr/bin/python
# Write out normalized region descriptions to reduce memory and time in experiments

from argparse import ArgumentParser
import json
from utils import *

__author__ = 'aishwarya'


def preprocess_descriptions(args):
    region_graphs_filename = os.path.join(args.dataset_dir, 'region_graphs.txt')
    region_graphs_file = open(region_graphs_filename)
    region_descriptions_filename = os.path.join(args.dataset_dir, 'region_descriptions.csv')
    region_descriptions_file = open(region_descriptions_filename, 'w')

    with open(os.path.join(args.dataset_dir, 'classifiers/data/label_names.txt')) as label_names_file:
        label_names = label_names_file.read().split('\n')

    num_regions_processed = 0
    for line in region_graphs_file:
        region = json.loads(line.strip())
        region_id = region['region_id']
        description = region['phrase']
        normalized_description = normalize_string(description)
        region_descriptions_file.write(str(region_id) + ',' + normalized_description + '\n')

        num_regions_processed += 1
        if num_regions_processed % 10000 == 0:
            print num_regions_processed, 'regions processed'

    region_descriptions_file.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    args = arg_parser.parse_args()
    preprocess_descriptions(args)