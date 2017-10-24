#!/usr/bin/env python

__author__ = 'aishwarya'

from argparse import ArgumentParser
import csv
import json
from utils import *


# Creates a list with full file paths of all images
def create_image_lists(args):
    region_graphs_filename = os.path.join(args.dataset_dir, 'region_graphs.txt')
    region_graphs_file = open(region_graphs_filename)
    image_list_dir = os.path.join(args.dataset_dir, 'image_lists')
    if not os.path.isdir(image_list_dir):
        os.mkdir(image_list_dir)

    image_list_file_num = 0
    image_list_file = os.path.join(image_list_dir, str(image_list_file_num) + '.csv')
    image_list_file_handle = open(image_list_file, 'w')
    image_list_writer = csv.writer(image_list_file_handle, delimiter=',')
    num_regions_processed = 0

    for line in region_graphs_file:
        region = json.loads(line.strip())
        image_path = get_image_path(args.dataset_dir, region['image_id'])
        row = [region['region_id'], image_path, region['x'], region['y'], region['width'], region['height']]
        image_list_writer.writerow(row)

        num_regions_processed += 1
        if num_regions_processed % 65536 == 0:
            print num_regions_processed, 'regions processed ...'

            image_list_file_handle.close()
            image_list_file_num += 1
            image_list_file = os.path.join(image_list_dir, str(image_list_file_num) + '.csv')
            image_list_file_handle = open(image_list_file, 'w')
            image_list_writer = csv.writer(image_list_file_handle, delimiter=',')

    image_list_file_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    args = arg_parser.parse_args()
    create_image_lists(args)
