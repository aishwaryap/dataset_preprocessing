#!/usr/bin/python

__author__='aishwarya'

from argparse import ArgumentParser
import os
import sys
import json
sys.path.append('../utils')
from json_wrapper import *


# Check if an object is in a region
def is_object_in_region(region, object):
    if object['x'] < region['x']:
        return False
    if object['x'] + object['width'] > region['x'] + region['width']:
        return False
    if object['y'] < region['y']:
        return False
    if object['y'] + object['height'] > region['y'] + region['height']:
        return False
    return True


# Split the region graphs json file into a file where each line is a json of one image,
# and only with necessary fields
def split_region_graphs(args):
    print 'Reading input file ...'
    region_graphs = load_json(os.path.join(args.dataset_dir, 'region_graphs.json'))
    print 'Finished reading input file'
    output_file = os.path.join(args.dataset_dir, 'region_graphs.txt')
    output_file_handle = open(output_file, 'w')
    num_images_processed = 0
    num_regions = 0
    num_objects = 0
    num_objects_outside_region = 0
    num_regions_with_single_object = 0
    for image in region_graphs:
        for region in image['regions']:
            # Store only necessary info
            minimal_region = dict()
            minimal_region['image_id'] = image['image_id']
            minimal_region['region_id'] = region['region_id']
            minimal_region['x'] = region['x']
            minimal_region['y'] = region['y']
            minimal_region['width'] = region['width']
            minimal_region['height'] = region['height']
            minimal_region['phrase'] = region['phrase']
            minimal_region['objects'] = list()
            for object in region['objects']:
                # Check that the object is actually in the region
                if is_object_in_region(region, object):
                    minimal_object = dict()
                    minimal_object['object_id'] = object['object_id']
                    minimal_object['x'] = object['x']
                    minimal_object['y'] = object['y']
                    minimal_object['width'] = object['width']
                    minimal_object['height'] = object['height']
                    minimal_region['objects'].append(minimal_object)
                    num_objects += 1
                else:
                    num_objects_outside_region += 1
            if len(minimal_region['objects']) == 1:
                num_regions_with_single_object += 1
            region_str = json.dumps(minimal_region)
            output_file_handle.write(region_str + '\n')
            num_regions += 1
        num_images_processed += 1
        if num_images_processed % 1000 == 0:
            print num_images_processed, 'images processed'
    output_file_handle.close()
    print 'Complete'
    print 'num_images_processed =', num_images_processed
    print 'num_regions =', num_regions
    print 'num_objects =', num_objects
    print 'num_objects_outside_region =', num_objects_outside_region
    print 'num_regions_with_single_object =', num_regions_with_single_object


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    args = arg_parser.parse_args()
    split_region_graphs(args)

