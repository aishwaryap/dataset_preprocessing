#!/usr/bin/python

from argparse import ArgumentParser
import os
import sys
import json
sys.path.append('../utils')
from json_wrapper import *

__author__ = 'aishwarya'


# Split the region graphs json file into a file where each line is a json of one image,
# and only with necessary fields
def split_region_graphs(args):
    print 'Reading input file ...'
    region_graphs = load_json(os.path.join(args.dataset_dir, 'region_graphs.json'))
    image_data = load_json(os.path.join(args.dataset_dir, 'image_data.json'))
    print 'Finished reading input file'

    image_sizes = dict()
    for image in image_data:
        image_sizes[image['image_id']] = dict()
        image_sizes[image['image_id']]['width'] = image['width']
        image_sizes[image['image_id']]['height'] = image['height']
    print 'Organized image sizes ...'

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

            # Correct for overflows in region bboxes
            minimal_region['x'] = max(minimal_region['x'], 0)
            minimal_region['x'] = min(minimal_region['x'], image_sizes[region['image_id']]['width'])
            minimal_region['y'] = max(minimal_region['y'], 0)
            minimal_region['y'] = min(minimal_region['y'], image_sizes[region['image_id']]['height'])

            minimal_region['width'] = region['width']
            minimal_region['height'] = region['height']
            minimal_region['phrase'] = region['phrase']
            minimal_region['objects'] = list()

            for object in region['objects']:
                minimal_object = dict()
                minimal_object['object_id'] = object['object_id']
                minimal_object['x'] = object['x']
                minimal_object['y'] = object['y']
                minimal_object['w'] = object['w']
                minimal_object['h'] = object['h']
                minimal_object['synsets'] = object['synsets']
                minimal_region['objects'].append(minimal_object)
                num_objects += 1

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
