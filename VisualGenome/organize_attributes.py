#!/usr/bin/python
# Creates
#   - a list of unique objects and attributes
#   - a mapping from regions to attributes associated with objects in it

from argparse import ArgumentParser
import json
import sys
import csv
from utils import *
sys.path.append('../utils')
from json_wrapper import *

__author__ = 'aishwarya'


def create_contents_list(args):
    attributes = load_json(os.path.join(args.dataset_dir, 'attributes.json'))
    print 'Loaded attributes ...'
    indexed_attributes = dict()
    for image in attributes:
        indexed_attributes[image['image_id']] = image['attributes']
    print 'Indexed attributes ...'

    region_graphs_filename = os.path.join(args.dataset_dir, 'region_graphs.txt')
    region_graphs_file = open(region_graphs_filename)

    region_objects_file = open(os.path.join(args.dataset_dir, 'region_objects.csv'), 'w')
    region_objects_writer = csv.writer(region_objects_file, delimiter=',')
    region_attributes_file = open(os.path.join(args.dataset_dir, 'region_attributes.csv'), 'w')
    region_attributes_writer = csv.writer(region_attributes_file, delimiter=',')
    region_synsets_file = open(os.path.join(args.dataset_dir, 'region_synsets.csv'), 'w')
    region_synsets_writer = csv.writer(region_synsets_file, delimiter=',')
    num_regions_processed = 0

    objects_file = open(os.path.join(args.dataset_dir, 'objects_list.txt'), 'w')
    synsets_file = open(os.path.join(args.dataset_dir, 'synsets_list.txt'), 'w')
    attributes_file = open(os.path.join(args.dataset_dir, 'attributes_list.txt'), 'w')

    for line in region_graphs_file:
        region = json.loads(line.strip())
        image_id = region['image_id']
        region_objects = [object['object_id'] for object in region['objects']]
        image_attributes = indexed_attributes[image_id]
        relevant_attributes = [object for object in image_attributes if object['object_id'] in region_objects]
        object_names = list()
        attribute_names = list()
        synset_names = list()
        for object in relevant_attributes:
            if 'names' in object:
                object_names += object['names']
            if 'attributes' in object:
                attribute_names += object['attributes']
            if 'synsets' in object:
                synset_names += object['synsets']
        objects_file.write('\n'.join([unicode(s).encode('ascii',errors='ignore') for s in object_names]) + '\n')
        attributes_file.write('\n'.join([unicode(s).encode('ascii', errors='ignore') for s in attribute_names]) + '\n')
        synsets_file.write('\n'.join([unicode(s).encode('ascii', errors='ignore') for s in synset_names]) + '\n')
        objects_row = [region['region_id']] + object_names
        objects_row_ascii = [unicode(s).encode('ascii',errors='ignore') for s in objects_row]
        region_objects_writer.writerow(objects_row_ascii)
        attributes_row = [region['region_id']] + attribute_names
        attributes_row_ascii = [unicode(s).encode('ascii',errors='ignore') for s in attributes_row]
        region_attributes_writer.writerow(attributes_row_ascii)
        synsets_row = [region['region_id']] + synset_names
        synsets_row_ascii = [unicode(s).encode('ascii', errors='ignore') for s in synsets_row]
        region_synsets_writer.writerow(synsets_row_ascii)
        num_regions_processed += 1
        if num_regions_processed % 10000 == 0:
            print num_regions_processed, 'regions processed ...'

    region_graphs_file.close()
    region_objects_file.close()
    region_attributes_file.close()
    region_synsets_file.close()

    objects_file.close()
    attributes_file.close()
    synsets_file.close()

    print 'Complete ...'


# In the region attributes/objects file, check that the list of attributes/objects per row is unique
def make_region_contents_unique(input_filename, output_filename, normalize=True):
    input_file = open(input_filename)
    reader = csv.reader(input_file, delimiter=',')
    output_file = open(output_filename, 'w')
    writer = csv.writer(output_file, delimiter=',')

    for row in reader:
        contents_list = row[1:]
        if normalize:
            new_row = [row[0]] + list(set([normalize_string(content_item) for content_item in contents_list
                                       if len(normalize_string(content_item)) > 0]))
        else:
            new_row = [row[0]] + list(set([content_item.strip() for content_item in contents_list
                                           if len(content_item.strip()) > 0]))
        writer.writerow(new_row)

    input_file.close()
    output_file.close()
    print 'Fixed file', input_filename


# For making list of objects and attributes unique
def make_list_unique(input_filename, output_filename, normalize=True):
    input_file = open(input_filename)
    contents_list = input_file.read().split('\n')
    input_file.close()

    if normalize:
        contents = set([normalize_string(content_item) for content_item in contents_list
                        if len(normalize_string(content_item)) > 0])
    else:
        contents = set([content_item.strip() for content_item in contents_list
                        if len(content_item.strip()) > 0])
    contents = list(contents)
    contents.sort()

    output_file = open(output_filename, 'w')
    output_file.write('\n'.join(contents))
    output_file.close()
    print 'Fixed file', input_filename


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')

    arg_parser.add_argument('--create-contents-list', action="store_true", default=False,
                            help='Create files that have lists of region objects and attributes')
    arg_parser.add_argument('--make-region-contents-unique', action="store_true", default=False,
                            help='Remove duplicates from each row in region objects and attributes files')
    arg_parser.add_argument('--make-contents-list-unique', action="store_true", default=False,
                            help='Remove duplicates from list of objects and attributes')
    args = arg_parser.parse_args()

    if args.create_contents_list:
        create_contents_list(args)

    if args.make_region_contents_unique:
        input_file = os.path.join(args.dataset_dir, 'region_objects.csv')
        output_file = os.path.join(args.dataset_dir, 'region_objects_unique.csv')
        make_region_contents_unique(input_file, output_file)

        input_file = os.path.join(args.dataset_dir, 'region_attributes.csv')
        output_file = os.path.join(args.dataset_dir, 'region_attributes_unique.csv')
        make_region_contents_unique(input_file, output_file)

        input_file = os.path.join(args.dataset_dir, 'region_synsets.csv')
        output_file = os.path.join(args.dataset_dir, 'region_synsets_unique.csv')
        make_region_contents_unique(input_file, output_file, False)

    if args.make_contents_list_unique:
        input_file = os.path.join(args.dataset_dir, 'objects_list.txt')
        output_file = os.path.join(args.dataset_dir, 'objects_list_unique.txt')
        make_list_unique(input_file, output_file)

        input_file = os.path.join(args.dataset_dir, 'attributes_list.txt')
        output_file = os.path.join(args.dataset_dir, 'attributes_list_unique.txt')
        make_list_unique(input_file, output_file)

        input_file = os.path.join(args.dataset_dir, 'synsets_list.txt')
        output_file = os.path.join(args.dataset_dir, 'synsets_list_unique.txt')
        make_list_unique(input_file, output_file, False)
