#!/usr/bin/python
# Store training features and labels from classifiers in a nice format

from argparse import ArgumentParser
import os
import csv
import random
import sys

__author__ = 'aishwarya'


# An intermediate step where we write the list of labels (objects and attributes), and regions relevant for
# training classifiers, given a threshold of label frequency
def save_labels_and_regions(args):
    region_counts = dict()

    object_counts_filename = os.path.join(args.dataset_dir, 'region_objects_stats.csv')
    objects = list()
    input_file = open(object_counts_filename)
    reader = csv.reader(input_file, delimiter=',')
    print 'Indexing object counts ...'
    for row in reader:
        region_counts[row[0]] = int(row[1])
        objects.append(row[0])
    input_file.close()
    print 'Indexed object counts'

    attribute_counts_filename = os.path.join(args.dataset_dir, 'region_attributes_stats.csv')
    attributes = list()
    input_file = open(attribute_counts_filename)
    reader = csv.reader(input_file, delimiter=',')
    print 'Indexing attribute counts ...'
    for row in reader:
        region_counts[row[0]] = int(row[1])
        attributes.append(row[0])
    input_file.close()
    print 'Indexed attribute counts'

    print 'Indexing regions ...'
    region_content_filename = os.path.join(args.dataset_dir, 'region_objects_unique.csv')
    input_file = open(region_content_filename)
    regions_per_content_item = dict()
    reader = csv.reader(input_file, delimiter=',')
    for row in reader:
        region_id = row[0]
        for content_item in row[1:]:
            if region_counts[content_item] >= args.frequency_threshold:
                if content_item not in regions_per_content_item:
                    regions_per_content_item[content_item] = list()
                regions_per_content_item[content_item].append(region_id)

    region_content_filename = os.path.join(args.dataset_dir, 'region_attributes_unique.csv')
    input_file = open(region_content_filename)
    reader = csv.reader(input_file, delimiter=',')
    for row in reader:
        region_id = row[0]
        for content_item in row[1:]:
            if region_counts[content_item] >= args.frequency_threshold:
                if content_item not in regions_per_content_item:
                    regions_per_content_item[content_item] = list()
                regions_per_content_item[content_item].append(region_id)
    print 'Indexed regions ...'

    relevant_labels = [label for (label, count) in region_counts.items() if count > args.frequency_threshold]
    relevant_labels.sort()
    print 'Identified relevant labels ...'

    relevant_labels = ['_'.join(label.split()) for label in relevant_labels]
    objects = ['_'.join(label.split()) for label in objects]
    attributes = ['_'.join(label.split()) for label in attributes]

    output_filename = os.path.join(args.dataset_dir, 'classifiers/data/label_names.txt')
    output_file = open(output_filename, 'w')
    output_file.write('\n'.join(relevant_labels))
    output_file.close()
    print 'Saved label names ...'

    relevant_objects = [label for label in relevant_labels if label in objects]
    output_filename = os.path.join(args.dataset_dir, 'classifiers/data/object_names.txt')
    output_file = open(output_filename, 'w')
    output_file.write('\n'.join(relevant_objects))
    output_file.close()
    print 'Saved object names ...'

    relevant_attributes = [label for label in relevant_labels if label in attributes]
    output_filename = os.path.join(args.dataset_dir, 'classifiers/data/attribute_names.txt')
    output_file = open(output_filename, 'w')
    output_file.write('\n'.join(relevant_attributes))
    output_file.close()
    print 'Saved attribute names ...'

    relevant_regions = set()
    for content_item in relevant_labels:
        relevant_regions = relevant_regions.union(regions_per_content_item[content_item])
    output_filename = os.path.join(args.dataset_dir, 'classifiers/data/relevant_regions.txt')
    output_file = open(output_filename, 'w')
    output_file.write('\n'.join(relevant_regions))
    output_file.close()
    print 'Saved relevant regions ...'

    print 'Finding a good train-test split ...'
    relevant_regions = list(relevant_regions)
    split_point = int(args.train_fraction * len(relevant_regions))
    good_split_found = False
    num_trials = 0

    # Keep shuffling the list till you find a train-test-split that works
    while not good_split_found:
        random.shuffle(relevant_regions)
        test_set = relevant_regions[split_point:]
        good_split_found = True
        min_regions = sys.maxint
        for label in relevant_labels:
            regions_in_test_set = [region for region in regions_per_content_item[label] if region in test_set]
            min_regions = min(min_regions, len(regions_in_test_set))
            if len(regions_in_test_set) < args.test_frequency_threshold:
                good_split_found = False
        num_trials += 1
        print num_trials, 'trials made, min_regions =', min_regions
    print 'Found a good split. Saving split ...'

    # Broke out of the while loop so current order is good
    train_set = relevant_regions[:split_point]
    test_set = relevant_regions[split_point:]
    output_filename = os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt')
    output_file = open(output_filename, 'w')
    output_file.write('\n'.join(train_set))
    output_file.close()
    output_filename = os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt')
    output_file = open(output_filename, 'w')
    output_file.write('\n'.join(test_set))
    output_file.close()
    print 'Regions and labels identified ...'


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--frequency-threshold', type=int, default=1000,
                            help='Consider objects and attributes with frequency greater than this')
    arg_parser.add_argument('--train-fraction', type=float, default=0.8,
                            help='Fraction of data used for training')
    arg_parser.add_argument('--test-frequency-threshold', type=int, default=50,
                            help='Number of regions per label needed in test set')

    args = arg_parser.parse_args()
    save_labels_and_regions(args)