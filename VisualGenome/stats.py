#!/usr/bin/python
# Computes some useful stats not available on the website

from argparse import ArgumentParser
from operator import itemgetter
import csv
import os

__author__ = 'aishwarya'

# Count how many regions have an object/attribute
def num_regions_per_content(input_filename, output_filename):
    region_counts = dict()
    input_file = open(input_filename)
    reader = csv.reader(input_file, delimiter=',')

    num_regions_processed = 0
    for row in reader:
        for content_item in row[1:]:
            if content_item not in region_counts:
                region_counts[content_item] = 0
            region_counts[content_item] += 1
        num_regions_processed += 1
        if num_regions_processed % 1000000 == 0:
            print num_regions_processed, 'regions processed'

    input_file.close()

    print 'Sorting counts ...'
    sorted_counts = sorted(region_counts.items(), key=itemgetter(1), reverse=True)
    print 'Sorted counts ...'

    output_file = open(output_filename, 'w')
    writer = csv.writer(output_file, delimiter=',')
    for (content_item, count) in sorted_counts:
        writer.writerow(list((content_item, count)))
    output_file.close()

    print 'Analyzed file', input_filename


# Count how many objects/attributes have more than x regions, for a few useful values of x
def above_threshold(input_filename):
    thresholds = [100000, 50000, 10000, 5000, 1000, 500, 100]

    input_file = open(input_filename)
    reader = csv.reader(input_file, delimiter=',')
    print 'Input file :', input_filename
    region_counts = list()
    for row in reader:
        region_counts.append((row[0], int(row[1])))

    for threshold in thresholds:
        print '>', threshold, ' regions :', len([1 for (content_item, count) in region_counts if count > threshold])

    input_file.close()


# How big is the set of regions that contains all occurrences of objects/attributes that occur in > threshold regions
# Basically, if I had to build a classifier for objects/attributes present in > threshold regions, how many regions
# would I need as data points
def regions_with_common_content(region_content_filename, content_list_filename):
    print 'Content list file =', content_list_filename
    print 'Region content file =', region_content_filename

    thresholds = [100000, 50000, 10000, 5000, 1000, 500, 100]

    input_file = open(content_list_filename)
    reader = csv.reader(input_file, delimiter=',')

    print 'Indexing content items ...'
    region_counts = dict()
    for row in reader:
        region_counts[row[0]] = int(row[1])
    input_file.close()
    print 'Indexed content items'

    print 'Indexing regions ...'
    input_file = open(region_content_filename)
    regions_per_content_item = dict()
    min_threshold = min(thresholds)
    reader = csv.reader(input_file, delimiter=',')

    for row in reader:
        region_id = row[0]
        for content_item in row[1:]:
            if region_counts[content_item] >= min_threshold:
                if content_item not in regions_per_content_item:
                    regions_per_content_item[content_item] = list()
                regions_per_content_item[content_item].append(region_id)
    print 'Indexed regions ...'

    for threshold in thresholds:
        relevant_content_items = [content_item for (content_item, count) in region_counts.items() if count > threshold]

        regions = set()
        for content_item in relevant_content_items:
            regions = regions.union(regions_per_content_item[content_item])
        print 'For threshold', threshold, ', num regions needed =', len(regions)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')

    arg_parser.add_argument('--num-regions-per-content', action="store_true", default=False,
                            help='For objects and attributes, how many regions have them')
    arg_parser.add_argument('--above-threshold', action="store_true", default=False,
                            help='# objects and attributes present in more regions than various thresholds')
    arg_parser.add_argument('--regions-with-common-content', action="store_true", default=False,
                            help='How big is the set of regions that contains all occurrences of objects/attributes '
                                 + 'that occur in > threshold regions')

    args = arg_parser.parse_args()

    if args.num_regions_per_content:
        input_file = os.path.join(args.dataset_dir, 'region_objects_unique.csv')
        output_file = os.path.join(args.dataset_dir, 'region_objects_stats.csv')
        num_regions_per_content(input_file, output_file)

        input_file = os.path.join(args.dataset_dir, 'region_attributes_unique.csv')
        output_file = os.path.join(args.dataset_dir, 'region_attributes_stats.csv')
        num_regions_per_content(input_file, output_file)

    if args.above_threshold:
        input_file = os.path.join(args.dataset_dir, 'region_objects_stats.csv')
        above_threshold(input_file)
        input_file = os.path.join(args.dataset_dir, 'region_attributes_stats.csv')
        above_threshold(input_file)

    if args.regions_with_common_content:
        region_content_filename = os.path.join(args.dataset_dir, 'region_objects_unique.csv')
        content_list_filename = os.path.join(args.dataset_dir, 'region_objects_stats.csv')
        regions_with_common_content(region_content_filename, content_list_filename)

        region_content_filename = os.path.join(args.dataset_dir, 'region_attributes_unique.csv')
        content_list_filename = os.path.join(args.dataset_dir, 'region_attributes_stats.csv')
        regions_with_common_content(region_content_filename, content_list_filename)
