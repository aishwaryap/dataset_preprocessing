#!/usr/bin/python
# Store training features and labels from classifiers in a nice format

from argparse import ArgumentParser
import os
import csv
import random
import numpy as np
import math

__author__ = 'aishwarya'


# An intermediate step where we write the list of labels (objects and attributes), and regions relevant for
# training classifiers, given a threshold of label frequency
def organize_labels_and_regions(args):
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
    regions_with_content = dict()
    reader = csv.reader(input_file, delimiter=',')
    for row in reader:
        region_id = row[0]
        regions_with_content[region_id] = row[1:]
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
        regions_with_content[region_id] = row[1:]
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

    print 'Starting search ...'
    # Keep shuffling the list till you find a train-test-split that works : 1 is often enough
    while not good_split_found:
        print 'Still looking...'
        random.shuffle(relevant_regions)
        print '\t Shuffled...'
        test_set = relevant_regions[split_point:]
        print '\t Test set size =', len(test_set)

        good_split_found = True

        print '\t Checking test label frequency  ...'
        label_counts = dict()
        for region in test_set:
            for label in regions_with_content[region]:
                if label in relevant_labels:
                    if label not in label_counts:
                        label_counts[label] = 0
                    label_counts[label] += 1

        print '\t Computed label counts ...'
        min_label_count = min(label_counts.values())
        print '\t min_label_count =', min_label_count

        if min_label_count >= args.test_frequency_threshold:
            break

        num_trials += 1
        print num_trials, 'trials made'

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


# Saves features for the regions in batch_regions. This is a separate function to allow parallelism
def write_batch_features(args):
    if args.verbose:
        print 'Batch', args.batch_num, 'features : Started'

    if args.in_train_set:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt')
    else:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt')
    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')

    start_idx = args.batch_num * args.batch_size
    end_idx = min(start_idx + args.batch_size, len(regions))
    batch_regions = regions[start_idx:end_idx]
    if args.verbose:
        print 'len(batch_regions) = ', len(batch_regions)
        print 'args.batch_num =', args.batch_num
        print 'args.batch_size =', args.batch_size
        print 'start_idx =', start_idx
        print 'len(regions) =', len(regions)
        print 'start_idx + args.batch_size =', start_idx + args.batch_size

    completed_regions = list()

    orig_features_folder = os.path.join(args.dataset_dir, 'regions_vgg_features')
    orig_features_files = [os.path.join(orig_features_folder, f) for f in os.listdir(orig_features_folder)]
    features = np.zeros((args.batch_size, args.feature_vector_size), dtype=float)

    num_files_processed = 0
    for input_filename in orig_features_files:
        input_file = open(input_filename)
        reader = csv.reader(input_file, delimiter=',')
        for row in reader:
            region_id = row[0].split('_')[1]
            if region_id in batch_regions:
                feature = row[1:]
                region_index = batch_regions.index(region_id)
                features[region_index] = feature
                completed_regions.append(region_id)
        input_file.close()
        num_files_processed += 1
        if args.verbose:
            print 'Batch', args.batch_num, 'features :', num_files_processed, 'files processed, dtype =', features.dtype

    if args.in_train_set:
        output_filename = os.path.join(args.dataset_dir, 'classifiers/data/features/train/'
                                       + str(args.batch_num) + '.csv')
    else:
        output_filename = os.path.join(args.dataset_dir, 'classifiers/data/features/test/'
                                       + str(args.batch_num) + '.csv')

    # np.savetxt(output_filename, features)

    with open(output_filename, 'w') as handle:
        writer = csv.writer(handle, delimiter=',')
        for i in range(features.shape[0]):
            writer.writerow(features[i, :])

    print 'Batch', args.batch_num, 'features complete'


# Saves multilabel vectors for the a batch of regions starting at start_idx.
# This is a separate function to allow parallelism
def write_batch_multilabels(args):
    if args.verbose:
        print 'Batch', args.batch_num, 'multilabel : Started'

    if args.in_train_set:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt')
    else:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt')
    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')

    start_idx = args.batch_num * args.batch_size
    end_idx = min(start_idx + args.batch_size, len(regions))
    batch_regions = regions[start_idx:end_idx]
    if args.verbose:
        print 'len(batch_regions) = ', len(batch_regions)
        print 'args.batch_num =', args.batch_num
        print 'args.batch_size =', args.batch_size
        print 'start_idx =', start_idx
        print 'len(regions) =', len(regions)
        print 'start_idx + args.batch_size =', start_idx + args.batch_size

    with open(os.path.join(args.dataset_dir, 'classifiers/data/label_names.txt')) as label_names_file:
        label_names = label_names_file.read().split('\n')

    if args.verbose:
        print 'Batch', args.batch_num, 'multilabel : Indexing regions'
    region_content_filename = os.path.join(args.dataset_dir, 'region_objects_unique.csv')
    input_file = open(region_content_filename)
    regions_with_content = dict()
    reader = csv.reader(input_file, delimiter=',')
    for row in reader:
        region_id = row[0]
        regions_with_content[region_id] = row[1:]

    region_content_filename = os.path.join(args.dataset_dir, 'region_attributes_unique.csv')
    input_file = open(region_content_filename)
    reader = csv.reader(input_file, delimiter=',')
    for row in reader:
        region_id = row[0]
        regions_with_content[region_id] = row[1:]
    if args.verbose:
        print 'Batch', args.batch_num, 'multilabel : Indexed regions'

    if args.in_train_set:
        output_filename = os.path.join(args.dataset_dir, 'classifiers/data/multilabels/train/'
                                       + str(args.batch_num) + '.csv')
    else:
        output_filename = os.path.join(args.dataset_dir, 'classifiers/data/multilabels/test/'
                                       + str(args.batch_num) + '.csv')
    output_file = open(output_filename, 'w')
    writer = csv.DictWriter(output_file, fieldnames=label_names)

    num_regions_processed = 0
    for region in batch_regions:
        row = dict()
        for label_name in label_names:
            if label_name in regions_with_content[region]:
                row[label_name] = 1
            else:
                row[label_name] = 0
        writer.writerow(row)
        num_regions_processed += 1
        if args.verbose and num_regions_processed % 1 == 0:
            print 'Batch', args.batch_num, 'multilabel :', num_regions_processed, 'regions processed'

    output_file.close()
    print 'Batch', args.batch_num, 'multilabel complete'


# Saves binary vector for the regions in batch_regions for the specified label.
# This is a separate function to allow parallelism
def write_individual_label(args):
    if args.verbose:
        print 'Label', args.label, ' : Started'

    if args.in_train_set:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt')
    else:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt')
    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')

    if args.verbose:
        print 'len(regions) =', len(regions)

    if args.verbose:
        print 'Label', args.label, ' : Indexing regions'
    region_content_filename = os.path.join(args.dataset_dir, 'region_objects_unique.csv')
    input_file = open(region_content_filename)
    regions_with_content = dict()
    reader = csv.reader(input_file, delimiter=',')
    for row in reader:
        region_id = row[0]
        regions_with_content[region_id] = row[1:]

    region_content_filename = os.path.join(args.dataset_dir, 'region_attributes_unique.csv')
    input_file = open(region_content_filename)
    reader = csv.reader(input_file, delimiter=',')
    for row in reader:
        region_id = row[0]
        regions_with_content[region_id] = row[1:]
    if args.verbose:
        print 'Label', args.label, ' : Indexed regions'

    max_batch_num = int(math.ceil(float(len(regions)) / args.batch_size))

    for batch_num in range(max_batch_num):
        start_idx = batch_num * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(regions))
        batch_regions = regions[start_idx:end_idx]
        if args.in_train_set:
            output_dir = os.path.join(args.dataset_dir, 'classifiers/data/binary_labels/train/' + args.label)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            output_filename = os.path.join(output_dir, str(batch_num) + '.csv')
        else:
            output_dir = os.path.join(args.dataset_dir, 'classifiers/data/binary_labels/test/' + args.label)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            output_filename = os.path.join(output_dir, str(batch_num) + '.csv')
        output_file = open(output_filename, 'w')
        label_values = list()
        for region in batch_regions:
            if args.label in regions_with_content[region]:
                label_values.append(1)
            else:
                label_values.append(0)
        writer = csv.writer(output_file)
        writer.writerow(label_values)
        output_file.close()
        if args.verbose:
            print batch_num + 1, 'batches processed ...'

    print 'Label =', args.label, 'complete'


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--verbose', action="store_true", default=False,
                            help='Print debug output')

    # Args to indicate that you need to organize labels and regions, and what is needed for this
    arg_parser.add_argument('--organize-labels-and-regions', action="store_true", default=False,
                            help='Some preprocessing to make actual writing of data faster. ' +
                                 'Also creates a train-test split')
    arg_parser.add_argument('--frequency-threshold', type=int, default=1000,
                            help='Consider objects and attributes with frequency greater than this')
    arg_parser.add_argument('--train-fraction', type=float, default=0.8,
                            help='Fraction of data used for training')
    arg_parser.add_argument('--test-frequency-threshold', type=int, default=50,
                            help='Number of regions per label needed in test set')

    # Select things to write
    arg_parser.add_argument('--write-multilabels', action="store_true", default=False,
                            help='Write multilabel vectors for a batch')
    arg_parser.add_argument('--write-individual-labels', action="store_true", default=False,
                            help='Write label vector for a batch for a label')
    arg_parser.add_argument('--write-features', action="store_true", default=False,
                            help='Write features for a batch')

    # Some options common to writing multilabels, individual labels, or features
    arg_parser.add_argument('--batch-size', type=int, default=65536,
                            help='Number of data points per file (features or labels)')
    arg_parser.add_argument('--batch-num', type=int, default=None,
                            help='Start batch at this index in regions file')
    arg_parser.add_argument('--in-train-set', action="store_true", default=False,
                            help='To distinguish between train and test set')

    arg_parser.add_argument('--label', type=str, default=None,
                            help='Label for --write-individual-label')

    arg_parser.add_argument('--feature-vector-size', type=int, default=4096,
                            help='Dimensionality of feature vectors')

    args = arg_parser.parse_args()

    if args.organize_labels_and_regions:
        organize_labels_and_regions(args)

    if args.write_multilabels:
        if args.batch_num is None or args.in_train_set is None:
            raise RuntimeError('--batch_num and --in-train-set required with --write-multilabels')
        write_batch_multilabels(args)

    if args.write_individual_labels:
        if args.in_train_set is None or args.label is None:
            raise RuntimeError('--batch_num, --label and --in-train-set required with --write-individual-labels')
        write_individual_label(args)

    if args.write_features:
        if args.batch_num is None or args.in_train_set is None:
            raise RuntimeError('--batch_num and --in-train-set required with --write-features')
        write_batch_features(args)
