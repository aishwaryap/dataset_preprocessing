#!/usr/bin/python
# Create 8-way data split for RL for queries

import os
import csv
import numpy as np
from argparse import ArgumentParser


def create_8_way_split(args):
    # Get a list of regions which contain selected synsets
    allowed_regions_file = os.path.join(args.dataset_dir, 'indoor/allowed_regions.txt')
    with open(allowed_regions_file) as handle:
        allowed_regions = handle.read().split('\n')
    print('Read allowed regions')

    # Note: These are based on old stats but that's probably fine because we just want to re-proportion the same
    # set of regions
    relevant_objects_file = os.path.join(args.dataset_dir, 'classifiers/data/object_names.txt')
    with open(relevant_objects_file) as handle:
        relevant_objects = handle.read().split('\n')
    print('Read relevant objects')
    relevant_attributes_file = os.path.join(args.dataset_dir, 'classifiers/data/attribute_names.txt')
    with open(relevant_attributes_file) as handle:
        relevant_attributes = handle.read().split('\n')
    print('Read relevant attributes')

    print('Finding a good train-test split ...')
    start_fraction = dict()
    start_fraction['train'] = 1.0 - args.train_label_fraction - args.val_label_fraction - args.test_label_fraction
    start_fraction['val'] = start_fraction['train'] + args.train_label_fraction
    start_fraction['test'] = start_fraction['val'] + args.val_label_fraction

    objects_split_point = dict()
    attributes_split_point = dict()
    for split in ['train', 'val', 'test']:
        objects_split_point[split] = int(start_fraction[split] * len(relevant_objects))
        attributes_split_point[split] = int(start_fraction[split] * len(relevant_attributes))

    good_split_found = False
    num_trials = 0
    region_sets = dict()
    print('Starting search ...')
    # Keep shuffling the list till you find a train-test-split that works : 1 is often enough
    while not good_split_found:
        print('Still looking...')
        np.random.shuffle(relevant_objects)
        np.random.shuffle(relevant_attributes)
        print('\t Shuffled...')

        labels = dict()
        pretrain_objects = relevant_objects[:objects_split_point['train']]
        pretrain_attributes = relevant_attributes[:attributes_split_point['train']]
        labels['pretrain'] = set(pretrain_objects + pretrain_attributes)

        train_objects = relevant_objects[objects_split_point['train']:objects_split_point['val']]
        train_attributes = relevant_attributes[attributes_split_point['train']:attributes_split_point['val']]
        labels['train'] = set(train_objects + train_attributes)

        val_objects = relevant_objects[objects_split_point['val']:objects_split_point['test']]
        val_attributes = relevant_attributes[attributes_split_point['val']:attributes_split_point['test']]
        labels['val'] = set(val_objects + val_attributes)

        test_objects = relevant_objects[objects_split_point['test']:]
        test_attributes = relevant_attributes[attributes_split_point['test']:]
        labels['test'] = set(test_objects + test_attributes)

        print('\t Building train, val and test sets ')
        region_sets = dict()
        for split in ['pretrain', 'train', 'val', 'test']:
            region_sets[split] = list()

        filenames = [
            os.path.join(args.dataset_dir, 'indoor/region_objects_unique.csv'),
            os.path.join(args.dataset_dir, 'indoor/region_attributes_unique.csv')
        ]
        handles = [open(f) for f in filenames]
        readers = [csv.reader(h) for h in handles]
        num_regions_processed = 0
        while True:
            try:
                region_ids = set()
                contents = set()
                for reader in readers:
                    row = reader.__next__()
                    region_ids.add(row[0])
                    contents = contents.union(row[1:])
                num_regions_processed += 1
                if len(region_ids) > 1:
                    raise RuntimeError('Region contents files not synced. Mismatch at line '
                                       + str(num_regions_processed))
                region_id = list(region_ids)[0]

                assigned = False
                for split in ['test', 'val', 'train']:
                    if not assigned and len(contents.intersection(labels[split])) > 0:
                        region_sets[split].append(region_id)
                        assigned = True
                if not assigned:
                    region_sets['pretrain'].append(region_id)

                if num_regions_processed % 10000 == 0:
                    print('Building train, val and test sets :', num_regions_processed, 'regions processed ...')
            except StopIteration:
                break

        fraction = dict()
        for split in ['train', 'val', 'test']:
            fraction[split] = len(region_sets[split]) / float(len(allowed_regions))
            print('Fraction of', split, ':', fraction[split])

        if (args.min_train_data_fraction <= fraction['train'] <= args.max_train_data_fraction and
                        args.min_val_data_fraction <= fraction['val'] <= args.max_val_data_fraction and
                        args.min_test_data_fraction <= fraction['test'] <= args.max_test_data_fraction):
            good_split_found = True

        num_trials += 1
        print(num_trials, 'trials made')

    print('Found a good split')

    # Broke out of the while loop so current order is good
    for split in ['pretrain', 'train', 'val', 'test']:
        np.random.shuffle(region_sets[split])
        output_filename = os.path.join(args.dataset_dir, 'split/8_way/policy_' + split + '_regions.txt')
        with open(output_filename, 'w') as handle:
            handle.write('\n'.join(region_sets[split]))

        split_point = int(args.subset_train_fraction * len(region_sets[split]))
        train_subset = region_sets[split][:split_point]
        test_subset = region_sets[split][split_point:]

        output_filename = os.path.join(args.dataset_dir, 'split/8_way/policy_' + split + '_classifier_train_regions.txt')
        with open(output_filename, 'w') as handle:
            handle.write('\n'.join(train_subset))

        output_filename = os.path.join(args.dataset_dir, 'split/8_way/policy_' + split + '_classifier_test_regions.txt')
        with open(output_filename, 'w') as handle:
            handle.write('\n'.join(test_subset))

    print('Regions and labels identified ...')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')

    # Args to indicate that you need to organize labels and regions, and what is needed for this
    arg_parser.add_argument('--train-label-fraction', type=float, default=0.4,
                            help='Fraction of labels to be newly seen at policy training time')
    arg_parser.add_argument('--val-label-fraction', type=float, default=0.2,
                            help='Fraction of labels to be newly seen at validation time')
    arg_parser.add_argument('--test-label-fraction', type=float, default=0.15,
                            help='Fraction of labels to be newly seen at test time')
    arg_parser.add_argument('--min-train-data-fraction', type=float, default=0.2,
                            help='Min fraction of data points to go into train set')
    arg_parser.add_argument('--max-train-data-fraction', type=float, default=0.4,
                            help='Max fraction of data points to go into train set')
    arg_parser.add_argument('--min-val-data-fraction', type=float, default=0.1,
                            help='Min fraction of data points to go into val set')
    arg_parser.add_argument('--max-val-data-fraction', type=float, default=0.3,
                            help='Max fraction of data points to go into val set')
    arg_parser.add_argument('--min-test-data-fraction', type=float, default=0.2,
                            help='Min fraction of data points to go into test set')
    arg_parser.add_argument('--max-test-data-fraction', type=float, default=0.4,
                            help='Max fraction of data points to go into test set')
    arg_parser.add_argument('--subset-train-fraction', type=float, default=0.6,
                            help='Within each split, fraction of points going into the "train" fraction')

    args = arg_parser.parse_args()

    create_8_way_split(args)
