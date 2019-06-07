#!/usr/bin/python
# Write out normalized region descriptions to reduce memory and time in experiments

from argparse import ArgumentParser
import json
from utils import *
import csv
import operator

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


def create_vocab(args):
    region_descriptions_filename = os.path.join(args.dataset_dir, 'region_descriptions.csv')
    if args.split == "all":
        region_list_file = os.path.join(args.dataset_dir, 'indoor/allowed_regions.txt')
    else:
        region_list_file = os.path.join(args.dataset_dir, 'split/predicate_novelty', args.split + '_regions.txt')
    print 'Reading regions ...'
    with open(region_list_file) as handle:
        regions = set(handle.read().splitlines())

    print 'Creating vocab ...'
    vocab = dict()
    num_regions_processed = 0
    with open(region_descriptions_filename) as handle:
        reader = csv.reader(handle, delimiter=',')
        for row in reader:
            if row[0] in regions:
                tokens = row[1].split('_')
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = 1
                    else:
                        vocab[token] += 1
            num_regions_processed += 1
            if num_regions_processed % 100000 == 0:
                print num_regions_processed, 'regions processed'

    vocab_file = os.path.join(args.dataset_dir, 'vocab/predicate_novelty', args.split + '_vocab.txt')
    with open(vocab_file, 'w') as handle:
        print 'Sorting vocab ..'
        words_with_frequency = list(vocab.items())
        words_with_frequency.sort(key=operator.itemgetter(1), reverse=True)
        words = [word for (word, freq) in words_with_frequency]
        print 'Writing vocab to file ...'
        handle.write('\n'.join(words))

    print 'Complete.'


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--split', type=str, default='test',
                            help='"train", "val", "test" or "all"')
    arg_parser.add_argument('--preprocess-descriptions', action='store_true', default=False,
                            help='Specify this for initial preprocessing of descriptions')
    arg_parser.add_argument('--create-vocab', action='store_true', default=False,
                            help='Compute vocabulary of split mentioned in --split')
    args = arg_parser.parse_args()

    if args.preprocess_descriptions:
        preprocess_descriptions(args)

    if args.create_vocab:
        create_vocab(args)
