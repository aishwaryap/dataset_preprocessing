# Some analysis to find a suitable set of hypernyms from which we can identify indoor objects

import os
import csv
from nltk.corpus import wordnet as wn
from argparse import ArgumentParser
from threading import Thread

__author__ = 'aishwarya'


def get_selected_synsets(args):
    synsets_file = os.path.join(args.dataset_dir, 'synsets_list_unique.txt')
    hypernyms_file = os.path.join('indoor_objects_hypernyms.txt')
    with open(synsets_file) as handle:
        synsets = [wn.synsets(s.strip())[0] for s in handle.read().split('\n')]
    with open(hypernyms_file) as handle:
        hypernyms = [wn.synsets(s.strip())[0] for s in handle.read().split('\n')]

    hyponyms = list()
    non_hyponyms = list()

    for synset in synsets:
        synset_hypernyms = set([i for i in synset.closure(lambda s: s.hypernyms())])
        if len(synset_hypernyms.intersection(hypernyms)) > 0:
            hyponyms.append(synset.name())
        else:
            non_hyponyms.append(synset.name())

    hyponyms_file = os.path.join(args.dataset_dir, 'selected_synsets.txt')
    with open(hyponyms_file, 'w') as handle:
        handle.write('\n'.join(hyponyms))
    non_hyponyms_file = os.path.join(args.dataset_dir, 'non_selected_synsets.txt')
    with open(non_hyponyms_file, 'w') as handle:
        handle.write('\n'.join(non_hyponyms))


def filter_region_contents(args, target_dir):
    allowed_regions_file = os.path.join(*[args.dataset_dir, target_dir, 'allowed_regions.txt'])
    with open(allowed_regions_file) as handle:
        allowed_regions = set([line.strip() for line in handle.readlines()])

    contents_input_file = os.path.join(args.dataset_dir, args.contents_file)
    contents_output_file = os.path.join(*[args.dataset_dir, target_dir, args.contents_file])

    input_handle = open(contents_input_file)
    output_handle = open(contents_output_file, 'w')
    reader = csv.reader(input_handle, delimiter=',')
    writer = csv.writer(output_handle, delimiter=',')
    num_regions_processed = 0
    for row in reader:
        region_id = row[0]
        if region_id in allowed_regions:
            writer.writerow(row)
        num_regions_processed += 1
        if num_regions_processed % 1000 == 0:
            print args.contents_file, ':', num_regions_processed, 'regions processed ...'
    input_handle.close()
    output_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--get-selected-synsets', action="store_true", default=False,
                            help='Identify which synsets are hyponyms (recursive) of a given hypernym list')

    arg_parser.add_argument('--filter-region-contents', action="store_true", default=False,
                            help='Create copies of any CSV whose first element in a row is a region ID which '
                                 'only contains allowed regions')
    arg_parser.add_argument('--contents-file', type=str, required=True,
                            help='Name of contents file to filter')

    args = arg_parser.parse_args()

    if args.get_selected_synsets:
        get_selected_synsets(args)

    if args.filter_region_contents:
        filter_region_contents(args, 'indoor')
