# Some analysis to find a suitable set of hypernyms from which we can identify indoor objects

import csv
from utils import *
from argparse import ArgumentParser

__author__ = 'aishwarya'


def analyze_objects(args):
    synsets_file = os.path.join(args.dataset_dir, 'synsets_list_unique.txt')
    hypernyms_file = os.path.join('indoor_objects_hypernyms.txt')
    with open(synsets_file) as handle:
        synset_names = handle.read().split('\n')
        synsets = [get_synset(s) for s in synset_names if get_synset(s) is not None]
    with open(hypernyms_file) as handle:
        synset_names = handle.read().split('\n')
        hypernyms = [get_synset(s) for s in synset_names if get_synset(s) is not None]

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

    region_synsets_filename = os.path.join(args.dataset_dir, 'region_synsets_unique.csv')
    allowed_regions = list()
    num_regions_processed = 0
    with open(region_synsets_filename) as region_synsets_file:
        reader = csv.reader(region_synsets_file)
        for row in reader:
            region_id = row[0]
            synsets = [get_synset(s) for s in row[1:] if get_synset(s) is not None]
            if len(hyponyms.intersection(synsets)) > 0:
                allowed_regions.append(region_id)
            num_regions_processed += 1
            if num_regions_processed % 100000 == 0:
                print num_regions_processed, 'regions processed'
    allowed_regions_file = os.path.join(args.dataset_dir, 'allowed_regions.txt')
    with open(allowed_regions_file, 'w') as handle:
        handle.write('\n'.join(allowed_regions))
    print 'Computed allowed regions'


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    args = arg_parser.parse_args()
    analyze_objects(args)
