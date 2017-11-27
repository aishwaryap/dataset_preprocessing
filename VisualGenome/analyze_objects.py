# Some analysis to find a suitable set of hypernyms from which we can identify indoor objects

import os
from nltk.corpus import wordnet as wn
from argparse import ArgumentParser

__author__ = 'aishwarya'


def analyze_objects(args):
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


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    args = arg_parser.parse_args()
    analyze_objects(args)