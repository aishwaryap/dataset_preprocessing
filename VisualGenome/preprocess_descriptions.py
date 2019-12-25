#!/usr/bin/python
# Write out normalized region descriptions to reduce memory and time in experiments

from argparse import ArgumentParser
from gensim.models import KeyedVectors
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

    if args.gensim_word_vectors_dir is not None:
        print('Reading word vectors ...')
        word_vectors_file = os.path.join(args.gensim_word_vectors_dir, 'word2vec/GoogleNews-vectors-negative300.bin')
        word_vectors = KeyedVectors.load_word2vec_format(word_vectors_file, binary=True)
    else:
        word_vectors = None

    num_regions_processed = 0
    for line in region_graphs_file:
        region = json.loads(line.strip())
        region_id = region['region_id']
        description = region['phrase']
        normalized_description = normalize_string(description, word_vectors)
        region_descriptions_file.write(str(region_id) + ',' + normalized_description + '\n')

        num_regions_processed += 1
        if num_regions_processed % 10000 == 0:
            print(num_regions_processed, 'regions processed')

    region_descriptions_file.close()


def remove_oov_words(args):
    orig_region_descriptions_filename = os.path.join(args.dataset_dir, 'region_descriptions.csv')
    new_region_descriptions_filename = os.path.join(args.dataset_dir, 'in_vocab_region_descriptions.csv')

    print('Reading word vectors ...')
    word_vectors_file = os.path.join(args.gensim_word_vectors_dir, 'word2vec/GoogleNews-vectors-negative300.bin')
    word_vectors = KeyedVectors.load_word2vec_format(word_vectors_file, binary=True)

    region_list_files = {
        'all': os.path.join(args.dataset_dir, 'indoor/allowed_regions.txt')
    }
    for split in ['policy_train', 'policy_val', 'policy_test']:
        region_list_files[split] = os.path.join(args.dataset_dir, 'split/predicate_novelty', split + '_regions.txt')

    print('Reading regions ...')
    regions = dict()
    vocab = dict()
    for split, region_list_file in region_list_files.items():
        with open(region_list_file) as handle:
            regions[split] = set(handle.read().splitlines())
        vocab[split] = dict()

    print('Processing regions ...')

    num_regions_processed = 0

    input_handle = open(orig_region_descriptions_filename)
    output_handle = open(new_region_descriptions_filename, 'w')

    reader = csv.reader(input_handle, delimiter=',')
    writer = csv.writer(output_handle, delimiter=',')
    for row in reader:
        region = row[0]
        tokens = row[1].split('_')
        in_vocab_tokens = [token for token in tokens if in_vocab(token, word_vectors)]
        output_row = [region, '_'.join(in_vocab_tokens)]
        writer.writerow(output_row)
        for split in regions.keys():
            if region in regions[split]:
                for token in in_vocab_tokens:
                    if token not in vocab[split]:
                        vocab[split][token] = 1
                    else:
                        vocab[split][token] += 1
        num_regions_processed += 1
        if num_regions_processed % 10000 == 0:
            print(num_regions_processed, 'regions processed')

    input_handle.close()
    output_handle.close()

    for split in vocab.keys():
        print('Writing vocab of', split)
        vocab_file = os.path.join(args.dataset_dir, 'vocab/predicate_novelty', split + '_vocab.txt')
        with open(vocab_file, 'w') as handle:
            print('Sorting vocab ..')
            words_with_frequency = list(vocab[split].items())
            words_with_frequency.sort(key=operator.itemgetter(1), reverse=True)
            words = [word for (word, freq) in words_with_frequency]
            print('Writing vocab to file ...')
            handle.write('\n'.join(words))

    print('Complete.')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--gensim-word-vectors-dir', type=str, default=None,
                            help='Path to Gensim word vectors')
    arg_parser.add_argument('--preprocess-descriptions', action='store_true', default=False,
                            help='Specify this for initial preprocessing of descriptions')
    arg_parser.add_argument('--remove-oov-words', action='store_true', default=False,
                            help='Remove OOV words and store vocab of each split')
    args = arg_parser.parse_args()

    if args.preprocess_descriptions:
        preprocess_descriptions(args)

    if args.remove_oov_words:
        remove_oov_words(args)
