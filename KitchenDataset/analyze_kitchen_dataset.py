#!/usr/bin/env python

__author__ = 'aishwarya'

from argparse import ArgumentParser
from operator import itemgetter
import gensim


def analyze_annotations(args):
    input_file_handle = open(args.annotations_text_file)
    text = input_file_handle.read().strip()
    input_file_handle.close()

    text_corpus = gensim.corpora.textcorpus.TextCorpus()
    document = list(text_corpus.preprocess_text(text))
    dictionary = gensim.corpora.dictionary.Dictionary([document])

    bow = dictionary.doc2bow(document)  # Bag of words
    descending_sorted_bow = sorted(bow, key=itemgetter(1))
    for (token_id, count) in descending_sorted_bow:
        print dictionary.get(token_id), ':', count


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    # arg_parser.add_argument('--raw-annotations-file', type=str, required=True,
    #                         help='Natural Language descriptions of kitchen dataset')
    arg_parser.add_argument('--annotations-text-file', type=str, required=True,
                            help='Text file with only descriptions from Kitchen dataset')
    args = arg_parser.parse_args()
    analyze_annotations(args)