#!/usr/bin/env python

__author__ = 'aishwarya'

from argparse import ArgumentParser
import json
import gensim
import os


# Creates a text file that has only the descriptions (one per line) for easy text analysis
def create_descriptions_text_file(args):
    input_file_handle = open(args.raw_annotations_file)
    output_file_handle = open(args.annotations_text_file, 'w')
    annotations_dict = json.loads(input_file_handle.read().strip())
    for image_name in annotations_dict:
        descriptions = annotations_dict[image_name]
        for description in descriptions:
            output_file_handle.write(description.strip() + '\n')
    output_file_handle.close()
    input_file_handle.close()


# Creates a gensim dictionary from text descriptions for text analysis
def create_annotations_dict(args):
    input_file_handle = open(args.annotations_text_file)
    text = input_file_handle.read().strip()
    input_file_handle.close()

    text_corpus = gensim.corpora.textcorpus.TextCorpus()
    document = list(text_corpus.preprocess_text(text))
    dictionary = gensim.corpora.dictionary.Dictionary([document])

    dictionary.save(args.annotations_dict_pkl)
    input_file_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--raw-annotations-file', type=str, required=True,
                            help='JSON file of Natural Language descriptions of kitchen dataset')
    arg_parser.add_argument('--annotations-text-file', type=str, required=True,
                            help='Text file with only descriptions from Kitchen dataset')
    arg_parser.add_argument('--annotations-dict-pkl', type=str, required=True,
                            help='Pickle with gensim dictionary of descriptions')
    args = arg_parser.parse_args()

    if not os.path.isfile(args.annotations_text_file):
        create_descriptions_text_file(args)
        print 'Created annotations text file'
    else:
        print 'Annotations text file exists. Not recreating ...'

    if not os.path.isfile(args.annotations_dict_pkl):
        create_annotations_dict(args)
        print 'Created annotations dict'
    else:
        print 'Annotations dict exists. Not recreating ...'
