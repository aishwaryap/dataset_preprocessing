#!/usr/bin/env python
# Convert Glove vectors to word2vec format

__author__ = 'aishwarya'

from argparse import ArgumentParser
from gensim.scripts.glove2word2vec import glove2word2vec

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--input-file', type=str, required=True,
                            help='Path to input Glove vectors file')
    arg_parser.add_argument('--output-file', type=str, required=True,
                            help='Path to output word2vec vectors file')
    args = arg_parser.parse_args()
    glove2word2vec(args.input_file, args.output_file)

