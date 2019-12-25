#!/usr/bin/env python
# Checks for consistency between Karpathy and Blumme files

__author__ = 'aishwarya'

import os
import re
import sys
from argparse import ArgumentParser
from unidecode import unidecode

sys.path.append('../utils')
from file_utils import create_dir
from json_wrapper import load_json, save_json


def process(sentence):
    # Special case handling of & nbsp;
    sentence = re.sub('&nbsp;', '', sentence)

    # Special case handling of <3
    sentence = re.sub('&lt;3', '<3', sentence)

    # Special case handling of n't
    sentence = re.sub('n\'t', ' n\'t', sentence)

    # Special case handling of &
    sentence = re.sub('&amp;', '&', sentence)

    # Special case handling of 's
    sentence = re.sub('\'s', '\' s', sentence)

    # Remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    tokens = sentence.split(' ')
    return tuple([token.lower() for token in tokens if len(token) > 0])


def check_image_lists(args):
    karpathy_image_list = os.path.join(args.dataset_dir, 'image_lists/flickr30k_karpathy_all_imlist.txt')
    with open(karpathy_image_list) as handle:
        karpathy_images = set(handle.read().splitlines())

    blumme_image_list = os.path.join(args.dataset_dir, 'image_lists/flickr30k_blumme_all_imlist.txt')
    with open(blumme_image_list) as handle:
        blumme_images = set(handle.read().splitlines())

    if karpathy_images == blumme_images:
        print 'Images are equal'
    else:
        karpathy_extra = karpathy_images.difference(blumme_images)
        blumme_extra = blumme_images.difference(karpathy_images)
        print '# Extra images in karpathy list =', len(karpathy_extra)
        print '# Extra images in blumme list =', len(blumme_extra)
        _ = raw_input()
        # print 'Extra images in karpathy list =', karpathy_extra
        # _ = raw_input()
        # print 'Extra images in blumme list =', blumme_extra


def check_sentences(args):
    karpathy_sentences = load_json(os.path.join(args.dataset_dir, 'annotations/flickr30k_karpathy_sentences.json'))
    blumme_sentences = load_json(os.path.join(args.dataset_dir, 'annotations/flickr30k_blumme_sentences.json'))
    common_images = set(karpathy_sentences.keys()).intersection(set(blumme_sentences.keys()))
    for image_id in common_images:
        this_image_karpathy_sentences = karpathy_sentences[image_id]
        this_image_blumme_sentences = blumme_sentences[image_id]
        if len(this_image_karpathy_sentences) != len(this_image_blumme_sentences):
            print 'Unequal number of sentences for image', image_id
            print '# Karpathy sentences =', len(this_image_karpathy_sentences)
            print '# Blumme sentences =', len(this_image_blumme_sentences)
            _ = raw_input()
        else:
            processed_karpathy_sentences = [process(s) for s in this_image_karpathy_sentences]
            processed_blumme_sentences = [process(s) for s in this_image_blumme_sentences]
            if set(processed_karpathy_sentences) != set (processed_blumme_sentences):
                print 'Sentences don\'t match for image', image_id
                for idx in range(len(processed_blumme_sentences)):
                    if processed_karpathy_sentences[idx] != processed_blumme_sentences[idx]:
                        print 'Karpathy sentence :\n\t', this_image_karpathy_sentences[idx]
                        print '\nBlumme sentence :\n\t', this_image_blumme_sentences[idx]
                        print 'Karpathy sentence :\n\t', processed_karpathy_sentences[idx]
                        print '\nBlumme sentence :\n\t', processed_blumme_sentences[idx]
                _ = raw_input()
    print 'Sentence checks complete'


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to Flickr30K dataset')
    args = arg_parser.parse_args()
    check_image_lists(args)
    check_sentences(args)
