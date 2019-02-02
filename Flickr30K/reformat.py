#!/usr/bin/env python
# Create image lists and annotations from
#     flickr30k/dataset.json
#     results.token

__author__ = 'aishwarya'

import os
import re
import csv
import sys
from argparse import ArgumentParser

sys.path.append('../utils')
from file_utils import create_dir
from json_wrapper import load_json, save_json


def write_image_list(images_dir, image_ids, image_list_filename):
    with open(image_list_filename, 'w') as handle:
        writer = csv.writer(handle, delimiter=',')
        for image_id in image_ids:
            row = [image_id, os.path.join(images_dir, image_id + '.jpg')]
            writer.writerow(row)


def main(args):
    image_lists_dir = os.path.join(args.dataset_dir, 'image_lists')
    create_dir(image_lists_dir)

    annotations_dir = os.path.join(args.dataset_dir, 'annotations')
    create_dir(annotations_dir)

    karpathy_file = os.path.join(args.dataset_dir, 'flickr30k/dataset.json')
    karpathy_annotations = load_json(karpathy_file)

    image_lists = dict()
    image_lists['train'] = list()
    image_lists['val'] = list()
    image_lists['test'] = list()

    karpathy_sentences_dict = dict()

    for image in karpathy_annotations['images']:
        image_id = re.sub('.jpg', '', image['filename'])
        image_lists[image['split']].append(image_id)
        karpathy_sentences_dict[image_id] = list()
        for sentence in image['sentences']:
            karpathy_sentences_dict[image_id].append(sentence['raw'])

    karpathy_annotations_output_file = os.path.join(annotations_dir, 'flickr30k_karpathy_sentences.json')
    save_json(karpathy_sentences_dict, karpathy_annotations_output_file)

    images_dir = os.path.join(args.dataset_dir, 'flickr30k_images')
    train_image_list_file = os.path.join(image_lists_dir, 'flickr30k_karpathy_train_imlist.txt')
    write_image_list(images_dir, image_lists['train'], train_image_list_file)
    val_image_list_file = os.path.join(image_lists_dir, 'flickr30k_karpathy_val_imlist.txt')
    write_image_list(images_dir, image_lists['val'], val_image_list_file)
    test_image_list_file = os.path.join(image_lists_dir, 'flickr30k_karpathy_test_imlist.txt')
    write_image_list(images_dir, image_lists['test'], test_image_list_file)
    trainval_image_list_file = os.path.join(image_lists_dir, 'flickr30k_karpathy_trainval_imlist.txt')
    write_image_list(images_dir, image_lists['train'] + image_lists['val'], trainval_image_list_file)
    all_image_list_file = os.path.join(image_lists_dir, 'flickr30k_karpathy_all_imlist.txt')
    write_image_list(images_dir, image_lists['train'] + image_lists['val'] + image_lists['test'],
                     all_image_list_file)

    blumme_annotations_file = os.path.join(args.dataset_dir, 'results_20130124.token')
    blumme_sentences_dict = dict()

    with open(blumme_annotations_file) as handle:
        reader = csv.reader(handle, delimiter='\t')
        for row in reader:
            [ids, sentence] = row
            [image_id, _] = ids.split('.jpg#')
            if image_id not in blumme_sentences_dict:
                blumme_sentences_dict[image_id] = list()
            blumme_sentences_dict[image_id].append(sentence)

    blumme_annotations_output_file = os.path.join(annotations_dir, 'flickr30k_blumme_sentences.json')
    save_json(blumme_sentences_dict, blumme_annotations_output_file)

    all_image_list_file = os.path.join(image_lists_dir, 'flickr30k_blumme_all_imlist.txt')
    write_image_list(images_dir, blumme_sentences_dict.keys(), all_image_list_file)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to Flickr30K dataset')
    args = arg_parser.parse_args()
    main(args)
