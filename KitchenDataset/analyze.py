#!/usr/bin/env python

__author__ = 'aishwarya'

from argparse import ArgumentParser
from operator import itemgetter
import gensim
import os
import json


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


# This is to check whether images in both ImageNET and Kitchen directory have annotations
def check_images(args):
    imagenet_images_dir = os.path.join(args.images_root_dir, 'ImageNET')
    imagenet_image_names = [image_name.split('.')[0] for image_name in os.listdir(imagenet_images_dir)]
    kitchen_images_dir = os.path.join(args.images_root_dir, 'Kitchen')
    kitchen_image_names = [image_name.split('.')[0] for image_name in os.listdir(kitchen_images_dir)]

    input_file_handle = open(args.raw_annotations_file)
    annotations_dict = json.loads(input_file_handle.read().strip())
    input_file_handle.close()

    imagenet_images_used = [image_name for image_name in imagenet_image_names if image_name in annotations_dict.keys()]
    kitchen_images_used = [image_name for image_name in kitchen_image_names if image_name in annotations_dict.keys()]
    print '# ImageNet images =', len(imagenet_image_names)
    print '# ImageNet images used =', len(imagenet_images_used)
    print '# kitchen images =', len(kitchen_image_names)
    print '# kitchen images used =', len(kitchen_images_used)


if __name__ == '__main__':
    arg_parser = ArgumentParser()

    arg_parser.add_argument('--analyze-annotations', action="store_true", default=False)
    arg_parser.add_argument('--annotations-text-file', type=str, default=None,
                            help='Text file with only descriptions from Kitchen dataset')

    arg_parser.add_argument('--check-images', action="store_true", default=False)
    arg_parser.add_argument('--raw-annotations-file', type=str, required=True,
                            help='JSON file of Natural Language descriptions of kitchen dataset')
    arg_parser.add_argument('--images-root-dir', type=str, default=None,
                            help='Path to dataset/images in the downloaded Kitchen dataset (should end at images)')

    args = arg_parser.parse_args()

    if args.analyze_annotations:
        if args.annotations_text_file is None:
            raise RuntimeError('--annotations-text-file ia required with --analyze-annotations')
        analyze_annotations(args)

    if args.check_images:
        if args.images_root_dir is None:
            raise RuntimeError('--images-root-dir, --raw-annotations-file are required with --check-images')
        check_images(args)