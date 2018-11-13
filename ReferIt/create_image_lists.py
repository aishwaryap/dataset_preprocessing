#!/usr/bin/env python
# Create image list for extracting VGG features

__author__ = 'aishwarya'

import os
import re
import csv
from argparse import ArgumentParser


def process(input_file, output_file, images_dir):
    with open(input_file) as handle:
        image_ids = set(handle.read().splitlines())
    crop_files = os.listdir(images_dir)
    crop_image_ids = [filename.split('_')[0] for filename in crop_files]
    matching_crops = [crop_files[idx] for idx in range(len(crop_files)) if crop_image_ids[idx] in image_ids]
    with open(output_file, 'w') as handle:
        writer = csv.writer(handle, delimiter=',')
        for crop_file in matching_crops:
            crop_path = os.path.join(images_dir, crop_file)
            writer.writerow([re.sub('.png', '', crop_file), crop_path])


def main(args):
    image_list_dir = os.path.join(args.dataset_dir, 'image_lists')
    if not os.path.isdir(image_list_dir):
        os.mkdir(image_list_dir)
    images_dir = os.path.join(args.dataset_dir, 'resized_imcrop')
    split_dir = 'split'
    for filename in os.listdir(split_dir):
        print 'Processing file', filename
        input_file = os.path.join(split_dir, filename)
        output_file = os.path.join(image_list_dir, filename)
        process(input_file, output_file, images_dir)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to Kitchen dataset')
    args = arg_parser.parse_args()
    main(args)


