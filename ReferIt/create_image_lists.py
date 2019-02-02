#!/usr/bin/env python
# Create image list for extracting VGG features

__author__ = 'aishwarya'

import os
import re
import csv
import numpy as np
from argparse import ArgumentParser


def create_edgebox_image_list(args):
    images_dir = os.path.join(args.dataset_dir, 'ImageCLEF/images/')
    edgebox_proposals_dir = os.path.join(args.dataset_dir, 'referit_edgeboxes_top100')
    image_list_file = os.path.join(args.dataset_dir, 'image_lists/referit_edgeboxes_imlist.csv')
    image_list_handle = open(image_list_file, 'w')
    image_list_writer = csv.writer(image_list_handle, delimiter=',')

    for filename in os.listdir(edgebox_proposals_dir):
        # print 'Processing file', filename
        image_id = re.sub('.txt', '', filename)
        image_file = os.path.join(images_dir, image_id + '.jpg')
        bboxes = np.loadtxt(os.path.join(edgebox_proposals_dir, filename)).astype(np.uint8)
        if bboxes.shape == (4,):
            bboxes = bboxes.reshape(1, 4)
        for idx in range(bboxes.shape[0]):
            bbox_id = image_id + '_' + str(idx)
            [x_min, y_min, width, height] = bboxes[idx]
            # Output row is [region ID, image file, x_min, y_min, width, height]
            output_row = [bbox_id, image_file, x_min, y_min, width, height]
            image_list_writer.writerow(output_row)

    image_list_handle.close()


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
        # print 'Processing file', filename
        input_file = os.path.join(split_dir, filename)
        output_file = os.path.join(image_list_dir, filename)
        process(input_file, output_file, images_dir)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to ReferIt dataset')
    args = arg_parser.parse_args()
    main(args)
    create_edgebox_image_list(args)

