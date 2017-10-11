#!/usr/bin/env python

__author__ = 'aishwarya'

from argparse import ArgumentParser
import os


# Creates a list with full file paths of all images
def create_image_list(args):
    image_names = os.listdir(args.images_dir)
    input_file_handle = open(args.image_list_file, 'w')
    for image_name in image_names:
        input_file_handle.write(os.path.join(args.images_dir, image_name) + '\n')
    input_file_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--images-dir', type=str, required=True,
                            help='Directory with images to list')
    arg_parser.add_argument('--image-list-file', type=str, required=True,
                            help='File to store image list')
    args = arg_parser.parse_args()
    create_image_list(args)
