#!/usr/bin/env python
# Write images that failed to download into a single script to try again

__author__ = 'aishwarya'

from argparse import ArgumentParser
import os


def find_missing_images(args):
    synset_dirs = os.listdir(args.images_dir)
    output_file_handle = open(args.retry_list, 'w')
    for synset in synset_dirs:
        images_dir = os.path.join(args.images_dir, synset)
        images = os.listdir(images_dir)
        empty_files = [f for f in images if os.stat(os.path.join(images_dir, f)).st_size == 0]
        urls_file = os.path.join(args.image_urls_dir, synset + '.txt')
        input_file_handle = open(urls_file)
        for line in input_file_handle:
            line = line.strip()
            if line.split(' ')[0] in empty_files:
                output_line = synset + '/' + line
                output_file_handle.write(output_line + '\n')
        input_file_handle.close()
    output_file_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--image-urls-dir', type=str, required=True,
                            help='Dir of lists of paired ImageNet image names and URLs')
    arg_parser.add_argument('--images-dir', type=str, required=True,
                            help='Dir of downloaded images')
    arg_parser.add_argument('--retry-list', type=str, required=True,
                            help='List of paired images and URLs to retry downloading')
    args = arg_parser.parse_args()
    find_missing_images(args)
