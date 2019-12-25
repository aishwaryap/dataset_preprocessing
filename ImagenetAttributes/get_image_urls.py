#!/usr/bin/env python

__author__ = 'aishwarya'

from argparse import ArgumentParser
import urllib
import os


def get_image_urls(args):
    file_handle = open(args.image_list_file)
    image_list = file_handle.read().strip().split('\n')
    image_list = [image_name.strip() for image_name in image_list]
    synsets = set([image_name.split('_')[0] for image_name in image_list])

    for synset in synsets:
        print 'Fetching URLs for synset', synset
        url = 'http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=' + synset
        url_handle = urllib.urlopen(url)
        text = url_handle.read()
        lines = text.strip().split('\n')
        image_name_url_pairs = [line.split(' ') for line in lines]
        required_pairs = [[image_name, url] for [image_name, url] in image_name_url_pairs if image_name in image_list]
        output_file = os.path.join(args.image_urls_dir, synset + '.txt')
        output_file_handle = open(output_file, 'w')
        for [image_name, url] in required_pairs:
            output_file_handle.write(image_name + ' ' + url + '\n')
        output_file_handle.close()
        url_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--image-list-file', type=str, required=True,
                            help='List of ImageNet image names')
    arg_parser.add_argument('--image-urls-dir', type=str, required=True,
                            help='Dir for lists of paired ImageNet image names and URLs')
    args = arg_parser.parse_args()
    get_image_urls(args)
