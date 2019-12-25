#!/usr/bin/env python

__author__ = 'aishwarya'

import os
import re
import csv
import numpy as np
from argparse import ArgumentParser


def check_resized_imcrop(args):
    resized_imcrop_dir = os.path.join(args.dataset_dir, 'resized_imcrop')
    resized_imcrop_files = os.listdir(resized_imcrop_dir)
    mask_file_list_file = '/u/aish/Downloads/mask_list.txt'
    with open(mask_file_list_file) as handle:
        mask_files = handle.read().splitlines()
    resized_imcrop_regions = [re.sub('.png', '', f) for f in resized_imcrop_files]
    mask_regions = [re.sub('.mat', '', f) for f in mask_files]
    if set(resized_imcrop_regions) == set(mask_regions):
        print('Regions match')
    else:
        print('Extra regions in resized_imcrop =',
              set(resized_imcrop_regions).difference(mask_regions))
        print('Extra regions in mask =',
              set(mask_regions).difference(resized_imcrop_regions))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to ReferIt dataset')
    args = arg_parser.parse_args()
    check_resized_imcrop(args)