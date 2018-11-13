#!/usr/bin/env python
# Moving resized_imcrop files into their directory
# These were created in the parent directory because I used + instead of os.path.join

__author__ = 'aishwarya'

import os
import re


def move():
    parent_dir = '/scratch/cluster/aish/ReferIt'
    target_dir = '/scratch/cluster/aish/ReferIt/resized_imcrop'
    match_regex = 'resized_imcrop[0-9]+_[0-9]+.png'
    for filename in os.listdir(parent_dir):
        if re.match(match_regex, filename):
            orig_file = os.path.join(parent_dir, filename)
            new_file = os.path.join(target_dir, filename)
            os.rename(orig_file, new_file)


def rename():
    target_dir = '/scratch/cluster/aish/ReferIt/resized_imcrop'
    for filename in os.listdir(target_dir):
        orig_file = os.path.join(target_dir, filename)
        new_file = os.path.join(target_dir, re.sub('resized_imcrop', '', filename))
        os.rename(orig_file, new_file)


if __name__ == '__main__':
    rename()
