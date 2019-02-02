#!/usr/bin/python

import os

__author__ = 'aishwarya'


# Fetch the last line of a text file
# Source:
# https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file
def tail(filepath):
    if not os.path.isfile(filepath) or os.path.getsize(filepath) <= 0:
        return None

    with open(filepath, "rb") as f:
        f.seek(-2, 2)             # Jump to the second last byte.
        while f.read(1) != b"\n": # Until EOL is found...
            try:
                f.seek(-2, 1)     # ...jump back the read byte plus one more.
            except IOError:
                f.seek(-1, 1)
                if f.tell() == 0:
                    break
        last = f.readline()       # Read last line.
    return last


# Read next line of a file and return None if EOF
def wrapped_next(file_handle):
    try:
        line = file_handle.next()
        return line
    except StopIteration:
        return None


def count_lines(filename):
    file_handle = open(filename)
    num_lines = len(file_handle.read().split('\n'))
    file_handle.close()
    return num_lines


def create_dir(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        create_dir(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)