#!/usr/bin/python
# Some useful json wrappers, mainly for interactive debugging

import json

__author__='aishwarya'


def load_json(filename):
    file_handle = open(filename)
    obj = json.loads(file_handle.read())
    file_handle.close()
    return obj


def save_json(obj, filename):
    file_handle = open(filename, 'w')
    json_str = json.dumps(obj)
    file_handle.write(json_str)
    file_handle.close()