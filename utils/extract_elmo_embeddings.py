# Extract ELMo embeddings for a JSON annotations file

import tensorflow as tf
import tensorflow_hub as hub
import h5py
from argparse import ArgumentParser

from json_wrapper import *


def main(args):
    module_url = "https://tfhub.dev/google/elmo/2"
    elmo = hub.Module(module_url, trainable=True)
    sentences_placeholder = tf.placeholder(tf.string)
    embeddings = elmo(sentences_placeholder, signature="default", as_dict=True)["elmo"]

    annotations = load_json(args.annotations_file)
    for image_id in annotations:



if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--annotations-file', type=str, required=True,
                            help='JSON file with annotations')
    arg_parser.add_argument('--output-file', type=str, required=True,
                            help='HDF5 file to store output')
    args = arg_parser.parse_args()
    main(args)
