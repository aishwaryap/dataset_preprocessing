# Extract ELMo embeddings for a JSON annotations file

import tensorflow as tf
import tensorflow_hub as hub
import h5py
import numpy as np
import numpy.matlib as mb
from argparse import ArgumentParser

from json_wrapper import *
import csv


def main(args):
    module_url = "https://tfhub.dev/google/elmo/2"
    elmo = hub.Module(module_url, trainable=False)
    sentences_placeholder = tf.placeholder(tf.string)
    elmo = elmo(sentences_placeholder, signature="default", as_dict=True)["elmo"]

    with open(args.pad_vector_file) as handle:
        reader = csv.reader(handle)
        pad_vector = [float(x) for x in reader.__next__()]

    annotations = load_json(args.annotations_file)
    num_images = len(annotations.keys())

    output_file_handle = h5py.File(args.output_file, 'w')

    with tf.Session() as sess:
        init_ops = [tf.global_variables_initializer(),
                    tf.tables_initializer()]
        sess.run(init_ops)

        num_images_done = 0

        for image_id in annotations:
            image_annotations = annotations[image_id]
            num_annotations = len(image_annotations)

            image_dataset = output_file_handle.create_dataset(image_id,
                                                              shape=(num_annotations, args.max_seq_len, 1024),
                                                              dtype='f')
            if num_annotations % args.max_batch_size == 0:
                num_batches = (num_annotations // args.max_batch_size)
            else:
                num_batches = (num_annotations // args.max_batch_size) + 1

            for batch_num in range(num_batches):
                batch_start_idx = batch_num * args.max_batch_size
                batch_end_idx = min((batch_num + 1) * args.max_batch_size, num_annotations)

                batch = image_annotations[batch_start_idx:batch_end_idx]
                batch_embeddings = sess.run(elmo,
                                            feed_dict={sentences_placeholder: batch})

                if len(batch_embeddings.shape) == 2:
                    batch_embeddings = np.expand_dims(batch_embeddings, axis=0)
                if batch_embeddings.shape[1] > args.max_seq_len:
                    batch_embeddings = batch_embeddings[:, :args.max_seq_len, :]
                elif batch_embeddings.shape[1] < args.max_seq_len:
                    num_reps = args.max_seq_len - batch_embeddings.shape[1]
                    padding = np.tile(pad_vector, (batch_embeddings.shape[0], num_reps, 1))
                    batch_embeddings = np.concatenate((batch_embeddings, padding), axis=1)

                image_dataset[batch_start_idx:batch_end_idx, :] = batch_embeddings

                print('Finished image', str(num_images_done), '/', str(num_images), ': batch', str(batch_num))

            num_images_done += 1
            if args.max_images is not None and num_images_done >= args.max_images:
                break

    output_file_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--annotations-file', type=str, required=True,
                            help='JSON file with annotations')
    arg_parser.add_argument('--output-file', type=str, required=True,
                            help='HDF5 file to store output')
    arg_parser.add_argument('--pad-vector-file', type=str, required=True,
                            help='CAV file with padding vector')
    arg_parser.add_argument('--max-batch-size', type=int, default=1024,
                            help='Max batch size')
    arg_parser.add_argument('--max-seq-len', type=int, default=40,
                            help='Max sequence length')
    arg_parser.add_argument('--max-images', type=int, default=None,
                            help='Max images to process')
    args = arg_parser.parse_args()
    main(args)
