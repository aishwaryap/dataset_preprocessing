#!/usr/bin/env python
# The features created by this are huge because the file written is text, not binary
# Tested on tensorflow 1.12 with python 3.5

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

import os
import re
import csv
from PIL import Image
import numpy as np
import h5py

from argparse import ArgumentParser

import sys
sys.path.append('/u/aish/Documents/Research/Code/models/research/slim/preprocessing')
from vgg_preprocessing import preprocess_image

slim = tf.contrib.slim


def get_numpy_array(image):
    image_data = np.array(image.getdata())
    return image_data.reshape(image.size[0], image.size[1], image_data.shape[1])


def get_init_fn(checkpoint_path):
    checkpoint_exclude_scopes = ["resnet_v2_101/logits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore)


def main(args):
    image_list_file = os.path.join(*[args.dataset_dir, 'image_lists', args.image_list_file])
    print('image_list_file =', image_list_file)
    with open(image_list_file) as handle:
        rows = handle.read().splitlines()
        rows = [row.split(',') for row in rows]
        image_files = [row[1] for row in rows]
        if len(rows[0]) > 2:
            crops = [row[2:] for row in rows]
        else:
            crops = [None] * len(image_files)
    print('len(image_files) =', len(image_files))

    output_file = os.path.join(*[args.dataset_dir, "resnet_fcn_features", args.output_file])
    output_file_handle = h5py.File(output_file, 'w')
    dataset_name = re.sub('.txt', '', args.image_list_file.split('/')[-1])
    hpy5_dataset = output_file_handle.create_dataset(dataset_name,
                                                     shape=(len(image_files), 32, 32, 2048),
                                                     dtype='f')

    images_placeholder = tf.placeholder(tf.float32, shape=(None, 512, 512, 3))
    preprocessed_batch = tf.map_fn(lambda img: preprocess_image(img,
                                                                output_height=512,
                                                                output_width=512,
                                                                is_training=False,
                                                                resize_side_min=512,
                                                                resize_side_max=512
                                                                ),
                                   images_placeholder)
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, all_layers = resnet_v2.resnet_v2_101(preprocessed_batch,
                                                  21,
                                                  is_training=False,
                                                  global_pool=False,
                                                  output_stride=16)
    init_fn = get_init_fn(args.ckpt_path)

    # This is to prevent a CuDNN error - https://github.com/tensorflow/tensorflow/issues/24828
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        init_fn(sess)

        num_images_in_batch = 0
        num_batches_done = 0
        batch_images = list()
        num_images_processed = 0

        if args.max_images_to_process is not None:
            max_images_to_process = args.max_images_to_process
        else:
            # Set it to a number that won't be hit
            max_images_to_process = len(image_files) + 1
        print('max_images_to_process =', max_images_to_process)

        inputs = zip(image_files, crops)
        for image_file, crop in inputs:
            pillow_image = Image.open(image_file)
            if crop is not None:
                pillow_image = pillow_image.crop(crop)
            pillow_image = pillow_image.resize((512, 512))
            np_image = get_numpy_array(pillow_image)
            np_image = np.expand_dims(np_image, 0)
            batch_images.append(np_image)

            num_images_in_batch += 1

            if num_images_in_batch >= args.batch_size or num_images_processed >= max_images_to_process:
                batch_images = np.concatenate(batch_images, 0)
                feed_dict = {images_placeholder: batch_images}
                output, activations = sess.run([net, all_layers], feed_dict=feed_dict)
                features = activations['resnet_v2_101/block4']
                batch_start_idx = num_batches_done * args.batch_size
                batch_end_idx = batch_start_idx + features.shape[0]
                print("batch_start_idx =", batch_start_idx)
                print("batch_end_idx =", batch_end_idx)
                print("features.shape =", features.shape)
                hpy5_dataset[batch_start_idx:batch_end_idx, :] = features

                print("Completed batch", num_batches_done)
                num_batches_done += 1
                num_images_processed += num_images_in_batch
                num_images_in_batch = 0
                print('num_images_processed =', num_images_processed)

                batch_images = list()
                if num_images_processed >= max_images_to_process:
                    print('Breaking loop: num_images_processed =', num_images_processed, ', max_images_to_process =', max_images_to_process)
                    break

        if len(batch_images) > 0:
            print('Completing remaining', len(batch_images), 'images')
            batch_images = np.concatenate(batch_images, 0)
            feed_dict = {images_placeholder: batch_images}
            output, activations = sess.run([net, all_layers], feed_dict=feed_dict)
            features = activations['resnet_v2_101/block4']
            batch_start_idx = num_batches_done * args.batch_size
            batch_end_idx = batch_start_idx + features.shape[0] - 1
            hpy5_dataset[batch_start_idx:batch_end_idx, :] = features
            num_batches_done += 1

    output_file_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to ReferIt dataset')
    arg_parser.add_argument('--ckpt-path', type=str, required=True,
                            help='Checkpoint path')
    arg_parser.add_argument('--image-list-file', type=str, required=True,
                            help='Image list file')
    arg_parser.add_argument('--output-file', type=str, required=True,
                            help='Output file')
    arg_parser.add_argument('--batch-size', type=int, default=16,
                            help='Batch size for feature extraction')
    arg_parser.add_argument('--max-images-to-process', type=int, default=None,
                            help='Stop processing after this many images - set to None if full list is needed')

    args = arg_parser.parse_args()
    main(args)
