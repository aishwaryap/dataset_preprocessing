#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

import re
import numpy as np
from PIL import Image

import sys
sys.path.append('/u/aish/Documents/Research/Code/models/research/slim/preprocessing')
from vgg_preprocessing import preprocess_image

slim = tf.contrib.slim


def get_numpy_array(image):
    image_data = np.array(image.getdata())
    return image_data.reshape(image.size[1], image.size[0], image_data.shape[1])


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


def get_network():
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
    return images_placeholder, net, all_layers


def process_images(image_files):
    images_placeholder, net, all_layers = get_network()
    ckpt_path = '/scratch/cluster/aish/tf_slim_models/resnet_v2_101.ckpt'
    init_fn = get_init_fn(ckpt_path)

    batch_images = list()
    for image_file in image_files:
        pillow_image = Image.open(image_file)
        pillow_image = pillow_image.resize((512, 512))
        pillow_image.save(re.sub('.jpg', '_resized.jpg', image_file))
        np_image = get_numpy_array(pillow_image)
        np_image = np.expand_dims(np_image, 0)
        batch_images.append(np_image)
    batch_images = np.concatenate(batch_images, 0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        init_fn(sess)

        feed_dict = {images_placeholder: batch_images}
        output, activations = sess.run([net, all_layers], feed_dict=feed_dict)

    print('output.shape = ', output.shape)
    print("activations['predictions'].shape =", activations['predictions'].shape)
    print("activations['resnet_v2_101/logits'].shape =", activations['resnet_v2_101/logits'].shape)


if __name__ == '__main__':
    image_files = ['/u/aish/Documents/temp/10000.jpg']
    process_images(image_files)