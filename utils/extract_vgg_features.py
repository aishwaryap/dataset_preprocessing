#!/usr/bin/python

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import csv
import caffe
from PIL import ImageFile
from file_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FeatureExtractor():
    def __init__(self, weights_path, image_net_proto, device_id=-1):
        if device_id >= 0:
            caffe.set_mode_gpu()
            caffe.set_device(device_id)
        else:
            caffe.set_mode_cpu()
        # Setup image processing net.
        phase = caffe.TEST
        self.image_net = caffe.Net(image_net_proto, weights_path, phase)
        image_data_shape = self.image_net.blobs['data'].data.shape
        self.transformer = caffe.io.Transformer({'data': image_data_shape})
        channel_mean = np.zeros(image_data_shape[1:])
        channel_mean_values = [104, 117, 123]
        assert channel_mean.shape[0] == len(channel_mean_values)
        for channel_index, mean_val in enumerate(channel_mean_values):
            channel_mean[channel_index, ...] = mean_val
        self.transformer.set_mean('data', channel_mean)
        self.transformer.set_channel_swap('data', (2, 1, 0))  # BGR
        self.transformer.set_transpose('data', (2, 0, 1))

    def set_image_batch_size(self, batch_size):
        self.image_net.blobs['data'].reshape(batch_size,
                                             *self.image_net.blobs['data'].data.shape[1:])

    # Crop should be of the format x, y, width, height
    def preprocess_image(self, image_path, verbose=False, crop=None):
        if verbose:
            print 'Preprocessing image', image_path

        image = plt.imread(image_path)
        if verbose:
            print 'Original image shape =', image.shape

        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        if image.shape[2] == 1:
            image = np.tile(image, (1, 1, 3))
        if verbose:
            print 'RGB image shape =', image.shape

        if crop is not None:
            [x, y, width, height] = [int(val) for val in crop]

            # Some checks to handle bbox annotation errors
            x_min = max(x, 0)
            x_max = min(x + width, image.shape[1] - 1)
            y_min = max(y, 0)
            y_max = min(y + height, image.shape[0] - 1)
            if x_min + 1 >= x_max:
                x_min = 0
                x_max = image.shape[1] - 1
            if y_min + 1 >= y_max:
                y_min = 0
                y_max = image.shape[0] - 1

            # Need to crop one at a time to avoid broadcasting in numpy
            image = image[:, range(x_min, x_max), :]
            image = image[range(y_min, y_max), :, :]
            print 'Cropped image shape =', image.shape

        preprocessed_image = self.transformer.preprocess('data', image)
        if verbose:
            print 'Preprocessed image has shape %s, range (%f, %f)' % \
                  (preprocessed_image.shape,
                   preprocessed_image.min(),
                   preprocessed_image.max())

        return preprocessed_image

    def image_to_feature(self, image, output_name='fc7'):
        net = self.image_net
        if net.blobs['data'].data.shape[0] > 1:
            batch = np.zeros_like(net.blobs['data'].data)
            batch[0] = image
        else:
            batch = image.reshape(net.blobs['data'].data.shape)
        net.forward(data=batch)
        feature = net.blobs[output_name].data[0].copy()
        return feature

    def compute_features(self, image_list, output_name='fc7', verbose=False):
        batch = np.zeros_like(self.image_net.blobs['data'].data)
        batch_shape = batch.shape
        batch_size = batch_shape[0]
        features_shape = (len(image_list),) + \
                         self.image_net.blobs[output_name].data.shape[1:]
        features = np.zeros(features_shape)

        for batch_start_index in range(0, len(image_list), batch_size):
            batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]

            for batch_index, row in enumerate(batch_list):
                image_path = row[1]
                if len(row) > 2:
                    crop = row[2:]
                else:
                    crop = None
                batch[batch_index:(batch_index + 1)] = self.preprocess_image(image_path, verbose=verbose, crop=crop)

            current_batch_size = min(batch_size, len(image_list) - batch_start_index)
            print 'Computing features for images %d-%d of %d' % \
                  (batch_start_index, batch_start_index + current_batch_size - 1,
                   len(image_list))
            self.image_net.forward(data=batch)
            features[batch_start_index:(batch_start_index + current_batch_size)] = \
                self.image_net.blobs[output_name].data[:current_batch_size]
        return features


def write_features_to_file(image_list, features, output_file):
    with open(output_file, 'a') as opfd:
        for i, row in enumerate(image_list):
            image_name = 'region_' + str(row[0])
            image_feature = features[i].tolist()
            feature_str = ','.join(map(str, image_feature))
            opfd.write('%s,%s\n' % (image_name, feature_str))


def compute_single_image_feature(feature_extractor, image_path, out_file):
    assert os.path.exists(image_path)
    preprocessed_image = feature_extractor.preprocess_image(image_path)
    feature = feature_extractor.image_to_feature(preprocessed_image)
    write_features_to_file([image_path], [feature], out_file)


def compute_image_list_features(feature_extractor, image_list_file, out_file, restart_log_file, verbose=False):
    assert os.path.exists(image_list_file)

    # Check restart log to find the line number to restart
    last_line = tail(restart_log_file)
    start_idx = 0
    if last_line is not None:
        start_idx = int(last_line.strip())
    end_idx = start_idx + args.write_after

    file_handle = open(image_list_file)
    reader = csv.reader(file_handle, delimiter=',')
    row_idx = 0

    while True:
        image_list = list()
        for row in reader:
            if row_idx in range(start_idx, end_idx):
                image_list.append(row)
            row_idx += 1
            if row_idx == end_idx:
                break

        features = feature_extractor.compute_features(image_list, verbose=verbose)
        write_features_to_file(image_list, features, out_file)

        if row_idx < end_idx:
            # Reading images for loop terminated because it is done
            break

        start_idx = end_idx
        end_idx += args.write_after
        restart_log = open(restart_log_file, 'a')
        restart_log.write('\n' + str(start_idx))
        restart_log.close()

    file_handle.close()


def main(args):
    # Create the restart log if it does not exist
    if not os.path.isfile(args.restart_log):
        restart_log = open(args.restart_log, 'w')
        restart_log.close()

    device_id = 0
    feature_extractor = FeatureExtractor(args.caffemodel_file, args.prototxt_file, device_id)
    feature_extractor.set_image_batch_size(args.batch_size)

    # compute features for a list of images in a file
    compute_image_list_features(feature_extractor, args.image_list_file, args.output_file, args.restart_log,
                                verbose=args.verbose)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--image-list-file', type=str, required=True,
                            help='Text file with paths to images. Rows format - \n' +
                                 '\tidentifier, image_file, x, y, width, height \n' +
                                 'identifier - name to region, [x, y, width, height] - crop of image (optional)')
    arg_parser.add_argument('--output-file', type=str, required=True,
                            help='File to store CSV features')
    arg_parser.add_argument('--prototxt-file', type=str, required=True,
                            help='File with deploy prototxt of model')
    arg_parser.add_argument('--caffemodel-file', type=str, required=True,
                            help='Caffemodel file with weights')
    arg_parser.add_argument('--batch-size', type=int, default=512,
                            help='Batch size')
    arg_parser.add_argument('--write-after', type=int, default=512,
                            help='Write after this many images (>= batch size) repeatedly')
    arg_parser.add_argument('--restart-log', type=str, required=True,
                            help='A file to log progress to handle restarts')
    arg_parser.add_argument('--verbose', action="store_true", default=False,
                            help='Print verbose debug output')
    args = arg_parser.parse_args()

    main(args)
    print 'Complete'
