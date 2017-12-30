#!/usr/bin/python
# This has some useful dataset checks that ideally should be run only once

from argparse import ArgumentParser
import json
import sys
import csv
import numpy as np
from matplotlib import pyplot as plt
from utils import *
sys.path.append('../utils')
from json_wrapper import *
from file_utils import *

__author__ = 'aishwarya'


def check_densities_first_step(args):
    path = os.path.join(args.dataset_dir, 'tmp_sum_cosine_sims/train/')
    print 'Sum cosine sims train - '
    for i in range(22):
        for j in range(22):
            for k in range(8):
                fn = path + str(i) + '_' + str(j) + '_' + str(k) + '.csv'
                if not os.path.isfile(fn):
                    print str(i) + '_' + str(j) + '_' + str(k) + '\t0'
                else:
                    with open(fn) as f:
                        a = f.read().strip().split(',')
                        if len(a) != 65536:
                            print str(i) + '_' + str(j) + '_' + str(k) + '\t' + str(len(a))
    print

    path = os.path.join(args.dataset_dir, 'tmp_sum_cosine_sims/test/')
    print 'Sum cosine sims test - '
    for i in range(8):
        for j in range(8):
            for k in range(8):
                fn = path + str(i) + '_' + str(j) + '_' + str(k) + '.csv'
                if not os.path.isfile(fn):
                    print str(i) + '_' + str(j) + '_' + str(k) + '\t0'
                else:
                    with open(fn) as f:
                        a = f.read().strip().split(',')
                        if len(a) != 65536:
                            print str(i) + '_' + str(j) + '_' + str(k) + '\t' + str(len(a))
    print

    path = os.path.join(args.dataset_dir, 'tmp_nbrs/train/')
    print 'Nbrs train - '
    for i in range(22):
        for j in range(22):
            for k in range(8):
                fn = path + str(i) + '_' + str(j) + '_' + str(k) + '.csv'
                if not os.path.isfile(fn):
                    print str(i) + '_' + str(j) + '_' + str(k) + '\t0'
                else:
                    num_lines = count_lines(fn)
                    if num_lines != 65536:
                        print str(i) + '_' + str(j) + '_' + str(k) + '\t' + str(num_lines)
    print

    path = os.path.join(args.dataset_dir, 'tmp_nbrs/test/')
    print 'Nbrs test - '
    for i in range(8):
        for j in range(8):
            for k in range(8):
                fn = path + str(i) + '_' + str(j) + '_' + str(k) + '.csv'
                if not os.path.isfile(fn):
                    print str(i) + '_' + str(j) + '_' + str(k) + '\t0'
                else:
                    num_lines = count_lines(fn)
                    if num_lines != 65536:
                        print str(i) + '_' + str(j) + '_' + str(k) + '\t' + str(num_lines)
    print


# Check that all images in regions exist
def check_images(args):
    region_graphs_filename = os.path.join(args.dataset_dir, 'region_graphs.txt')
    region_graphs_file = open(region_graphs_filename)
    num_regions_processed = 0
    for line in region_graphs_file:
        region_graph = json.loads(line.strip())
        assert(get_image_path(args.dataset_dir, region_graph['image_id'], verify=True) is not None)
        num_regions_processed += 1
        if num_regions_processed % 10000 == 0 and args.verbose:
            print num_regions_processed, 'regions processed ...'
    region_graphs_file.close()
    print 'All images present'


# Find the ranges of x, y, width, height for regions
def check_bbox_ranges(args):
    region_graphs_file = open(os.path.join(args.dataset_dir, 'region_graphs.txt'))
    num_regions_processed = 0
    min_x = min_y = min_w = min_h = sys.maxint
    max_x = max_y = max_w = max_h = -sys.maxint
    for line in region_graphs_file:
        region = json.loads(line.strip())
        min_x = min(min_x, region['x'])
        min_y = min(min_y, region['y'])
        min_w = min(min_w, region['width'])
        min_h = min(min_h, region['height'])
        max_x = max(max_x, region['x'])
        max_y = max(max_y, region['y'])
        max_w = max(max_w, region['width'])
        max_h = max(max_h, region['height'])
        num_regions_processed += 1
        if num_regions_processed % 50000 == 0 and args.verbose:
            print num_regions_processed, 'regions processed'
    print 'min_x =', min_x
    print 'min_y =', min_y
    print 'min_w =', min_w
    print 'min_h =', min_h
    print 'max_x =', max_x
    print 'max_y =', max_y
    print 'max_w =', max_w
    print 'max_h =', max_h


# Replace negative values in x and y with appropriate borders of image
def correct_bboxes(args):
    image_data = load_json(os.path.join(args.dataset_dir, 'image_data.json'))
    print 'Loaded image metadata ...'

    image_sizes = dict()
    for image in image_data:
        image_sizes[image['image_id']] = dict()
        image_sizes[image['image_id']]['width'] = image['width']
        image_sizes[image['image_id']]['height'] = image['height']
    print 'Organized image sizes ...'

    region_graphs_file = open(os.path.join(args.dataset_dir, 'region_graphs.txt'))
    corrected_file = open(os.path.join(args.dataset_dir, 'region_graphs_corrected.txt'), 'w')
    num_regions_processed = 0
    for line in region_graphs_file:
        region = json.loads(line.strip())
        region['x'] = max(region['x'], 0)
        region['x'] = min(region['x'], image_sizes[region['image_id']]['width'])
        region['y'] = max(region['y'], 0)
        region['y'] = min(region['y'], image_sizes[region['image_id']]['height'])
        if region['width'] < 0 or region['height'] < 0:
            raise RuntimeError('Bad width/height in region :\n' + json.dumps(region))
        region_str = json.dumps(region)
        corrected_file.write(region_str + '\n')
        num_regions_processed += 1
        if num_regions_processed % 10000 == 0 and args.verbose:
            print num_regions_processed, 'regions processed'
    region_graphs_file.close()
    corrected_file.close()
    print 'Corrected regions'


def count_greyscale_images(args):
    region_graphs_filename = os.path.join(args.dataset_dir, 'region_graphs.txt')
    region_graphs_file = open(region_graphs_filename)
    num_regions_processed = 0
    num_greyscale_images = 0
    for line in region_graphs_file:
        region_graph = json.loads(line.strip())
        image_path = get_image_path(args.dataset_dir, region_graph['image_id'])
        image = plt.imread(image_path)
        if len(image.shape) < 3 or image.shape[2] == 1:
            num_greyscale_images += 1
        num_regions_processed += 1
        if num_regions_processed % 10000 == 0 and args.verbose:
            print num_regions_processed, 'regions processed ...'
    region_graphs_file.close()
    print 'num_greyscale_images =', num_greyscale_images


# Checks on whether VGG feature extraction finished successfully for all regions
# This is badly designed in the sense that it's output needs to be inspected manually
def check_vgg_feature_extraction(args):
    image_lists_dir = os.path.join(args.dataset_dir, 'image_lists')
    image_lists = os.listdir(image_lists_dir)
    for image_list in image_lists:
        image_list_num = image_list.split('.')[0]
        print '\n\nChecking list', image_list_num
        image_list_filename = os.path.join(image_lists_dir, image_list)
        image_list_file = open(image_list_filename)
        features_filename = os.path.join(*[args.dataset_dir, 'regions_vgg_features', image_list_num + '.csv'])
        if not os.path.isfile(features_filename):
            print 'No features file!!!'
            continue
        features_file = open(features_filename)
        line_num = 0
        while True:
            line_num += 1
            next_region = wrapped_next(image_list_file)
            next_feature = wrapped_next(features_file)
            if next_region is None and next_feature is not None:
                print 'More features than expected. Extra features starting at line', line_num
                break
            if next_feature is None and next_region is not None:
                print 'Fewer features than expected. First region not present at line', line_num
                break
            if next_region is None and next_feature is None:
                print 'No errors :-)'
                break

            # If control comes here we got a line from each file
            feature_region_id = next_feature.split(',')[0].split('_')[1]
            region_id = next_region.split(',')[0]
            if feature_region_id != region_id:
                print 'ID mismatch at line', line_num
                print 'Feature region id =', feature_region_id
                print 'Expected region id =', region_id
                break

        image_list_file.close()
        features_file.close()


# A simple sanity check on reorganized data for classification
def check_multilabels(args):
    metadata = [
        # {
        #     'regions_filename': os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'),
        #     'multilabels_dir': os.path.join(args.dataset_dir, 'classifiers/data/multilabels/train/'),
        #     'is_train_set': True
        # },
        {
            'regions_filename': os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'),
            'multilabels_dir': os.path.join(args.dataset_dir, 'classifiers/data/multilabels/test/'),
            'is_train_set': False
        }
    ]

    rerun_file = open(args.rerun_script_filename, 'w')

    for metadata_instance in metadata:
        print 'Checking regions file', metadata_instance['regions_filename']
        num_regions = count_lines(metadata_instance['regions_filename'])

        c1 = 'python create_classifier_data.py \\\n --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n' + \
             '--write-multilabels \\\n --batch-num='
        c2 = ' \\\n --verbose'
        num_incomplete_batches = 0

        if args.verbose:
            print 'Checking multilabels'
        with open(os.path.join(args.dataset_dir, 'classifiers/data/label_names.txt')) as label_names_file:
            label_names = label_names_file.read().split('\n')
        multilabels_files = [os.path.join(metadata_instance['multilabels_dir'], f)
                             for f in os.listdir(metadata_instance['multilabels_dir'])]
        num_multilabels = 0
        num_batches_done = 0

        for multilabels_file in multilabels_files:
            if args.verbose:
                print 'File: ', multilabels_file
            multilabels = np.loadtxt(multilabels_file, delimiter=',')
            if multilabels is None or len(multilabels.shape) < 2:
                num_incomplete_batches += 1
                batch_num = multilabels_file.split('.')[0].split('/')[-1]
                command = c1 + str(batch_num) + c2
                if metadata_instance['is_train_set']:
                    command += ' \\\n --in-train-set'
                rerun_file.write(command + '\n')
            else:
                assert (multilabels.shape[1] == len(label_names))
                num_multilabels += multilabels.shape[0]
            num_batches_done += 1
            if args.verbose:
                print num_batches_done, 'batches checked'

        if args.verbose:
            print 'num_incomplete_batches =', num_incomplete_batches
        if num_incomplete_batches == 0:
            assert (num_regions == num_multilabels)
            print 'Multilabels check passed...'

    rerun_file.close()


# A simple sanity check on reorganized data for classification
def check_features(args):
    metadata = [
        {
            'regions_filename': os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'),
            'features_dir': os.path.join(args.dataset_dir, 'classifiers/data/features/train/'),
            'multilabels_dir': os.path.join(args.dataset_dir, 'classifiers/data/multilabels/train/'),
            'individual_labels_dir': os.path.join(args.dataset_dir, 'classifiers/data/binary_labels/train/'),
            'is_train_set': True
        },
        {
            'regions_filename': os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'),
            'features_dir': os.path.join(args.dataset_dir, 'classifiers/data/features/test/'),
            'multilabels_dir': os.path.join(args.dataset_dir, 'classifiers/data/multilabels/test/'),
            'individual_labels_dir': os.path.join(args.dataset_dir, 'classifiers/data/binary_labels/test/'),
            'is_train_set': False
        }
    ]

    rerun_file = open(args.rerun_script_filename, 'w')

    for metadata_instance in metadata:
        print 'Checking regions file', metadata_instance['regions_filename']
        num_regions = count_lines(metadata_instance['regions_filename'])

        c1 = 'python create_classifier_data.py \\\n --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n' + \
             '--write-features \\\n --batch-num='
        c2 = ' \\\n --verbose'
        num_incomplete_batches = 0

        features_files = [os.path.join(metadata_instance['features_dir'], f)
                          for f in os.listdir(metadata_instance['features_dir'])]

        if args.verbose:
            print 'Checking features'
        num_feature_vectors = 0
        num_batches_done = 0
        for features_file in features_files:
            if args.verbose:
                print 'File :', features_file
            try:
                features = np.loadtxt(features_file)
                if features is None or len(features.shape) < 2:
                    num_incomplete_batches += 1
                    batch_num = features_file.split('.')[0].split('/')[-1]
                    command = c1 + str(batch_num) + c2
                    if metadata_instance['is_train_set']:
                        command += ' \\\n --in-train-set'
                    rerun_file.write(command + '\n')
                else:
                    assert (features.shape[1] == 4096)
                    num_feature_vectors += features.shape[0]
            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except:
                num_incomplete_batches += 1
                batch_num = features_file.split('.')[0].split('/')[-1]
                command = c1 + str(batch_num) + c2
                if metadata_instance['is_train_set']:
                    command += ' \\\n --in-train-set'
                rerun_file.write(command + '\n')
            num_batches_done += 1
            if args.verbose:
                print num_batches_done, 'batches checked'

        if args.verbose:
            print 'num_incomplete_batches =', num_incomplete_batches
        if num_incomplete_batches == 0:
            assert (num_regions == num_feature_vectors)
            print 'Features check passed...'

    rerun_file.close()


# A simple sanity check on reorganized data for classification
def check_individual_labels(args):
    metadata = [
        {
            'regions_filename': os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'),
            'individual_labels_dir': os.path.join(args.dataset_dir, 'classifiers/data/binary_labels/train/'),
            'is_train_set': True
        },
        {
            'regions_filename': os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'),
            'individual_labels_dir': os.path.join(args.dataset_dir, 'classifiers/data/binary_labels/test/'),
            'is_train_set': False
        }
    ]

    rerun_file = open(args.rerun_script_filename, 'w')

    for metadata_instance in metadata:
        print 'Checking regions file', metadata_instance['regions_filename']
        num_regions = count_lines(metadata_instance['regions_filename'])

        c1 = 'python create_classifier_data.py \\\n --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n' + \
             '--write-individual-labels \\\n --label='
        c2 = ' \\\n --verbose'

        if args.verbose:
            print 'Checking individual labels'
        label_dirs = [os.path.join(metadata_instance['individual_labels_dir'], f)
                      for f in os.listdir(metadata_instance['individual_labels_dir'])]

        for label in label_dirs:
            label_successful = True
            label_name = label.split('/')[-1]
            num_labels = 0

            if args.verbose:
                print 'Label: ', label_name

            label_files = [os.path.join(label, f) for f in os.listdir(label)]
            for label_file in label_files:
                labels = np.loadtxt(label_file, delimiter=',')
                if labels is None:
                    label_successful = False

                    command = c1 + label_name + c2
                    if metadata_instance['is_train_set']:
                        command += ' \\\n --in-train-set'
                    rerun_file.write(command + '\n')
                    break   # No need to read more files of this label
                else:
                    num_labels += labels.shape[0]

            if args.verbose:
                print 'Label', label_name, ' success:', label_successful
            if label_successful:
                assert (num_regions == num_labels)
                print 'All labels present'

    rerun_file.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--verbose', action="store_true", default=False,
                            help='Print verbose debug output')

    arg_parser.add_argument('--check-images', action="store_true", default=False,
                            help='Check that all images in region graphs exist')
    arg_parser.add_argument('--correct-bboxes', action="store_true", default=False,
                            help='Correct bboxes of overflows of x, y')
    arg_parser.add_argument('--check-bbox-ranges', action="store_true", default=False,
                            help='Find min and max of x, y, width, height')
    arg_parser.add_argument('--count-greyscale-images', action="store_true", default=False,
                            help='Count greyscale images')
    arg_parser.add_argument('--check-vgg-features', action="store_true", default=False,
                            help='Check for errors in VGG feature extraction')
    arg_parser.add_argument('--check-multilabels', action="store_true", default=False,
                            help='Check reorganized multilabels fore classifiers')
    arg_parser.add_argument('--check-features', action="store_true", default=False,
                            help='Check reorganized features fore classifiers')
    arg_parser.add_argument('--check-individual-labels', action="store_true", default=False,
                            help='Check reorganized binary labels for classifiers')
    arg_parser.add_argument('--check-densities-first-step', action="store_true", default=False,
                            help='Check first step of computing densities and neighours')

    # For checking classifier data
    arg_parser.add_argument('--rerun-script-filename', type=str, default=None,
                            help='Write commands to rerun')

    args = arg_parser.parse_args()

    if args.check_images:
        check_images(args)
    if args.correct_bboxes:
        correct_bboxes(args)
    if args.check_bbox_ranges:
        check_bbox_ranges(args)
    if args.count_greyscale_images:
        count_greyscale_images(args)
    if args.check_vgg_features:
        check_vgg_feature_extraction(args)
    if args.check_multilabels:
        check_multilabels(args)
    if args.check_features:
        check_features(args)
    if args.check_individual_labels:
        check_individual_labels(args)
    if args.check_densities_first_step:
        check_densities_first_step(args)


	
