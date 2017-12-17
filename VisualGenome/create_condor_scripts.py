#!/usr/bin/env python

from argparse import ArgumentParser
import os
import math

__author__ = 'aishwarya'


def extract_regions_vgg_features(args):
    image_list_files = os.listdir(os.path.join(args.dataset_dir, 'image_lists'))
    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    for image_list_file in image_list_files:
        image_list_file_path = os.path.join(*[args.dataset_dir, 'image_lists', image_list_file])
        image_list_file_num = image_list_file.split('.')[0]

        # Write Condor script that submits the bash script on condor
        condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir,
                                                 image_list_file_num + '.sh'])
        condor_script_file = open(condor_script_file_name, 'w')
        condor_script_file.write('universe = vanilla\n')
        condor_script_file.write('Initialdir = /u/aish/Documents/Research/Code/dataset_preprocessing/utils/\n')
        condor_script_file.write('Executable = /lusr/bin/python\n')
        condor_script_file.write('Arguments = extract_vgg_features.py \\\n')
        condor_script_file.write('\t\t --image-list-file=' + image_list_file_path + ' \\\n')
        condor_script_file.write('\t\t --output-file=' +
                                 os.path.join(*[args.dataset_dir, 'regions_vgg_features/' +
                                                image_list_file_num + '.csv']) + ' \\\n')
        condor_script_file.write('\t\t --prototxt-file=/scratch/cluster/aish/CaffeModels/vgg7k.prototxt \\\n')
        condor_script_file.write('\t\t --caffemodel-file=/scratch/cluster/aish/CaffeModels/vgg7k.caffemodel \\\n')
        condor_script_file.write('\t\t --restart-log=' +
                                 os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir, 'restart',
                                                image_list_file_num + '.csv']) + ' \\\n')
        condor_script_file.write('\t\t --verbose\n')
        condor_script_file.write('+Group   = "GRAD"\n')
        condor_script_file.write('+Project = "AI_ROBOTICS"\n')
        condor_script_file.write('+ProjectDescription = "VisualGenome - Regions VGG-7k feature extraction"\n')
        condor_script_file.write('JobBatchName = "VisualGenome - Regions VGG-7k feature extraction"\n')
        condor_script_file.write('Requirements = TARGET.GPUSlot\n')
        condor_script_file.write('getenv = True\n')
        condor_script_file.write('request_GPUs = 1\n')
        condor_script_file.write('+GPUJob = true\n')
        condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir, 'log',
                                                         image_list_file_num + '.log']) + '\n')
        condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir, 'err',
                                                           image_list_file_num + '.err']) + '\n')
        condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir, 'out',
                                                             image_list_file_num + '.out']) + '\n')
        condor_script_file.write('Notification = complete\n')
        condor_script_file.write('Notify_user = aish@cs.utexas.edu\n')
        condor_script_file.write('Queue 1\n')
        condor_script_file.close()

        condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


# Scripts for first step in computing densities
def density_process_batch_pair(args):
    # Pairs of region file name and is_train_set
    metadata = [
        (os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'), True),
        (os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'), False)
    ]

    # Make scripts dir
    scripts_dir = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir])
    if not os.path.isdir(scripts_dir):
        os.mkdir(scripts_dir)
    scripts_sub_dirs = [os.path.join(scripts_dir, d) for d in ['train', 'test']]
    for scripts_sub_dir in scripts_sub_dirs:
        if not os.path.isdir(scripts_sub_dir):
            os.mkdir(scripts_sub_dir)

    # Make log dirs
    log_dir = os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_sub_dirs = [os.path.join(log_dir, d) for d in ['train', 'test']]
    for log_sub_dir in log_sub_dirs:
        if not os.path.isdir(log_sub_dir):
            os.mkdir(log_sub_dir)
        log_sub_sub_dirs = [os.path.join(log_sub_dir, d) for d in ['log', 'err', 'out']]
        for log_sub_sub_dir in log_sub_sub_dirs:
            if not os.path.isdir(log_sub_sub_dir):
                os.mkdir(log_sub_sub_dir)

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    max_sub_batch_num = int(math.ceil(float(args.batch_size) / args.sub_batch_size))

    for (regions_filename, is_train_set) in metadata:
        with open(regions_filename) as regions_file:
            regions = regions_file.read().split('\n')
        max_batch_num = int(math.ceil(float(len(regions)) / args.batch_size))

        for batch_num_i in range(max_batch_num):
            for batch_num_j in range(max_batch_num):
                for sub_batch_num in range(max_sub_batch_num):
                    if is_train_set:
                        condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir,
                                                                 'train/' + str(batch_num_i) + '_' + str(batch_num_j)
                                                                 + '_' + str(sub_batch_num) + '.sh'])
                    else:
                        condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir,
                                                                 'test/' + str(batch_num_i) + '_' + str(batch_num_j)
                                                                 + '_' + str(sub_batch_num) + '.sh'])

                    condor_script_file = open(condor_script_file_name, 'w')
                    condor_script_file.write('universe = vanilla\n')
                    condor_script_file.write('Initialdir = ' +
                                             '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

                    condor_script_file.write('Executable = /lusr/bin/python\n')

                    condor_script_file.write('Arguments = compute_densities_and_neighbours.py \\\n')
                    condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
                    if is_train_set:
                        condor_script_file.write('\t\t --in-train-set \\\n')
                    condor_script_file.write('\t\t --process-batch-pair \\\n')
                    condor_script_file.write('\t\t --batch-num-i=' + str(batch_num_i) + ' \\\n')
                    condor_script_file.write('\t\t --batch-num-j=' + str(batch_num_j) + ' \\\n')
                    condor_script_file.write('\t\t --sub-batch-num=' + str(sub_batch_num) + ' \\\n')
                    condor_script_file.write('\t\t --num-nbrs=' + str(args.num_nbrs) + ' \n')

                    condor_script_file.write('+Group   = "GRAD"\n')
                    condor_script_file.write('+Project = "AI_ROBOTICS"\n')
                    condor_script_file.write('+ProjectDescription = "VisualGenome - Computing densities - Batch pair"\n')
                    condor_script_file.write('JobBatchName = "VisualGenome - Computing densities - Batch pair"\n')
                    condor_script_file.write('Requirements = InMastodon\n')

                    if is_train_set:
                        condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                           'train/log', str(batch_num_i) + '_' +
                                                                           str(batch_num_j) + '_'
                                                                           + str(sub_batch_num) + '.log']) + '\n')
                        condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                             'train/err', str(batch_num_i) + '_' +
                                                                             str(batch_num_j) + '_'
                                                                           + str(sub_batch_num) + '.err']) + '\n')
                        condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                              'train/out', str(batch_num_i) + '_' +
                                                                              str(batch_num_j) + '_'
                                                                            + str(sub_batch_num) + '.out']) + '\n')
                    else:
                        condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                           'test/log', str(batch_num_i) + '_' +
                                                                           str(batch_num_j) + '_'
                                                                           + str(sub_batch_num) + '.log']) + '\n')
                        condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                             'test/err', str(batch_num_i) + '_' +
                                                                             str(batch_num_j) + '_'
                                                                           + str(sub_batch_num) + '.err']) + '\n')
                        condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                              'test/out', str(batch_num_i) + '_' +
                                                                              str(batch_num_j) + '_'
                                                                            + str(sub_batch_num) + '.out']) + '\n')

                    condor_script_file.write('Queue 1\n')
                    condor_script_file.close()

                    condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


# Scripts to aggregate densities
def compute_densities(args):
    # Pairs of region file name and is_train_set
    metadata = [
        (os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'), True),
        (os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'), False)
    ]

    # Make scripts dir
    scripts_dir = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir])
    if not os.path.isdir(scripts_dir):
        os.mkdir(scripts_dir)
    scripts_sub_dirs = [os.path.join(scripts_dir, d) for d in ['train', 'test']]
    for scripts_sub_dir in scripts_sub_dirs:
        if not os.path.isdir(scripts_sub_dir):
            os.mkdir(scripts_sub_dir)

    # Make log dirs
    log_dir = os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_sub_dirs = [os.path.join(log_dir, d) for d in ['train', 'test']]
    for log_sub_dir in log_sub_dirs:
        if not os.path.isdir(log_sub_dir):
            os.mkdir(log_sub_dir)
        log_sub_sub_dirs = [os.path.join(log_sub_dir, d) for d in ['log', 'err', 'out']]
        for log_sub_sub_dir in log_sub_sub_dirs:
            if not os.path.isdir(log_sub_sub_dir):
                os.mkdir(log_sub_sub_dir)

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    for (regions_filename, is_train_set) in metadata:
        with open(regions_filename) as regions_file:
            regions = regions_file.read().split('\n')
        max_batch_num = int(math.ceil(float(len(regions)) / args.batch_size))

        for batch_num in range(max_batch_num):
            if is_train_set:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'train',
                                                       str(batch_num) + '.sh'])
            else:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'test',
                                                         str(batch_num) + '.sh'])

            condor_script_file = open(condor_script_file_name, 'w')
            condor_script_file.write('universe = vanilla\n')
            condor_script_file.write('Initialdir = ' +
                                     '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

            condor_script_file.write('Executable = /lusr/bin/python\n')

            condor_script_file.write('Arguments = compute_densities_and_neighbours.py \\\n')
            condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
            if is_train_set:
                condor_script_file.write('\t\t --in-train-set \\\n')
            condor_script_file.write('\t\t --aggregate-densities \\\n')
            condor_script_file.write('\t\t --batch-num-i=' + str(batch_num) + ' \\\n')
            condor_script_file.write('\t\t --num-nbrs=' + str(args.num_nbrs) + ' \n')

            condor_script_file.write('+Group   = "GRAD"\n')
            condor_script_file.write('+Project = "AI_ROBOTICS"\n')
            condor_script_file.write('+ProjectDescription = "VisualGenome - Writing features"\n')
            condor_script_file.write('JobBatchName = "VisualGenome - Writing features"\n')
            condor_script_file.write('Requirements = InMastodon\n')

            if is_train_set:
                condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                   'train/log', str(batch_num) + '.log']) + '\n')
                condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                     'train/err', str(batch_num) + '.err']) + '\n')
                condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                      'train/out', str(batch_num) + '.out']) + '\n')
            else:
                condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                   'test/log', str(batch_num) + '.log']) + '\n')
                condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                     'test/err', str(batch_num) + '.err']) + '\n')
                condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                      'test/out', str(batch_num) + '.out']) + '\n')

            condor_script_file.write('Queue 1\n')
            condor_script_file.close()

            condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


# Scripts to aggregate neighbours
def compute_nbrs(args):
    # Pairs of region file name and is_train_set
    metadata = [
        (os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'), True),
        (os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'), False)
    ]

    # Make scripts dir
    scripts_dir = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir])
    if not os.path.isdir(scripts_dir):
        os.mkdir(scripts_dir)
    scripts_sub_dirs = [os.path.join(scripts_dir, d) for d in ['train', 'test']]
    for scripts_sub_dir in scripts_sub_dirs:
        if not os.path.isdir(scripts_sub_dir):
            os.mkdir(scripts_sub_dir)

    # Make log dirs
    log_dir = os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_sub_dirs = [os.path.join(log_dir, d) for d in ['train', 'test']]
    for log_sub_dir in log_sub_dirs:
        if not os.path.isdir(log_sub_dir):
            os.mkdir(log_sub_dir)
        log_sub_sub_dirs = [os.path.join(log_sub_dir, d) for d in ['log', 'err', 'out']]
        for log_sub_sub_dir in log_sub_sub_dirs:
            if not os.path.isdir(log_sub_sub_dir):
                os.mkdir(log_sub_sub_dir)

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    for (regions_filename, is_train_set) in metadata:
        with open(regions_filename) as regions_file:
            regions = regions_file.read().split('\n')
        max_batch_num = int(math.ceil(float(len(regions)) / args.batch_size))

        for batch_num in range(max_batch_num):
            if is_train_set:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'train',
                                                       str(batch_num) + '.sh'])
            else:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'test',
                                                         str(batch_num) + '.sh'])

            condor_script_file = open(condor_script_file_name, 'w')
            condor_script_file.write('universe = vanilla\n')
            condor_script_file.write('Initialdir = ' +
                                     '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

            condor_script_file.write('Executable = /lusr/bin/python\n')

            condor_script_file.write('Arguments = compute_densities_and_neighbours.py \\\n')
            condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
            if is_train_set:
                condor_script_file.write('\t\t --in-train-set \\\n')
            condor_script_file.write('\t\t --aggregate-nbrs \\\n')
            condor_script_file.write('\t\t --batch-num-i=' + str(batch_num) + ' \\\n')
            condor_script_file.write('\t\t --num-nbrs=' + str(args.num_nbrs) + ' \n')

            condor_script_file.write('+Group   = "GRAD"\n')
            condor_script_file.write('+Project = "AI_ROBOTICS"\n')
            condor_script_file.write('+ProjectDescription = "VisualGenome - Writing features"\n')
            condor_script_file.write('JobBatchName = "VisualGenome - Writing features"\n')
            condor_script_file.write('Requirements = InMastodon\n')

            if is_train_set:
                condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                   'train/log', str(batch_num) + '.log']) + '\n')
                condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                     'train/err', str(batch_num) + '.err']) + '\n')
                condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                      'train/out', str(batch_num) + '.out']) + '\n')
            else:
                condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                   'test/log', str(batch_num) + '.log']) + '\n')
                condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                     'test/err', str(batch_num) + '.err']) + '\n')
                condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                      'test/out', str(batch_num) + '.out']) + '\n')

            condor_script_file.write('Queue 1\n')
            condor_script_file.close()

            condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


# Scripts to write label vectors of multilabel data
def write_features(args):
    # Pairs of region file name and is_train_set
    metadata = [
        (os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'), True),
        (os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'), False)
    ]

    # Make scripts dir
    scripts_dir = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir])
    if not os.path.isdir(scripts_dir):
        os.mkdir(scripts_dir)
    scripts_sub_dirs = [os.path.join(scripts_dir, d) for d in ['train', 'test']]
    for scripts_sub_dir in scripts_sub_dirs:
        if not os.path.isdir(scripts_sub_dir):
            os.mkdir(scripts_sub_dir)

    # Make log dirs
    log_dir = os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_sub_dirs = [os.path.join(log_dir, d) for d in ['train', 'test']]
    for log_sub_dir in log_sub_dirs:
        if not os.path.isdir(log_sub_dir):
            os.mkdir(log_sub_dir)
        log_sub_sub_dirs = [os.path.join(log_sub_dir, d) for d in ['log', 'err', 'out']]
        for log_sub_sub_dir in log_sub_sub_dirs:
            if not os.path.isdir(log_sub_sub_dir):
                os.mkdir(log_sub_sub_dir)

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    for (regions_filename, is_train_set) in metadata:
        with open(regions_filename) as regions_file:
            regions = regions_file.read().split('\n')
        max_batch_num = int(math.ceil(float(len(regions)) / args.batch_size))

        for batch_num in range(max_batch_num):
            if is_train_set:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'train',
                                                       str(batch_num) + '.sh'])
            else:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'test',
                                                         str(batch_num) + '.sh'])

            condor_script_file = open(condor_script_file_name, 'w')
            condor_script_file.write('universe = vanilla\n')
            condor_script_file.write('Initialdir = ' +
                                     '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

            condor_script_file.write('Executable = /lusr/bin/python\n')

            condor_script_file.write('Arguments = create_classifier_data.py \\\n')
            condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
            condor_script_file.write('\t\t --write-features \\\n')
            condor_script_file.write('\t\t --batch-num=' + str(batch_num) + ' \\\n')
            if is_train_set:
                condor_script_file.write('\t\t --in-train-set \\\n')
            condor_script_file.write('\t\t --verbose\n')

            condor_script_file.write('+Group   = "GRAD"\n')
            condor_script_file.write('+Project = "AI_ROBOTICS"\n')
            condor_script_file.write('+ProjectDescription = "VisualGenome - Writing features"\n')
            condor_script_file.write('JobBatchName = "VisualGenome - Writing features"\n')
            condor_script_file.write('Requirements = InMastodon\n')

            if is_train_set:
                condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                   'train/log', str(batch_num) + '.log']) + '\n')
                condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                     'train/err', str(batch_num) + '.err']) + '\n')
                condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                      'train/out', str(batch_num) + '.out']) + '\n')
            else:
                condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                   'test/log', str(batch_num) + '.log']) + '\n')
                condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                     'test/err', str(batch_num) + '.err']) + '\n')
                condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                      'test/out', str(batch_num) + '.out']) + '\n')

            condor_script_file.write('Queue 1\n')
            condor_script_file.close()

            condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


# Scripts to write label vectors of multilabel data
def write_multilabels(args):
    # Pairs of region file name and is_train_set
    metadata = [
        (os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'), True),
        (os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'), False)
    ]

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    # Make scripts dir
    scripts_dir = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir])
    if not os.path.isdir(scripts_dir):
        os.mkdir(scripts_dir)
    scripts_sub_dirs = [os.path.join(scripts_dir, d) for d in ['train', 'test']]
    for scripts_sub_dir in scripts_sub_dirs:
        if not os.path.isdir(scripts_sub_dir):
            os.mkdir(scripts_sub_dir)

    # Make log dirs
    log_dir = os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_sub_dirs = [os.path.join(log_dir, d) for d in ['train', 'test']]
    for log_sub_dir in log_sub_dirs:
        if not os.path.isdir(log_sub_dir):
            os.mkdir(log_sub_dir)
        log_sub_sub_dirs = [os.path.join(log_sub_dir, d) for d in ['log', 'err', 'out']]
        for log_sub_sub_dir in log_sub_sub_dirs:
            if not os.path.isdir(log_sub_sub_dir):
                os.mkdir(log_sub_sub_dir)

    for (regions_filename, is_train_set) in metadata:
        with open(regions_filename) as regions_file:
            regions = regions_file.read().split('\n')
        max_batch_num = int(math.ceil(float(len(regions)) / args.batch_size))

        for batch_num in range(max_batch_num):
            if is_train_set:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'train',
                                                       str(batch_num) + '.sh'])
            else:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'test',
                                                         str(batch_num) + '.sh'])

            condor_script_file = open(condor_script_file_name, 'w')
            condor_script_file.write('universe = vanilla\n')
            condor_script_file.write('Initialdir = ' +
                                     '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

            condor_script_file.write('Executable = /lusr/bin/python\n')

            condor_script_file.write('Arguments = create_classifier_data.py \\\n')
            condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
            condor_script_file.write('\t\t --write-multilabels \\\n')
            condor_script_file.write('\t\t --batch-num=' + str(batch_num) + ' \\\n')
            if is_train_set:
                condor_script_file.write('\t\t --in-train-set \\\n')
            condor_script_file.write('\t\t --verbose\n')

            condor_script_file.write('+Group   = "GRAD"\n')
            condor_script_file.write('+Project = "AI_ROBOTICS"\n')
            condor_script_file.write('+ProjectDescription = "VisualGenome - Writing multilabel data"\n')
            condor_script_file.write('JobBatchName = "VisualGenome - Writing multilabel data"\n')
            condor_script_file.write('Requirements = InMastodon\n')

            if is_train_set:
                condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                   'train/log', str(batch_num) + '.log']) + '\n')
                condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                     'train/err', str(batch_num) + '.err']) + '\n')
                condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                      'train/out', str(batch_num) + '.out']) + '\n')
            else:
                condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                   'test/log', str(batch_num) + '.log']) + '\n')
                condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                     'test/err', str(batch_num) + '.err']) + '\n')
                condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                                      'test/out', str(batch_num) + '.out']) + '\n')

            condor_script_file.write('Queue 1\n')
            condor_script_file.close()

            condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


# Scripts to write label vectors of multilabel data
def write_individual_labels(args):
    # Pairs of region file name and is_train_set
    metadata = [
        (os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'), True),
        (os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'), False)
    ]

    with open(os.path.join(args.dataset_dir, 'classifiers/data/label_names.txt')) as label_names_file:
        label_names = label_names_file.read().split('\n')

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    # Make scripts dir
    scripts_dir = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir])
    if not os.path.isdir(scripts_dir):
        os.mkdir(scripts_dir)
    scripts_sub_dirs = [os.path.join(scripts_dir, d) for d in ['train', 'test']]
    for scripts_sub_dir in scripts_sub_dirs:
        if not os.path.isdir(scripts_sub_dir):
            os.mkdir(scripts_sub_dir)

    # Make log dirs
    log_dir = os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_sub_dirs = [os.path.join(log_dir, d) for d in ['train', 'test']]
    for log_sub_dir in log_sub_dirs:
        if not os.path.isdir(log_sub_dir):
            os.mkdir(log_sub_dir)
        log_sub_sub_dirs = [os.path.join(log_sub_dir, d) for d in ['log', 'err', 'out']]
        for log_sub_sub_dir in log_sub_sub_dirs:
            if not os.path.isdir(log_sub_sub_dir):
                os.mkdir(log_sub_sub_dir)

    num_iterations_finished = 0
    for (regions_filename, is_train_set) in metadata:
        print 'is_train_set =', is_train_set
        for label in label_names:
            with open(regions_filename) as regions_file:
                regions = regions_file.read().split('\n')

            if is_train_set:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir,
                                                         'train', label + '.sh'])
            else:
                condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir,
                                                         'test', label + '.sh'])

            condor_script_file = open(condor_script_file_name, 'w')
            condor_script_file.write('universe = vanilla\n')
            condor_script_file.write('Initialdir = ' +
                                     '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

            condor_script_file.write('Executable = /lusr/bin/python\n')

            condor_script_file.write('Arguments = create_classifier_data.py \\\n')
            condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
            condor_script_file.write('\t\t --write-individual-labels \\\n')
            condor_script_file.write('\t\t --label=' + label + ' \\\n')
            if is_train_set:
                condor_script_file.write('\t\t --in-train-set \\\n')
            condor_script_file.write('\t\t --verbose\n')

            condor_script_file.write('+Group   = "GRAD"\n')
            condor_script_file.write('+Project = "AI_ROBOTICS"\n')
            condor_script_file.write('+ProjectDescription = "VisualGenome - Writing individual label data"\n')
            condor_script_file.write('JobBatchName = "VisualGenome - Writing individual label data"\n')
            condor_script_file.write('Requirements = InMastodon\n')

            if is_train_set:
                condor_script_file.write('Log = ' +
                                         os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                      'train/log', label + '.log']) + '\n')
                condor_script_file.write('Error = ' +
                                         os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                      'train/err', label + '.err']) + '\n')
                condor_script_file.write('Output = ' +
                                         os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                      'train/out', label + '.out']) + '\n')
            else:
                condor_script_file.write('Log = ' +
                                         os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                      'test/log', label + '.log']) + '\n')
                condor_script_file.write('Error = ' +
                                         os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                      'test/err', label + '.err']) + '\n')
                condor_script_file.write('Output = ' +
                                         os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                      'test/out', label + '.out']) + '\n')

            condor_script_file.write('Queue 1\n')
            condor_script_file.close()

            condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

            num_iterations_finished += 1
            if num_iterations_finished % 100 == 0:
                print num_iterations_finished, 'labels finished'

    condor_submit_file.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Visual Genome dataset dir')
    arg_parser.add_argument('--condor-dir', type=str, required=True,
                            help='Subdirectory under dataset-dir/condor_scripts and dataset-dir/condor_log')

    arg_parser.add_argument('--extract-regions-vgg-features', action="store_true", default=False,
                            help='Extract VGG features of regions')

    arg_parser.add_argument('--batch-size', type=int, default=65536,
                            help='Number of data points per file (features or labels)')
    arg_parser.add_argument('--write-multilabels', action="store_true", default=False,
                            help='Write multilabel vectors for a batch')
    arg_parser.add_argument('--write-individual-labels', action="store_true", default=False,
                            help='Write label vectors for a batch for a label')
    arg_parser.add_argument('--write-features', action="store_true", default=False,
                            help='Write features for a batch')

    arg_parser.add_argument('--densities-process-pair', action="store_true", default=False,
                            help='Step 1 for computing densities and neighbours')
    arg_parser.add_argument('--compute-densities', action="store_true", default=False,
                            help='Compute densities for batch i')
    arg_parser.add_argument('--compute-nbrs', action="store_true", default=False,
                            help='Compute neighbours for batch i')
    arg_parser.add_argument('--num-nbrs', type=int, default=50,
                            help='Number of nbrs to compute')
    arg_parser.add_argument('--sub-batch-size', type=int, default=32768,
                            help='For processing a pair of batches, number of rows of batch j to take')

    args = arg_parser.parse_args()

    if args.extract_regions_vgg_features:
        extract_regions_vgg_features(args)

    if args.write_multilabels:
        write_multilabels(args)

    if args.write_individual_labels:
        write_individual_labels(args)

    if args.write_features:
        write_features(args)

    if args.densities_process_pair:
        density_process_batch_pair(args)

    if args.compute_densities:
        compute_densities(args)

    if args.compute_nbrs:
        compute_nbrs(args)
