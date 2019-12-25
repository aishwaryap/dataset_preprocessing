#!/usr/bin/env python

from argparse import ArgumentParser
import os
import re
import math
import csv
import operator
import itertools

__author__ = 'aishwarya'


def create_dirs(args, train_test=True):
    # Make scripts dir
    scripts_dir = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir])
    if not os.path.isdir(scripts_dir):
        os.mkdir(scripts_dir)
    if train_test:
        scripts_sub_dirs = [os.path.join(scripts_dir, d) for d in ['train', 'test']]
        for scripts_sub_dir in scripts_sub_dirs:
            if not os.path.isdir(scripts_sub_dir):
                os.mkdir(scripts_sub_dir)

    # Make log dirs
    log_dir = os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if train_test:
        log_sub_dirs = [os.path.join(log_dir, d) for d in ['train', 'test']]
        for log_sub_dir in log_sub_dirs:
            if not os.path.isdir(log_sub_dir):
                os.mkdir(log_sub_dir)
            log_sub_sub_dirs = [os.path.join(log_sub_dir, d) for d in ['log', 'err', 'out']]
            for log_sub_sub_dir in log_sub_sub_dirs:
                if not os.path.isdir(log_sub_sub_dir):
                    os.mkdir(log_sub_sub_dir)
    else:
        log_sub_dirs = [os.path.join(log_dir, d) for d in ['log', 'err', 'out', 'restart']]
        for log_sub_dir in log_sub_dirs:
            if not os.path.isdir(log_sub_dir):
                os.mkdir(log_sub_dir)


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


def split_region_features(args):
    # Pairs of region file name and is_train_set
    metadata = [
        (os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt'), True),
        (os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt'), False)
    ]

    create_dirs(args)

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

            condor_script_file.write('Arguments = split_region_features.py \\\n')
            condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
            if is_train_set:
                condor_script_file.write('\t\t --in-train-set \\\n')
            condor_script_file.write('\t\t --batch-num=' + str(batch_num) + ' \n')

            condor_script_file.write('+Group   = "GRAD"\n')
            condor_script_file.write('+Project = "AI_ROBOTICS"\n')
            condor_script_file.write('+ProjectDescription = "VisualGenome - Splitting features"\n')
            condor_script_file.write('JobBatchName = "VisualGenome - Splitting features"\n')
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


# Scripts for first step in computing densities
def density_process_batch_pair(args):
    if args.split_group == 'orig':
        splits = ['train', 'test']
        regions_files = [os.path.join(args.dataset_dir, 'classifiers/data/', split + '_regions.txt')
                         for split in splits]
    else:
        if args.split_group == 'predicate_novelty':
            split_parts = [['policy'], ['train', 'val', 'test'], ['classifier'], ['train', 'val', 'test']]
        else:
            split_parts = [['policy'], ['pretrain', 'train', 'val', 'test'], ['classifier'], ['train', 'test']]
        splits = ['_'.join(x) for x in list(itertools.product(*split_parts))]
        regions_files = [os.path.join(args.dataset_dir, 'split', args.split_group, split + '_regions.txt')
                         for split in splits]

    create_dirs(args, train_test=False)

    condor_submit_file_name = os.path.join(args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh')
    condor_submit_file = open(condor_submit_file_name, 'w')

    for regions_filename, split in zip(regions_files, splits):
        print 'split =', split
        print 'regions_filename =', regions_filename
        with open(regions_filename) as regions_file:
            regions = regions_file.read().split('\n')
            num_regions = len(regions)
        max_batch_num = int(math.ceil(float(num_regions) / args.batch_size))

        condor_scripts_subdir = os.path.join(args.dataset_dir, 'condor_scripts', args.condor_dir)
        condor_log_dirs = [os.path.join(args.dataset_dir, 'condor_log', args.condor_dir, log_type)
                           for log_type in ['err', 'out', 'log']]
        for condor_dir in [condor_scripts_subdir] + condor_log_dirs:
            split_condor_dir = os.path.join(condor_dir, split)
            if not os.path.isdir(split_condor_dir):
                os.mkdir(split_condor_dir)

        for batch_num_i in range(max_batch_num):
            batch_i_start_idx = batch_num_i * args.batch_size
            batch_i_end_idx = min(batch_i_start_idx + args.batch_size, num_regions)
            batch_i_size = batch_i_end_idx - batch_i_start_idx
            for batch_num_j in range(max_batch_num):
                batch_j_start_idx = batch_num_j * args.batch_size
                batch_j_end_idx = min(batch_j_start_idx + args.batch_size, num_regions)
                batch_j_size = batch_j_end_idx - batch_j_start_idx
                max_sub_batch_num = min(int(math.ceil(float(batch_i_size) / args.sub_batch_size)),
                                        int(math.ceil(float(batch_j_size) / args.sub_batch_size)))

                for sub_batch_num in range(max_sub_batch_num):
                    condor_script_file_name = os.path.join(condor_scripts_subdir, split,
                                                           str(batch_num_i) + '_' + str(batch_num_j)
                                                           + '_' + str(sub_batch_num) + '.sh')

                    condor_script_file = open(condor_script_file_name, 'w')
                    condor_script_file.write('universe = vanilla\n')
                    condor_script_file.write('Initialdir = ' +
                                             '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

                    condor_script_file.write('Executable = /lusr/bin/python\n')

                    condor_script_file.write('Arguments = compute_densities_and_neighbours.py \\\n')
                    condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
                    condor_script_file.write('\t\t --split-group=' + str(args.split_group) + ' \\\n')
                    condor_script_file.write('\t\t --region-set=' + str(split) + ' \\\n')
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
                    condor_script_file.write('request_memory = 15 GB\n')

                    condor_script_file.write('Log = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                                     'log', split, str(batch_num_i) + '_' +
                                                                     str(batch_num_j) + '_' + str(sub_batch_num) +
                                                                     '.log') + '\n')
                    condor_script_file.write('Error = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                                       'err', split, str(batch_num_i) + '_' +
                                                                       str(batch_num_j) + '_' + str(sub_batch_num) +
                                                                       '.err') + '\n')
                    condor_script_file.write('Output = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                                        'out', split, str(batch_num_i) + '_' +
                                                                        str(batch_num_j) + '_' + str(sub_batch_num) +
                                                                        '.out') + '\n')

                    condor_script_file.write('Queue 1\n')
                    condor_script_file.close()

                    condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


def resubmit_process_batch_pair(args):
    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    if args.split_group == 'orig':
        splits = ['train', 'test']
    else:
        if args.split_group == 'predicate_novelty':
            split_parts = [['policy'], ['train', 'val', 'test'], ['classifier'], ['train', 'val', 'test']]
        else:
            split_parts = [['policy'], ['pretrain', 'train', 'val', 'test'], ['classifier'], ['train', 'test']]
        splits = ['_'.join(x) for x in list(itertools.product(*split_parts))]

    for split in splits:
        output_dirs = [os.path.join(*[args.dataset_dir, subdir, args.split_group, split]) for subdir in
                       ['tmp_sum_cosine_sims', 'tmp_nbrs', 'tmp_farthest_nbrs']]
        condor_scripts_dir = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, split])
        scripts = [re.sub('.sh', '', f) for f in os.listdir(condor_scripts_dir)]
        completed_scripts = set(scripts)
        for output_dir in output_dirs:
            completed_scripts = completed_scripts.intersection([re.sub('.csv', '', f) for f in
                                                                os.listdir(output_dir)])
        incomplete_scripts = set(scripts).difference(completed_scripts)
        submit_cmds = ['condor_submit ' + os.path.join(split, f + '.sh') for f in incomplete_scripts]
        condor_submit_file.write('\n'.join(submit_cmds) + '\n')


# Scripts to aggregate densities
def compute_densities(args):
    if args.split_group == 'orig':
        splits = ['train', 'test']
        regions_files = [os.path.join(args.dataset_dir, 'classifiers/data/', split + '_regions.txt')
                         for split in splits]
    else:
        if args.split_group == 'predicate_novelty':
            split_parts = [['policy'], ['train', 'val', 'test'], ['classifier'], ['train', 'val', 'test']]
        else:
            split_parts = [['policy'], ['pretrain', 'train', 'val', 'test'], ['classifier'], ['train', 'test']]
        splits = ['_'.join(x) for x in list(itertools.product(*split_parts))]
        regions_files = [os.path.join(args.dataset_dir, 'split', args.split_group, split + '_regions.txt')
                         for split in splits]

    create_dirs(args, train_test=False)

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    for regions_filename, split in zip(regions_files, splits):
        with open(regions_filename) as regions_file:
            regions = regions_file.read().split('\n')
        max_batch_num = int(math.ceil(float(len(regions)) / args.batch_size))

        condor_scripts_subdir = os.path.join(args.dataset_dir, 'condor_scripts', args.condor_dir)
        condor_log_dirs = [os.path.join(args.dataset_dir, 'condor_log', args.condor_dir, log_type)
                           for log_type in ['err', 'out', 'log']]
        for condor_dir in [condor_scripts_subdir] + condor_log_dirs:
            split_condor_dir = os.path.join(condor_dir, split)
            if not os.path.isdir(split_condor_dir):
                os.mkdir(split_condor_dir)

        for batch_num in range(max_batch_num):
            condor_script_file_name = os.path.join(condor_scripts_subdir, split, str(batch_num) + '.sh')

            condor_script_file = open(condor_script_file_name, 'w')
            condor_script_file.write('universe = vanilla\n')
            condor_script_file.write('Initialdir = ' +
                                     '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

            condor_script_file.write('Executable = /lusr/bin/python\n')

            condor_script_file.write('Arguments = compute_densities_and_neighbours.py \\\n')
            condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
            condor_script_file.write('\t\t --split-group=' + str(args.split_group) + ' \\\n')
            condor_script_file.write('\t\t --region-set=' + str(split) + ' \\\n')
            condor_script_file.write('\t\t --aggregate-densities \\\n')
            condor_script_file.write('\t\t --batch-num-i=' + str(batch_num) + ' \\\n')
            condor_script_file.write('\t\t --num-nbrs=' + str(args.num_nbrs) + ' \n')

            condor_script_file.write('+Group   = "GRAD"\n')
            condor_script_file.write('+Project = "AI_ROBOTICS"\n')
            condor_script_file.write('+ProjectDescription = "VisualGenome - Writing features"\n')
            condor_script_file.write('JobBatchName = "VisualGenome - Writing features"\n')
            condor_script_file.write('Requirements = InMastodon\n')
            condor_script_file.write('request_memory = 15 GB\n')

            condor_script_file.write('Log = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                             'log', split, str(batch_num) + '.log') + '\n')
            condor_script_file.write('Error = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                               'err', split, str(batch_num) + '.err') + '\n')
            condor_script_file.write('Output = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                                'out', split, str(batch_num) + '.out') + '\n')

            condor_script_file.write('Queue 1\n')
            condor_script_file.close()

            condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


# Scripts to aggregate neighbours
def compute_nbrs(args):
    if args.split_group == 'orig':
        splits = ['train', 'test']
        regions_files = [os.path.join(args.dataset_dir, 'classifiers/data/', split + '_regions.txt')
                         for split in splits]
    else:
        if args.split_group == 'predicate_novelty':
            split_parts = [['policy'], ['train', 'val', 'test'], ['classifier'], ['train', 'val', 'test']]
        else:
            split_parts = [['policy'], ['pretrain', 'train', 'val', 'test'], ['classifier'], ['train', 'test']]
        splits = ['_'.join(x) for x in list(itertools.product(*split_parts))]
        regions_files = [os.path.join(args.dataset_dir, 'split', args.split_group, split + '_regions.txt')
                         for split in splits]

    create_dirs(args, train_test=False)

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    for regions_filename, split in zip(regions_files, splits):
        with open(regions_filename) as regions_file:
            regions = regions_file.read().split('\n')
        max_batch_num = int(math.ceil(float(len(regions)) / args.batch_size))

        condor_scripts_subdir = os.path.join(args.dataset_dir, 'condor_scripts', args.condor_dir)
        condor_log_dirs = [os.path.join(args.dataset_dir, 'condor_log', args.condor_dir, log_type)
                           for log_type in ['err', 'out', 'log']]
        for condor_dir in [condor_scripts_subdir] + condor_log_dirs:
            split_condor_dir = os.path.join(condor_dir, split)
            if not os.path.isdir(split_condor_dir):
                os.mkdir(split_condor_dir)

        for batch_num in range(max_batch_num):
            condor_script_file_name = os.path.join(condor_scripts_subdir, split, str(batch_num) + '.sh')

            condor_script_file = open(condor_script_file_name, 'w')
            condor_script_file.write('universe = vanilla\n')
            condor_script_file.write('Initialdir = ' +
                                     '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

            condor_script_file.write('Executable = /lusr/bin/python\n')

            condor_script_file.write('Arguments = compute_densities_and_neighbours.py \\\n')
            condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
            condor_script_file.write('\t\t --split-group=' + str(args.split_group) + ' \\\n')
            condor_script_file.write('\t\t --region-set=' + str(split) + ' \\\n')
            condor_script_file.write('\t\t --aggregate-nbrs \\\n')
            condor_script_file.write('\t\t --batch-num-i=' + str(batch_num) + ' \\\n')
            condor_script_file.write('\t\t --num-nbrs=' + str(args.num_nbrs) + ' \n')
            condor_script_file.write('request_memory = 15 GB\n')

            condor_script_file.write('+Group   = "GRAD"\n')
            condor_script_file.write('+Project = "AI_ROBOTICS"\n')
            condor_script_file.write('+ProjectDescription = "VisualGenome - Writing features"\n')
            condor_script_file.write('JobBatchName = "VisualGenome - Writing features"\n')
            condor_script_file.write('Requirements = InMastodon\n')

            condor_script_file.write('Log = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                             'log', split, str(batch_num) + '.log') + '\n')
            condor_script_file.write('Error = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                               'err', split, str(batch_num) + '.err') + '\n')
            condor_script_file.write('Output = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                                'out', split, str(batch_num) + '.out') + '\n')

            condor_script_file.write('Queue 1\n')
            condor_script_file.close()

            condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


# Scripts to aggregate neighbours
def compute_farthest_nbrs(args):
    if args.split_group == 'orig':
        splits = ['train', 'test']
        regions_files = [os.path.join(args.dataset_dir, 'classifiers/data/', split + '_regions.txt')
                         for split in splits]
    else:
        if args.split_group == 'predicate_novelty':
            split_parts = [['policy'], ['train', 'val', 'test'], ['classifier'], ['train', 'val', 'test']]
        else:
            split_parts = [['policy'], ['pretrain', 'train', 'val', 'test'], ['classifier'], ['train', 'test']]
        splits = ['_'.join(x) for x in list(itertools.product(*split_parts))]
        regions_files = [os.path.join(args.dataset_dir, 'split', args.split_group, split + '_regions.txt')
                         for split in splits]

    create_dirs(args, train_test=False)

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    for regions_filename, split in zip(regions_files, splits):
        with open(regions_filename) as regions_file:
            regions = regions_file.read().split('\n')
        max_batch_num = int(math.ceil(float(len(regions)) / args.batch_size))

        condor_scripts_subdir = os.path.join(args.dataset_dir, 'condor_scripts', args.condor_dir)
        condor_log_dirs = [os.path.join(args.dataset_dir, 'condor_log', args.condor_dir, log_type)
                           for log_type in ['err', 'out', 'log']]
        for condor_dir in [condor_scripts_subdir] + condor_log_dirs:
            split_condor_dir = os.path.join(condor_dir, split)
            if not os.path.isdir(split_condor_dir):
                os.mkdir(split_condor_dir)

        for batch_num in range(max_batch_num):
            condor_script_file_name = os.path.join(condor_scripts_subdir, split, str(batch_num) + '.sh')

            condor_script_file = open(condor_script_file_name, 'w')
            condor_script_file.write('universe = vanilla\n')
            condor_script_file.write('Initialdir = ' +
                                     '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

            condor_script_file.write('Executable = /lusr/bin/python\n')

            condor_script_file.write('Arguments = compute_densities_and_neighbours.py \\\n')
            condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
            condor_script_file.write('\t\t --split-group=' + str(args.split_group) + ' \\\n')
            condor_script_file.write('\t\t --region-set=' + str(split) + ' \\\n')
            condor_script_file.write('\t\t --aggregate-farthest-nbrs \\\n')
            condor_script_file.write('\t\t --batch-num-i=' + str(batch_num) + ' \n')

            condor_script_file.write('+Group   = "GRAD"\n')
            condor_script_file.write('+Project = "AI_ROBOTICS"\n')
            condor_script_file.write('+ProjectDescription = "VisualGenome - Writing features"\n')
            condor_script_file.write('JobBatchName = "VisualGenome - Writing features"\n')
            condor_script_file.write('Requirements = InMastodon\n')
            condor_script_file.write('request_memory = 15 GB\n')

            condor_script_file.write('Log = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                             'log', split, str(batch_num) + '.log') + '\n')
            condor_script_file.write('Error = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                               'err', split, str(batch_num) + '.err') + '\n')
            condor_script_file.write('Output = ' + os.path.join(args.dataset_dir, 'condor_log', args.condor_dir,
                                                                'out', split, str(batch_num) + '.out') + '\n')

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

    create_dirs(args)

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

    create_dirs(args)

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

    create_dirs(args)

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


# Trains binary classifiers for num-labels most frequent objects and attributes, each with num-examples examples from
# batches 0 - max-train-batch-num
def train_binary_classifiers(args):
    create_dirs(args, train_test=False)

    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    labels = list()
    object_stats_file = os.path.join(args.dataset_dir, 'indoor/region_objects_stats.csv')
    with open(object_stats_file) as handle:
        reader = csv.reader(handle, delimiter=',')
        for (row_idx, row) in enumerate(reader):
            if row_idx >= args.num_labels:
                break
            labels.append((row[0], int(row[1])))
    attributes_stats_file = os.path.join(args.dataset_dir, 'indoor/region_attributes_stats.csv')
    with open(attributes_stats_file) as handle:
        reader = csv.reader(handle, delimiter=',')
        for (row_idx, row) in enumerate(reader):
            if row_idx >= args.num_labels:
                break
            labels.append((row[0], int(row[1])))

    labels.sort(key=operator.itemgetter(1), reverse=True)
    labels = labels[:args.num_labels]

    for (label, count) in labels:
        print 'Processing label', label
        condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, label + '.sh'])

        condor_script_file = open(condor_script_file_name, 'w')
        condor_script_file.write('universe = vanilla\n')
        condor_script_file.write('Initialdir = ' +
                                 '/u/aish/Documents/Research/Code/dataset_preprocessing/VisualGenome/\n')

        condor_script_file.write('Executable = /lusr/bin/python\n')

        condor_script_file.write('Arguments = train_binary_classifier.py \\\n')
        condor_script_file.write('\t\t --dataset-dir=/scratch/cluster/aish/VisualGenome \\\n')
        condor_script_file.write('\t\t --label=' + label + ' \\\n')
        condor_script_file.write('\t\t --num-samples-per-batch=' + str(args.num_examples) + ' \\\n')
        restart_log_file = os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir, 'restart', label + '.txt'])
        condor_script_file.write('\t\t --restart-log=' + restart_log_file + ' \\\n')
        condor_script_file.write('\t\t --verbose\n')

        condor_script_file.write('+Group   = "GRAD"\n')
        condor_script_file.write('+Project = "AI_ROBOTICS"\n')
        condor_script_file.write('+ProjectDescription = "VisualGenome - Training classifiers"\n')
        condor_script_file.write('JobBatchName = "VisualGenome - Training classifiers"\n')
        condor_script_file.write('Requirements = InMastodon\n')

        condor_script_file.write('Log = ' +
                                 os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                'log', label + '.log']) + '\n')
        condor_script_file.write('Error = ' +
                                 os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                'err', label + '.err']) + '\n')
        condor_script_file.write('Output = ' +
                                 os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir,
                                                'out', label + '.out']) + '\n')

        condor_script_file.write('Queue 1\n')
        condor_script_file.close()

        condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Visual Genome dataset dir')
    arg_parser.add_argument('--condor-dir', type=str, required=True,
                            help='Subdirectory under dataset-dir/condor_scripts and dataset-dir/condor_log')
    arg_parser.add_argument('--split-group', type=str, required=True,
                            help='Which split of dataset - one of "orig", "predicate_novelty", "8_way"')

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
    arg_parser.add_argument('--resubmit-process-pair', action="store_true", default=False,
                            help='Find scripts that didn\'t finish step 1')
    arg_parser.add_argument('--compute-densities', action="store_true", default=False,
                            help='Compute densities for batch i')
    arg_parser.add_argument('--compute-nbrs', action="store_true", default=False,
                            help='Compute neighbours for batch i')
    arg_parser.add_argument('--compute-farthest-nbrs', action="store_true", default=False,
                            help='Compute farthest neighbours for batch i')
    arg_parser.add_argument('--num-nbrs', type=int, default=50,
                            help='Number of nbrs to compute')
    arg_parser.add_argument('--sub-batch-size', type=int, default=8192,
                            help='For processing a pair of batches, number of rows of batch j to take')

    arg_parser.add_argument('--split-region-features', action="store_true", default=False,
                            help='Split region features')

    arg_parser.add_argument('--train-binary-classifiers', action="store_true", default=False,
                            help='Train binary classifiers for frequent objects and attributes')
    arg_parser.add_argument('--num-labels', type=int, default=50,
                            help='Number of labels to learn classifiers for')
    arg_parser.add_argument('--num-examples', type=int, default=100,
                            help='Number of examples per batch to train classifiers for')

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

    if args.compute_farthest_nbrs:
        compute_farthest_nbrs(args)

    if args.split_region_features:
        split_region_features(args)

    if args.train_binary_classifiers:
        train_binary_classifiers(args)

    if args.resubmit_process_pair:
        resubmit_process_batch_pair(args)
