#!/usr/bin/env python

from argparse import ArgumentParser
import os

__author__ = 'aishwarya'


def create_dirs(args):
    # Make scripts dir
    scripts_dir = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir])
    if not os.path.isdir(scripts_dir):
        os.mkdir(scripts_dir)

    # Make log dirs
    log_dir = os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_sub_dirs = [os.path.join(log_dir, d) for d in ['log', 'err', 'out', 'restart']]
    for log_sub_dir in log_sub_dirs:
        if not os.path.isdir(log_sub_dir):
            os.mkdir(log_sub_dir)


def extract_vgg_features(args):
    create_dirs(args)
    image_list_files = os.listdir(os.path.join(args.dataset_dir, 'Kitchen/split'))
    condor_submit_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir, 'submit.sh'])
    condor_submit_file = open(condor_submit_file_name, 'w')

    for image_list_file in image_list_files:
        image_list_file_path = os.path.join(*[args.dataset_dir, 'Kitchen/split', image_list_file])
        cropped_image_list_file = image_list_file[:-4]  # Remove extension

        # Write Condor script that submits the bash script on condor
        condor_script_file_name = os.path.join(*[args.dataset_dir, 'condor_scripts', args.condor_dir,
                                                 cropped_image_list_file + '.sh'])
        condor_script_file = open(condor_script_file_name, 'w')
        condor_script_file.write('universe = vanilla\n')
        condor_script_file.write('Initialdir = /u/aish/Documents/Research/Code/dataset_preprocessing/utils/\n')
        condor_script_file.write('Executable = /lusr/bin/python\n')
        condor_script_file.write('Arguments = extract_vgg_features.py \\\n')
        condor_script_file.write('\t\t --image-list-file=' + image_list_file_path + ' \\\n')
        condor_script_file.write('\t\t --output-file=' +
                                 os.path.join(*[args.dataset_dir, 'vgg_features/' +
                                                cropped_image_list_file + '.csv']) + ' \\\n')
        condor_script_file.write('\t\t --prototxt-file=/scratch/cluster/aish/CaffeModels/vgg7k.prototxt \\\n')
        condor_script_file.write('\t\t --caffemodel-file=/scratch/cluster/aish/CaffeModels/vgg7k.caffemodel \\\n')
        condor_script_file.write('\t\t --restart-log=' +
                                 os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir, 'restart',
                                                cropped_image_list_file + '.csv']) + ' \\\n')
        condor_script_file.write('\t\t --verbose\n')
        condor_script_file.write('+Group   = "GRAD"\n')
        condor_script_file.write('+Project = "AI_ROBOTICS"\n')
        condor_script_file.write('+ProjectDescription = "Kitchen dataset - Regions VGG-7k feature extraction"\n')
        condor_script_file.write('JobBatchName = "Kitchen dataset - Regions VGG-7k feature extraction"\n')
        condor_script_file.write('Requirements = TARGET.GPUSlot\n')
        condor_script_file.write('getenv = True\n')
        condor_script_file.write('request_GPUs = 1\n')
        condor_script_file.write('+GPUJob = true\n')
        condor_script_file.write('Log = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir, 'log',
                                                           cropped_image_list_file + '.log']) + '\n')
        condor_script_file.write('Error = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir, 'err',
                                                             cropped_image_list_file + '.err']) + '\n')
        condor_script_file.write('Output = ' + os.path.join(*[args.dataset_dir, 'condor_log', args.condor_dir, 'out',
                                                              cropped_image_list_file + '.out']) + '\n')
        condor_script_file.write('Queue 1\n')
        condor_script_file.close()

        condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Kitchen dataset dir')
    arg_parser.add_argument('--condor-dir', type=str, required=True,
                            help='Subdirectory under dataset-dir/condor_scripts and dataset-dir/condor_log')
    arg_parser.add_argument('--extract-vgg-features', action="store_true", default=False,
                            help='Extract VGG features')

    args = arg_parser.parse_args()
    if args.extract_vgg_features:
        extract_vgg_features(args)