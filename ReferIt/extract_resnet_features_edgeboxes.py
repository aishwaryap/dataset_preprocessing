#!/usr/bin/env python
# Write scripts to extract Resnet features of edgeboxes

from argparse import ArgumentParser
import os
import sys
import re
sys.path.append('../utils')
from file_utils import create_dir


def script_start_str(venv_name, code_dir):
    script_str = 'source /scratch/cluster/aish/' + venv_name + '/bin/activate \n' + \
                 'export PYTHONPATH=/scratch/cluster/aish/' + venv_name + '/lib/python3.5/site-packages \n' + \
                 'export PATH=/scratch/cluster/aish/cudnn/cuda:/opt/cuda-9.0/:$PATH \n' + \
                 'export LD_LIBRARY_PATH=/scratch/cluster/aish/cudnn/cuda/lib64:/opt/cuda-9.0/lib64/ \n' + \
                 'export CUDA_HOME=/scratch/cluster/aish/cudnn/cuda/:/opt/cuda-9.0/ \n' + \
                 'cd ' + os.path.join(code_dir, 'utils') + '\n'
    return script_str


def train_cmd(image_list_file, output_file, args):
    cmd_str = 'python3 extract_resnet_fcn_features.py \\\n' + \
              '    --dataset-dir=' + args.dataset_dir + ' \\\n' + \
              '    --ckpt-path=' + args.resnet_ckpt_path + ' \\\n' + \
              '    --image-list-file=' + image_list_file + ' \\\n' + \
              '    --output-file=' + output_file + ' \n'
    return cmd_str


def script_per_file(image_list_file, orig_output_dir, args):
    orig_output_file = os.path.join(orig_output_dir, re.sub('.txt', '.hdf5', image_list_file))
    final_output_file = os.path.join(args.final_output_dir, re.sub('.txt', '.hdf5', image_list_file))
    script_str = train_cmd(image_list_file, orig_output_file, args)
    script_str += 'scp ' + orig_output_file + ' aish@hati:' + final_output_file + ' \n'
    return script_str


def bash_cmd(bash_script, out_file, err_file):
    return '(./' + bash_script + ' | tee ' + out_file + ') 3>&1 1>&2 2>&3 | tee ' + err_file + '\n'


def ssh_cmd(ssh_machine, scripts_dir, run_script):
    cmd_str = 'ssh aish@' + ssh_machine + \
              ' \'cd ' + scripts_dir + '; ' + \
              'screen -dmS resnet_edgeboxes bash -c ' + run_script + '\'\n'
    return cmd_str


def create_scripts(args):
    orig_output_dir = os.path.join(args.dataset_dir, args.orig_output_dir)
    create_dir(orig_output_dir)
    scripts_dir = os.path.join(*[args.dataset_dir, 'bash_scripts', args.scripts_dir])
    create_dir(scripts_dir)
    log_dir = os.path.join(*[args.dataset_dir, 'bash_log', args.scripts_dir])
    create_dir(log_dir)

    image_list_dir = os.path.join(args.dataset_dir, 'image_lists/referit_edgeboxes')
    image_lists = os.listdir(image_list_dir)
    num_machines = 15
    if len(image_lists) % num_machines == 0:
        num_lists_per_machine = (len(image_lists) // num_machines)
    else:
        num_lists_per_machine = (len(image_lists) // num_machines) + 1

    submit_file = os.path.join(scripts_dir, 'submit.sh')
    submit_file_handle = open(submit_file, 'w')

    for machine_idx in range(num_machines):
        script_name = 'script_' + str(machine_idx) + '.sh'
        main_script_file = os.path.join(scripts_dir, script_name)
        with open(main_script_file, 'w') as handle:
            code_dir = '/u/aish/Documents/Research/Code/dataset_preprocessing'
            handle.write(script_start_str(args.venv_name, code_dir))

            first_image_idx = machine_idx * num_lists_per_machine
            last_image_idx = min((machine_idx + 1) * num_lists_per_machine, len(image_lists))
            for image_list_idx in range(first_image_idx, last_image_idx):
                image_list_file = os.path.join(image_list_dir, image_lists[image_list_idx])
                handle.write(script_per_file(image_list_file, orig_output_dir, args))
        os.chmod(main_script_file, 0o777)

        out_file = os.path.join(log_dir, str(machine_idx) + '.out')
        err_file = os.path.join(log_dir, str(machine_idx) + '.err')
        run_file = os.path.join(scripts_dir, 'run_' + str(machine_idx) + '.sh')
        with open(run_file, 'w') as handle:
            handle.write(bash_cmd(script_name, out_file, err_file))
        os.chmod(run_file, 0o777)

        machine_name = 'titan-' + str(machine_idx + 1)
        submit_file_handle.write(ssh_cmd(machine_name, scripts_dir, run_file))

    submit_file_handle.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to ReferIt dataset')
    arg_parser.add_argument('--resnet-ckpt-path', type=str, required=True,
                            help='Resnet Checkpoint path')
    arg_parser.add_argument('--orig-output-dir', type=str, required=True,
                            help='Subdirectory under dataset-dir to originally store output')
    arg_parser.add_argument('--scripts-dir', type=str, required=True,
                            help='Subdirectory under dataset-dir/bash_scripts to store scripts')
    arg_parser.add_argument('--final-output-dir', type=str, required=True,
                            help='Full path to external directory to copy to')
    arg_parser.add_argument('--venv-name', type=str, required=True,
                            help='Name of virtualenv to use')

    args = arg_parser.parse_args()
    create_scripts(args)
