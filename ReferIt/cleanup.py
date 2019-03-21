#!/usr/bin/env python
# Moving resized_imcrop files into their directory
# These were created in the parent directory because I used + instead of os.path.join

__author__ = 'aishwarya'

import os
import re
import h5py
import traceback
import shutil
import paramiko
from scp import SCPClient


def move():
    parent_dir = '/scratch/cluster/aish/ReferIt'
    target_dir = '/scratch/cluster/aish/ReferIt/resized_imcrop'
    match_regex = 'resized_imcrop[0-9]+_[0-9]+.png'
    for filename in os.listdir(parent_dir):
        if re.match(match_regex, filename):
            orig_file = os.path.join(parent_dir, filename)
            new_file = os.path.join(target_dir, filename)
            os.rename(orig_file, new_file)


def move_edgeboxes(image_list_file, orig_dir, target_dir):
    with open(image_list_file) as handle:
        image_list = set(handle.read().splitlines())
    for filename in os.listdir(orig_dir):
        print('Processing:', filename)
        if re.sub('.hdf5', '', filename) in image_list:
            print('\tMoving', filename)
            orig_file = os.path.join(orig_dir, filename)
            new_file = os.path.join(target_dir, filename)
            shutil.move(orig_file, new_file)


def move_edgeboxes_remote(image_list_file, orig_dir, remote_target_dir, ssh_host, ssh_user):
    with open(image_list_file) as handle:
        image_list = set(handle.read().splitlines())
    ssh = None
    client = None
    try:
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=ssh_host, username=ssh_user)
        client = SCPClient(ssh.get_transport())
        for filename in os.listdir(orig_dir):
            print('Processing:', filename)
            if re.sub('.hdf5', '', filename) in image_list:
                print('\tMoving', filename)
                local_path = os.path.join(orig_dir, filename)
                remote_path = os.path.join(remote_target_dir, filename)
                client.put(local_path, remote_path)
                os.remove(local_path)
    except Exception:
        if ssh:
            ssh.close()
        if client:
            client.close()
        raise
    finally:
        if ssh:
            ssh.close()
        if client:
            client.close()


def rename():
    target_dir = '/scratch/cluster/aish/ReferIt/resized_imcrop'
    for filename in os.listdir(target_dir):
        orig_file = os.path.join(target_dir, filename)
        new_file = os.path.join(target_dir, re.sub('resized_imcrop', '', filename))
        os.rename(orig_file, new_file)


def fix_edgebox_hdf5():
    features_dir = '/u/aish/Documents/ReferIt_link/resnet_fcn_features/edgeboxes/'
    dataset_name_prefix = 'referit_edgeboxes/'
    bad_files_file = '/u/aish/Documents/temp/bad_edgeboxes_toshiba.txt'
    bad_files_handle = open(bad_files_file, 'w')
    checked_files_file = '/u/aish/Documents/temp/checked_edgeboxes_toshiba.txt'
    problem_files = ['1088.hdf5', '6763.hdf5', '40297.hdf5', '8461.hdf5', '7686.hdf5', '7692.hdf5']
    with open(checked_files_file, 'r') as handle:
        files_to_skip = problem_files + handle.read().splitlines()
    checked_files_handle = open(checked_files_file, 'a')

    #files_to_clean = [f for f in os.listdir(features_dir) if f not in files_to_skip]
    files_to_clean = problem_files

    for filename in files_to_clean:
        print('Processing file', filename)

        orig_file = os.path.join(features_dir, filename)
        try:
            orig_handle = h5py.File(orig_file, 'r')
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except:
            print(traceback.format_exc())
            bad_files_handle.write(filename + '\n')
            continue    # Problem opening file

        new_dataset_name = re.sub('.hdf5', '', filename)
        if new_dataset_name in orig_handle.keys():
            orig_handle.close()
            checked_files_handle.write(filename + '\n')
            checked_files_handle.flush()
            continue    # File is already fixed

        temp_file = os.path.join(features_dir, 'temp.hdf5')
        temp_handle = h5py.File(temp_file, 'w')
        orig_dataset_name = os.path.join(dataset_name_prefix, new_dataset_name)
        temp_handle.copy(source=orig_handle[orig_dataset_name], dest=new_dataset_name)
        temp_handle.close()
        os.remove(orig_file)
        os.rename(temp_file, orig_file)
        checked_files_handle.write(filename + '\n')
        checked_files_handle.flush()

    bad_files_handle.close()
    checked_files_handle.close()


if __name__ == '__main__':
    orig_dir = '/scratch/cluster/aish/ReferIt/resnet_fcn_features/edgeboxes/'

    image_list_file = '/scratch/cluster/aish/ReferIt/split/referit_test_imlist.txt'
    target_dir = '/u/aish/Documents/ReferIt_link_T5/resnet_fcn_features/edgeboxes/'
    move_edgeboxes_remote(image_list_file, orig_dir, target_dir, 'hati', 'aish')
    image_list_file = '/scratch/cluster/aish/ReferIt/split/referit_train_imlist.txt'
    target_dir = '/u/aish/Documents/ReferIt_link/resnet_fcn_features/edgeboxes/'
    move_edgeboxes_remote(image_list_file, orig_dir, target_dir, 'hati', 'aish')
    image_list_file = '/scratch/cluster/aish/ReferIt/split/referit_val_imlist.txt'
    target_dir = '/u/aish/Documents/ReferIt_link_T5/resnet_fcn_features/edgeboxes/'
    move_edgeboxes_remote(image_list_file, orig_dir, target_dir, 'hati', 'aish')
