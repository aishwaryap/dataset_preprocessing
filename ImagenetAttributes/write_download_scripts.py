#!/usr/bin/env python

__author__ = 'aishwarya'

from argparse import ArgumentParser
import os


def write_download_scripts(args):
    url_files = os.listdir(args.image_urls_dir)
    condor_submit_file_name = os.path.join(args.condor_scripts_dir, 'submit.sh')
    condor_submit_file = open(condor_submit_file_name, 'w')

    for url_file in url_files:
        url_file_path = os.path.join(args.image_urls_dir, url_file)
        synset_name = url_file.split('.')[0]

        # Make directory for images
        image_dir = os.path.join(args.images_dir, synset_name)
        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)

        # Write bash script
        bash_script_file_name = os.path.join(args.bash_scripts_dir, synset_name + '.sh')
        bash_script_file = open(bash_script_file_name, 'w')
        bash_script_file.write('#!/usr/bin/env bash\n')
        bash_script_file.write('cat ' + url_file_path + ' | parallel --colsep ' ' wget -O {1} {2}\n')
        bash_script_file.close()

        # Write Condor script
        condor_script_file_name = os.path.join(args.condor_scripts_dir, synset_name + '.sh')
        condor_script_file = open(condor_script_file_name, 'w')
        condor_script_file.write('universe = vanilla\n')
        condor_script_file.write('Initialdir = ' + image_dir + '\n')
        condor_script_file.write('Executable = /lusr/bin/bash\n')
        condor_script_file.write('Arguments = ' + bash_script_file_name + '\n')
        condor_script_file.write('+Group   = "GRAD"\n')
        condor_script_file.write('+Project = "AI_ROBOTICS"\n')
        condor_script_file.write('+ProjectDescription = "ImageNet download"\n')
        condor_script_file.write('JobBatchName = "ImageNet download"\n')
        condor_script_file.write('Requirements = InMastodon && Narsil == True\n')
        condor_script_file.write('Log = ' + os.path.join(args.condor_log_dir, 'log/' + synset_name + '.log') + '\n')
        condor_script_file.write('Error = ' + os.path.join(args.condor_log_dir, 'err/' + synset_name + '.err') + '\n')
        condor_script_file.write('Output = ' + os.path.join(args.condor_log_dir, 'out/' + synset_name + '.out') + '\n')
        condor_script_file.write('Notification = complete\n')
        condor_script_file.write('Notify_user = aish@cs.utexas.edu\n')
        condor_script_file.write('Queue 1\n')
        condor_script_file.close()
        condor_submit_file.write('condor_submit ' + condor_script_file_name + '\n')

    condor_submit_file.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--image-urls-dir', type=str, required=True,
                            help='Dir for lists of paired ImageNet image names and URLs')
    arg_parser.add_argument('--images-dir', type=str, required=True,
                            help='Dir for images')
    arg_parser.add_argument('--bash-scripts-dir', type=str, required=True,
                            help='Dir for bash scripts that do the downloading')
    arg_parser.add_argument('--condor-scripts-dir', type=str, required=True,
                            help='Dir for condor scripts that submit the bash scripts')
    arg_parser.add_argument('--condor-log-dir', type=str, required=True,
                            help='Dir for condor logs')

    args = arg_parser.parse_args()
    write_download_scripts(args)