#!/usr/bin/python
# This trains a binary SVM for one object/attribute using region VGG features

from argparse import ArgumentParser
import pickle
import os
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
import sys
import numpy as np
sys.path.append('../utils/')
from file_utils import *

__author__ = 'aishwarya'


def train_binary_classifier(args):
    # Create the restart log if it does not exist
    if not os.path.isfile(args.restart_log):
        if args.verbose:
            print 'Recreating restart log'
        restart_log = open(args.restart_log, 'w')
        restart_log.close()

    # Check restart log to find the line number to restart
    last_line = tail(args.restart_log)
    start_batch_num = 0
    if last_line is not None:
        start_batch_num = int(last_line.strip()) + 1
    if args.verbose:
        print 'Start batch num =', start_batch_num

    # Load the classifier or instantiate a new one
    classifier_file = os.path.join(args.dataset_dir, 'classifiers/binary_svms/' + args.label + '.pkl')
    if os.path.isfile(classifier_file):
        with open(classifier_file, 'rb') as classifier_file_handle:
            classifier = pickle.load(classifier_file_handle)
        if args.verbose:
            print 'Loaded existing classifier'
    else:
        if args.verbose:
            print 'Instantiated new classifier'
        classifier = SGDClassifier(loss='hinge')

    for batch_num in range(start_batch_num, args.max_train_batch_num + 1):
        if args.verbose:
            print 'Training batch', batch_num

        # Load features
        features_file = os.path.join(args.dataset_dir, 'classifiers/data/features/train/' + str(batch_num) + '.csv')
        features = np.loadtxt(features_file, delimiter=',')
        if args.verbose:
            print '\tLoaded features ...'

        # Load labels
        label_file = os.path.join(args.dataset_dir, 'classifiers/data/binary_labels/train/' + args.label + '/' +
                                   str(batch_num) + '.csv')
        print 'label_file = ', str(label_file)
        labels = np.loadtxt(label_file, delimiter=',', dtype=np.int)
        if args.verbose:
            print '\tLoaded labels ...'
        print 'Labels : ', str(labels)
        print 'Labels.shape : ', str(labels.shape)

        # Subsample if needed
        if args.num_samples_per_batch < args.batch_size:
            indices = range(len(labels))
            selected_indices = np.random.choice(indices, size=args.num_samples_per_batch, replace=False)
            features = features[selected_indices, :]
            labels = labels[selected_indices]
        print 'Labels : ', str(labels)
        print 'Labels.shape : ', str(labels.shape)

        # Update classifier
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=labels)
        sample_weights = [class_weights[label] for label in labels.tolist()]
        classifier = classifier.partial_fit(features, labels, classes=[0, 1], sample_weight=sample_weights)
        if args.verbose:
            print '\tUpdated classifier ...'

        # Save classifier checkpoint
        with open(classifier_file, 'wb') as classifier_file_handle:
            pickle.dump(classifier, classifier_file_handle)
        if args.verbose:
            print '\tSaved updated classifier ...'

        # Update restart log
        restart_log = open(args.restart_log, 'a')
        restart_log.write('\n' + str(batch_num))
        restart_log.close()
        if args.verbose:
            print '\tUpdated restart log ...'


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--verbose', action="store_true", default=False,
                            help='Print debug output')
    arg_parser.add_argument('--batch-size', type=int, default=65536,
                            help='Number of data points per file (features or labels)')
    arg_parser.add_argument('--label', type=str, required=True,
                            help='Label to train classifier for')
    arg_parser.add_argument('--max-train-batch-num', type=int, default=0,
                            help='Give this as an argument to prevent reading train regions file')
    arg_parser.add_argument('--num-samples-per-batch', type=int, default=65536,
                            help='To constrain the number of training examples')
    arg_parser.add_argument('--restart-log', type=str, required=True,
                            help='A file to log progress to handle restarts')
    args = arg_parser.parse_args()
    train_binary_classifier(args)
