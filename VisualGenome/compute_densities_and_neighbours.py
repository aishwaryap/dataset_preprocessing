# Computes densities (average cosine similarity in VGG feature space) and k nearest neighbours for each region

import numpy as np
import os
import csv
import ast
import operator
from argparse import ArgumentParser
from sklearn.metrics.pairwise import cosine_similarity

__author__ = 'aishwarya'


def load_batch(filename, sub_batch_size=None, sub_batch_num=None):
    batch = list()
    with open(filename) as handle:
        reader = csv.reader(handle, delimiter=',')
        for (idx, row) in enumerate(reader):
            if sub_batch_num is None or sub_batch_size is None or \
                            idx in range(sub_batch_num*sub_batch_size, (sub_batch_num+1)*sub_batch_size):
                batch.append([float(x) for x in row])
        return np.array(batch)


# Read batches i and j of regions. Compute an i x j matrix of cosine similarities
# Store k nearest neighbours of each point and summed similarities
def process_batch_pair(args):
    print 'Reading regions ...'
    if args.in_train_set:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt')
    else:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt')
    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')

    print 'Loading batch i =', args.batch_num_i, '...'

    sum_cosine_sims_dir = os.path.join(args.dataset_dir, 'tmp_sum_cosine_sims/')
    if not os.path.isdir(sum_cosine_sims_dir):
        os.mkdir(sum_cosine_sims_dir)
    nbrs_dir = os.path.join(args.dataset_dir, 'tmp_nbrs/')
    if not os.path.isdir(nbrs_dir):
        os.mkdir(nbrs_dir)

    if args.in_train_set:
        batch_i_file = os.path.join(args.dataset_dir, 'classifiers/data/features/train/'
                                    + str(args.batch_num_i) + '.csv')

        sum_cosine_sims_dir = sum_cosine_sims_dir + 'train/'
        if not os.path.isdir(sum_cosine_sims_dir):
            os.mkdir(sum_cosine_sims_dir)
        nbrs_dir = nbrs_dir + 'train/'
        if not os.path.isdir(nbrs_dir):
            os.mkdir(nbrs_dir)
    else:
        batch_i_file = os.path.join(args.dataset_dir, 'classifiers/data/features/test/'
                                    + str(args.batch_num_i) + '.csv')

        sum_cosine_sims_dir = sum_cosine_sims_dir + 'test/'
        if not os.path.isdir(sum_cosine_sims_dir):
            os.mkdir(sum_cosine_sims_dir)
        nbrs_dir = nbrs_dir + 'test/'
        if not os.path.isdir(nbrs_dir):
            os.mkdir(nbrs_dir)

    batch_i = load_batch(batch_i_file)
    print 'Loaded batch i =', args.batch_num_i, '...'

    if args.batch_num_j is None or args.batch_num_i == args.batch_num_j:
        print 'Computing cosine sims ...'
        cosine_sims = cosine_similarity(batch_i,
                                        batch_i[
                                            args.sub_batch_num * args.sub_batch_size:
                                            min(args.sub_batch_num * (args.sub_batch_size + 1),
                                                batch_i.shape[0]), :])
        print 'Computed cosine sims ...'

        batch_j_regions = regions[args.batch_num_i * args.batch_size:
                                  min((args.batch_num_i + 1) * args.batch_size, len(regions))]
        if args.sub_batch_num is not None and args.sub_batch_size is not None:
            batch_j_regions = batch_j_regions[args.sub_batch_num * args.sub_batch_size:
                                              min((args.sub_batch_num + 1) * args.sub_batch_size, len(batch_j_regions))]

        sum_cosine_sims_file = os.path.join(sum_cosine_sims_dir,
                                            str(args.batch_num_i) + '_' + str(args.batch_num_i)
                                            + '_' + str(args.sub_batch_num) + '.csv')
        nbrs_file = os.path.join(nbrs_dir, str(args.batch_num_i) + '_' + str(args.batch_num_i)
                                 + '_' + str(args.sub_batch_num) + '.csv')
    else:
        print 'Loading batch j =', args.batch_num_j, '...'
        if args.in_train_set:
            batch_j_file = os.path.join(args.dataset_dir, 'classifiers/data/features/train/'
                                        + str(args.batch_num_j) + '.csv')
        else:
            batch_j_file = os.path.join(args.dataset_dir, 'classifiers/data/features/test/'
                                        + str(args.batch_num_j) + '.csv')
        batch_j = load_batch(batch_j_file, sub_batch_num=args.sub_batch_num, sub_batch_size=args.sub_batch_size)
        print 'Loaded batch j =', args.batch_num_j, '...'

        batch_j_regions = regions[args.batch_num_j * args.batch_size:
                                  min((args.batch_num_j+1) * args.batch_size, len(regions))]
        if args.sub_batch_num is not None and args.sub_batch_size is not None:
            batch_j_regions = batch_j_regions[args.sub_batch_num * args.sub_batch_size:
                                              min((args.sub_batch_num + 1) * args.sub_batch_size + 1, len(batch_j_regions))]

        print 'Computing cosine sims ...'
        cosine_sims = cosine_similarity(batch_i, batch_j)
        print 'Computed cosine sims ...'

        sum_cosine_sims_file = os.path.join(sum_cosine_sims_dir,
                                            str(args.batch_num_i) + '_' + str(args.batch_num_j)
                                            + '_' + str(args.sub_batch_num) + '.csv')
        nbrs_file = os.path.join(nbrs_dir, str(args.batch_num_i) + '_' + str(args.batch_num_j)
                                 + '_' + str(args.sub_batch_num) + '.csv')

    # Compute row sums of cosine sims
    print 'Computing row sums of cosine sims ...'
    sum_cosine_sims = np.sum(cosine_sims, axis=1)
    print 'Computed row sums of cosine sims ...'
    print 'Writing row sums of cosine sims ...'
    with open(sum_cosine_sims_file, 'w') as handle:
        writer = csv.writer(handle, delimiter=',')
        writer.writerow([float(x) for x in sum_cosine_sims])
    print 'Finished writing row sums of cosine sims ...'

    # Compute and store nearest neighbours with distances
    print 'Computing nearest neighbours ...'
    nbr_start_pos = cosine_sims.shape[1] - args.num_nbrs
    argpartition = np.argpartition(cosine_sims, nbr_start_pos, axis=1)
    print 'Computed argpartition ...'
    print 'Computing and writing neighbours ...'
    with open(nbrs_file, 'w') as handle:
        for row_num in range(cosine_sims.shape[0]):
            argpartition_row = argpartition[row_num, :]
            nbr_indices = argpartition_row[nbr_start_pos:].tolist()
            print nbr_indices
            nbrs = [batch_j_regions[x] for x in nbr_indices]
            cosine_sims_row = cosine_sims[row_num, :]
            nbr_cosine_sims = cosine_sims_row[nbr_indices]
            output_row = str(zip(nbrs, nbr_cosine_sims.tolist())) + '\n'
            handle.write(output_row)
    print 'Finished writing neighbours ...'


# Aggregate batchwise sums of cosine sims
def aggregate_batch_cosine_sims(args):
    partial_sum_cosine_sims_dir = os.path.join(args.dataset_dir, 'tmp_sum_cosine_sims/')
    densities_file = os.path.join(args.dataset_dir, 'densities/')
    if not os.path.isdir(densities_file):
        os.mkdir(densities_file)
    if args.in_train_set:
        partial_sum_cosine_sims_dir = partial_sum_cosine_sims_dir + 'train/'
        densities_file = densities_file + 'train/'
    else:
        partial_sum_cosine_sims_dir = partial_sum_cosine_sims_dir + 'test/'
        densities_file = densities_file + 'test/'
    if not os.path.isdir(densities_file):
        os.mkdir(densities_file)
    densities_file += str(args.batch_num_i) + '.csv'

    # Fetch files relevant to batch i
    partial_sum_cosine_sims_files = [os.path.join(partial_sum_cosine_sims_dir, f)
                                     for f in os.listdir(partial_sum_cosine_sims_dir)
                                     if f.startswith(str(args.batch_num_i) + '_')]
    sum_cosine_sims = None
    for (idx, filename) in enumerate(partial_sum_cosine_sims_files):
        partial_sum = load_batch(filename)
        if sum_cosine_sims is None:
            sum_cosine_sims = partial_sum
        else:
            sum_cosine_sims += partial_sum
        print idx + 1, 'partial sums added ...'

    print 'Reading regions ...'
    if args.in_train_set:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/train_regions.txt')
    else:
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/test_regions.txt')
    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')

    print 'Computing and writing densities ...'
    densities = sum_cosine_sims / len(regions)
    with open(densities_file, 'w') as handle:
        writer = csv.writer(handle, delimiter=',')
        writer.writerow([float(x) for x in densities])
    print 'Batch', args.batch_num_i, 'complete'


# Aggregate nearest neighbours
def aggregate_nbrs(args):
    partial_nbrs_dir = os.path.join(args.dataset_dir, 'tmp_nbrs/')
    nbrs_file = os.path.join(args.dataset_dir, 'nbrs/')
    if not os.path.isdir(nbrs_file):
        os.mkdir(nbrs_file)
    if args.in_train_set:
        partial_nbrs_dir = partial_nbrs_dir + 'train/'
        nbrs_file = nbrs_file + 'train/'
    else:
        partial_nbrs_dir = partial_nbrs_dir + 'test/'
        nbrs_file = nbrs_file + 'test/'
    if not os.path.isdir(nbrs_file):
        os.mkdir(nbrs_file)
    nbrs_file += str(args.batch_num_i) + '.txt'

    # Fetch files relevant to batch i
    partial_nbrs_files = [os.path.join(partial_nbrs_dir, f) for f in os.listdir(partial_nbrs_dir)
                          if f.startswith(str(args.batch_num_i) + '_')]
    handles = [open(filename) for filename in partial_nbrs_files]

    num_regions_done = 0
    with open(nbrs_file, 'w') as output_file:
        while True:
            try:
                partial_nbrs = list()
                for handle in handles:
                    line = handle.next().strip()
                    partial_nbrs += ast.literal_eval(line)
                partial_nbrs.sort(key=operator.itemgetter(1), reverse=True)
                nbrs = partial_nbrs[:args.num_neighbours]
                output_file.write(str(nbrs) + '\n')
                num_regions_done += 1
                if num_regions_done % 1000 == 0:
                    print num_regions_done, 'regions done'
            except StopIteration:
                break


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')

    arg_parser.add_argument('--process-batch-pair', action="store_true", default=False,
                            help='Initial processing with a pair of batches i and j')
    arg_parser.add_argument('--aggregate-densities', action="store_true", default=False,
                            help='Compute densities for batch i')
    arg_parser.add_argument('--aggregate-nbrs', action="store_true", default=False,
                            help='Compute neighbours for batch i')

    arg_parser.add_argument('--batch-num-i', type=int, required=True,
                            help='For processing a pair of batches - i; Also use this to specify batch num ' +
                                 'for single batch operations')
    arg_parser.add_argument('--batch-num-j', type=int, default=None,
                            help='For processing a pair of batches - j; May be skipped if i and j are the same')
    arg_parser.add_argument('--batch-size', type=int, default=65536,
                            help='Regions batch size')
    arg_parser.add_argument('--sub-batch-size', type=int, default=32768,
                            help='For processing a pair of batches, number of rows of batch j to take')
    arg_parser.add_argument('--sub-batch-num', type=int, default=None,
                            help='For processing a pair of batches, start of sub-batch in batch j')
    arg_parser.add_argument('--num-nbrs', type=int, required=True,
                            help='Number of nearest neighbours to be found per region')
    arg_parser.add_argument('--in-train-set', action="store_true", default=False,
                            help='To distinguish between whether batches are in train or test set')

    args = arg_parser.parse_args()

    if args.process_batch_pair:
        process_batch_pair(args)

    if args.aggregate_densities:
        aggregate_batch_cosine_sims(args)

    if args.aggregate_nbrs:
        aggregate_nbrs(args)
