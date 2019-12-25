# Computes densities (average cosine similarity in VGG feature space) and k nearest neighbours for each region

import numpy as np
import os
import csv
import ast
import operator
from argparse import ArgumentParser
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_dir

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


def get_regions_file(args):
    if args.split_group == 'orig':
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/', args.region_set + '_regions.txt')
    else:
        regions_filename = os.path.join(args.dataset_dir, 'split', args.split_group, args.region_set + '_regions.txt')
    return regions_filename


def get_features_dir(args):
    if args.split_group == 'orig':
        regions_filename = os.path.join(args.dataset_dir, 'classifiers/data/features', args.region_set)
    else:
        regions_filename = os.path.join(args.dataset_dir, 'vgg_features', args.split_group, args.region_set)
    return regions_filename


# Read batches i and j of regions. Compute an i x j matrix of cosine similarities
# Store k nearest neighbours of each point and summed similarities
def process_batch_pair(args):
    print 'Reading regions ...'
    regions_filename = get_regions_file(args)
    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')

    print 'Loading batch i =', args.batch_num_i, '...'

    sum_cosine_sims_dir = os.path.join(args.dataset_dir, 'tmp_sum_cosine_sims', args.split_group, args.region_set)
    create_dir(sum_cosine_sims_dir)
    nbrs_dir = os.path.join(args.dataset_dir, 'tmp_nbrs', args.split_group, args.region_set)
    create_dir(nbrs_dir)
    farthest_nbrs_dir = os.path.join(args.dataset_dir, 'tmp_farthest_nbrs', args.split_group, args.region_set)
    create_dir(farthest_nbrs_dir)

    features_dir = get_features_dir(args)
    batch_i_file = os.path.join(features_dir, str(args.batch_num_i) + '.csv')
    batch_j_file = os.path.join(features_dir, str(args.batch_num_j) + '.csv')

    batch_i = load_batch(batch_i_file)
    print 'Loaded batch i =', args.batch_num_i, '...'

    sum_cosine_sims_file = os.path.join(sum_cosine_sims_dir,
                                        str(args.batch_num_i) + '_' + str(args.batch_num_j)
                                        + '_' + str(args.sub_batch_num) + '.csv')
    nbrs_file = os.path.join(nbrs_dir, str(args.batch_num_i) + '_' + str(args.batch_num_j)
                             + '_' + str(args.sub_batch_num) + '.csv')
    farthest_nbrs_file = os.path.join(farthest_nbrs_dir, str(args.batch_num_i) + '_' + str(args.batch_num_j)
                                      + '_' + str(args.sub_batch_num) + '.csv')

    if args.batch_num_j is None or args.batch_num_i == args.batch_num_j:
        print 'Computing cosine sims ...'
        cosine_sims = cosine_similarity(batch_i,
                                        batch_i[
                                            args.sub_batch_num * args.sub_batch_size:
                                            min((args.sub_batch_num + 1) * args.sub_batch_size,
                                                batch_i.shape[0]), :])
        print 'Computed cosine sims ...'

        batch_j_regions = regions[args.batch_num_i * args.batch_size:
                                  min((args.batch_num_i + 1) * args.batch_size, len(regions))]
        if args.sub_batch_num is not None and args.sub_batch_size is not None:
            batch_j_regions = batch_j_regions[args.sub_batch_num * args.sub_batch_size:
                                              min((args.sub_batch_num + 1) * args.sub_batch_size, len(batch_j_regions))]

    else:
        print 'Loading batch j =', args.batch_num_j, '...'
        batch_j = load_batch(batch_j_file, sub_batch_num=args.sub_batch_num, sub_batch_size=args.sub_batch_size)
        print 'Loaded batch j =', args.batch_num_j, '...'

        batch_j_regions = regions[args.batch_num_j * args.batch_size:
                                  min((args.batch_num_j+1) * args.batch_size, len(regions))]
        if args.sub_batch_num is not None and args.sub_batch_size is not None:
            batch_j_regions = batch_j_regions[args.sub_batch_num * args.sub_batch_size:
                                              min((args.sub_batch_num + 1) * args.sub_batch_size, len(batch_j_regions))]

        print 'Computing cosine sims ...'
        cosine_sims = cosine_similarity(batch_i, batch_j)
        print 'Computed cosine sims ...'

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

    # Compute farthest nbrs
    sims_argmin = np.argmin(cosine_sims, axis=1)
    print 'Computed argmin ...'
    print 'Computing and writing farthest neighbours ...'
    with open(farthest_nbrs_file, 'w') as handle:
        for row_num in range(cosine_sims.shape[0]):
            farthest_nbr_idx = sims_argmin[row_num]
            farthest_nbr_sim = cosine_sims[row_num, farthest_nbr_idx]
            farthest_nbr = batch_j_regions[farthest_nbr_idx]
            output_row = str([(farthest_nbr, farthest_nbr_sim)]) + '\n'
            handle.write(output_row)
    print 'Finished writing farthest neighbours ...'


# Aggregate batchwise sums of cosine sims
def aggregate_batch_cosine_sims(args):
    partial_sum_cosine_sims_dir = os.path.join(args.dataset_dir, 'tmp_sum_cosine_sims', args.split_group,
                                               args.region_set)
    densities_dir = os.path.join(args.dataset_dir, 'densities', args.split_group, args.region_set)
    create_dir(densities_dir)
    densities_file = os.path.join(densities_dir, str(args.batch_num_i) + '.csv')

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

    regions_filename = get_regions_file(args)
    with open(regions_filename) as regions_file:
        regions = regions_file.read().split('\n')

    print 'Computing and writing densities ...'
    densities = sum_cosine_sims / len(regions)
    with open(densities_file, 'w') as handle:
        writer = csv.writer(handle, delimiter=',')
        writer.writerow([float(x) for [x] in densities.T.tolist()])
    print 'Batch', args.batch_num_i, 'complete'


# Aggregate nearest neighbours
def aggregate_nbrs(args):
    print 'Computing nbrs'
    partial_nbrs_dir = os.path.join(args.dataset_dir, 'tmp_nbrs', args.split_group, args.region_set)
    nbrs_dir = os.path.join(args.dataset_dir, 'nbrs', args.split_group, args.region_set)
    create_dir(nbrs_dir)
    nbrs_file = os.path.join(nbrs_dir, str(args.batch_num_i) + '.txt')
    print 'Created needed directories ...'

    # Fetch files relevant to batch i
    partial_nbrs_files = [os.path.join(partial_nbrs_dir, f) for f in os.listdir(partial_nbrs_dir)
                          if f.startswith(str(args.batch_num_i) + '_')]
    handles = [open(filename) for filename in partial_nbrs_files]
    print 'Fetched', len(handles), 'files ...'

    num_regions_done = 0
    with open(nbrs_file, 'w') as output_file:
        while len(handles) > 0:
            partial_nbrs = list()
            handles_done = list()
            for (handle_num, handle) in enumerate(handles):
                try:
                    line = handle.next().strip()
                    partial_nbrs += ast.literal_eval(line)
                except StopIteration:
                    print 'Handle', handle_num, 'done'
                    handles_done.append(handle_num)
            leftover_handles = [h for (idx, h) in enumerate(handles) if idx not in handles_done]
            handles = leftover_handles
            partial_nbrs.sort(key=operator.itemgetter(1), reverse=True)
            nbrs = partial_nbrs[:args.num_nbrs]
            output_file.write(str(nbrs) + '\n')
            num_regions_done += 1
            if num_regions_done % 1 == 0:
                print num_regions_done, 'regions done'


# Aggregate farthest neighbours
def aggregate_farthest_nbrs(args):
    print 'Computing farthest nbrs'
    partial_nbrs_dir = os.path.join(args.dataset_dir, 'tmp_farthest_nbrs', args.split_group, args.region_set)
    farthest_nbrs_dir = os.path.join(args.dataset_dir, 'farthest_nbrs', args.split_group, args.region_set)
    create_dir(farthest_nbrs_dir)
    farthest_nbrs_file = os.path.join(farthest_nbrs_dir, str(args.batch_num_i) + '.csv')
    print 'Created needed directories ...'

    # Fetch files relevant to batch i
    partial_nbrs_files = [os.path.join(partial_nbrs_dir, f) for f in os.listdir(partial_nbrs_dir)
                          if f.startswith(str(args.batch_num_i) + '_')]
    handles = [open(filename) for filename in partial_nbrs_files]
    print 'Fetched', len(handles), 'files ...'

    num_regions_done = 0
    with open(farthest_nbrs_file, 'w') as output_file:
        while len(handles) > 0:
            partial_nbrs = list()
            handles_done = list()
            for (handle_num, handle) in enumerate(handles):
                try:
                    line = handle.next().strip()
                    partial_nbrs += ast.literal_eval(line)
                except StopIteration:
                    print 'Handle', handle_num, 'done'
                    handles_done.append(handle_num)
            leftover_handles = [h for (idx, h) in enumerate(handles) if idx not in handles_done]
            handles = leftover_handles
            partial_nbrs.sort(key=operator.itemgetter(1))
            farthest_nbr = partial_nbrs[0]
            output_file.write('"' + str(farthest_nbr) + '"\n')
            num_regions_done += 1
            if num_regions_done % 1 == 0:
                print num_regions_done, 'regions done'


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Path to dataset')
    arg_parser.add_argument('--split-group', type=str, required=True,
                            help='Which split of dataset - one of "orig", "predicate_novelty", "8_way"')
    arg_parser.add_argument('--region-set', type=str, required=True,
                            help='orig: train, test; '
                                 'predicate_novelty: policy_[train,val,test]_classifier_[train,val,test]; '
                                 '8_way: policy_[pretrain, train, val, test]_classifier_[train,test]')

    arg_parser.add_argument('--process-batch-pair', action="store_true", default=False,
                            help='Initial processing with a pair of batches i and j')
    arg_parser.add_argument('--aggregate-densities', action="store_true", default=False,
                            help='Compute densities for batch i')
    arg_parser.add_argument('--aggregate-nbrs', action="store_true", default=False,
                            help='Compute neighbours for batch i')
    arg_parser.add_argument('--aggregate-farthest-nbrs', action="store_true", default=False,
                            help='Compute farthest neighbours for batch i')

    arg_parser.add_argument('--batch-num-i', type=int, required=True,
                            help='For processing a pair of batches - i; Also use this to specify batch num ' +
                                 'for single batch operations')
    arg_parser.add_argument('--batch-num-j', type=int, default=None,
                            help='For processing a pair of batches - j; May be skipped if i and j are the same')
    arg_parser.add_argument('--batch-size', type=int, default=65536,
                            help='Regions batch size')
    arg_parser.add_argument('--sub-batch-size', type=int, default=8192,
                            help='For processing a pair of batches, number of rows of batch j to take')
    arg_parser.add_argument('--sub-batch-num', type=int, default=None,
                            help='For processing a pair of batches, start of sub-batch in batch j')
    arg_parser.add_argument('--num-nbrs', type=int, default=50,
                            help='Number of nearest neighbours to be found per region')

    args = arg_parser.parse_args()

    if args.process_batch_pair:
        process_batch_pair(args)

    if args.aggregate_densities:
        aggregate_batch_cosine_sims(args)

    if args.aggregate_nbrs:
        aggregate_nbrs(args)

    if args.aggregate_farthest_nbrs:
        aggregate_farthest_nbrs(args)

