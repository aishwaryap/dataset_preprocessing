import h5py
from argparse import ArgumentParser
import numpy as np


def main(output_file):
    with h5py.File(output_file, 'w') as handle:
        for image_id in range(99610):
            image_dataset = handle.create_dataset(str(image_id),
                                                  shape=(10, 40, 1024),
                                                  dtype='f')
            values = np.random.random((10, 40, 1024))
            image_dataset[:] = values
            if image_id % 100 == 0:
                print('Completed', image_id, 'images')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--output-file', type=str, required=True,
                            help='HDF5 file to store output')
    args = arg_parser.parse_args()
    main(args.output_file)
