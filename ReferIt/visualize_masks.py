import os, re
import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave, imread
from PIL import Image


def get_numpy_array(image):
    image_data = np.array(image.getdata())
    return image_data.reshape(image.size[0], image.size[1], image_data.shape[1])


def visualize(image_file, mask_file, output_file):
    mask = loadmat(mask_file)['segimg_t']
    print('mask.shape =', mask.shape)
    reversed_mask = (mask == 0).astype(np.uint8)
    print('reversed_mask.shape =', reversed_mask.shape)
    np_image = imread(image_file)
    for channel in range(np_image.shape[2]):
        np_image[:, :, channel] = np.multiply(np_image[:, :, channel], reversed_mask)
    imsave(output_file, np_image)


if __name__ == '__main__':
    path = '/u/aish/Documents/temp/'
    image_file = os.path.join(path, '10000.jpg')
    mask_files = [os.path.join(path, x) for x in
                  ['10000_1.mat', '10000_2.mat', '10000_3.mat', '10000_4.mat', '10000_5.mat']]
    for mask_file in mask_files:
        output_file = re.sub('.mat', '.jpg', mask_file)
        visualize(image_file, mask_file, output_file)