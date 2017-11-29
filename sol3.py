#!/usr/bin/env python3
#############################################################
# FILE : sol3.py
# WRITER : Raz Karl , razkarl , 311143127
# EXERCISE : Image Processing Project 3 2017-2018
# DESCRIPTION:
#############################################################
import numpy as np
import sys
from matplotlib.image import imread
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

# Constants
PIXEL_INTENSITY_MAX = 255
PIXEL_INTENSITIES = PIXEL_INTENSITY_MAX + 1
PIXEL_RANGE = (0, PIXEL_INTENSITIES)
PIXEL_RANGE_NORMALIZED = (0, 1)
PIXEL_CHANNELS_RGB = 3
PIXEL_CHANNELS_RGBA = 4

# Picture representation modes
MODE_GRAYSCALE = 1
MODE_RGB = 2

# Default test image


# Helper methods
def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation.
    :param filename:        string containing the image filename to read.
    :param representation:  representation code, either 1 or 2 defining whether
                            the output should be a greyscale image (1) or an
                            RGB image (2).
    :return:                returns an image represented by a matrix of type
                            .float64 with intensities normalized to the
                            range [0,1]
    """
    im = imread(filename)
    im_float = im.astype(np.float64)
    if (representation == MODE_GRAYSCALE):
        im_float = rgb2gray(im_float)
    return im_float / PIXEL_INTENSITY_MAX

def main():
    """
    Tests for sol2.py
    """
    TEST_IMAGE = "jerusalem.jpg"
    image = read_image(TEST_IMAGE, MODE_GRAYSCALE)
    plt.imshow(image, plt.cm.gray)
    plt.show()


if (__name__ == "__main__"):
    main()
