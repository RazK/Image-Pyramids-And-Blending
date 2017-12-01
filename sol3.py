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
from scipy.ndimage.filters import convolve as fconvolve
from scipy.signal import convolve as sconvolve, convolve2d

# Constants
PIXEL_INTENSITY_MAX = 255
PIXEL_INTENSITIES = PIXEL_INTENSITY_MAX + 1
PIXEL_RANGE = (0, PIXEL_INTENSITIES)
PIXEL_RANGE_NORMALIZED = (0, 1)
PIXEL_CHANNELS_RGB = 3
PIXEL_CHANNELS_RGBA = 4

# Picture representation modes
MODE_GRAY = 1
MODE_RGB = 2

# Kernels
KERNEL_DX = np.array([[1, 0, -1]])
KERNEL_DY = KERNEL_DX.transpose()
KERNEL_GAUSS_BASE = np.array([1, 1])

# Pyramids
PYRAMID_SMALLEST_LEVEL = 4
SUBSAMPLE_STEP = 2

# Default test image
TEST_IMAGE = "jerusalem.jpg"


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
    if (representation == MODE_GRAY):
        im_float = rgb2gray(im_float)
    return im_float / PIXEL_INTENSITY_MAX


def gaussian_kernel(kernel_size):
    """
    Returns a 1-D Gaussian kernel of the given size, normalized s.t. sum=1
    Examples:
        gaussian_kernel(1) = [1]
        gaussian_kernel(2) = [ 0.5  0.5]
        gaussian_kernel(3) = [ 0.25  0.5   0.25]
        gaussian_kernel(4) = [ 0.125  0.375  0.375  0.125]
    :param kernel_size: the size of the gaussian kernel (an odd integer).
    :return: a 1-D gaussian kernel of the given size.
    """
    kernel = np.array([1])
    for i in range(0, kernel_size - 1):
        kernel = sconvolve(kernel, KERNEL_GAUSS_BASE) / 2
    return kernel


def gaussian_kernel_2D(kernel_size):
    """
    Returns a 2-D Gaussian kernel of the given size, normalized s.t. sum=1
    Examples:
        gaussian_kernel(1) = [[1]]
        gaussian_kernel(2) = [[ 0.25  0.25]
                              [ 0.25  0.25]]
        gaussian_kernel(3) = [[ 0.0625  0.125   0.0625]
                              [ 0.125   0.25    0.125 ]
                              [ 0.0625  0.125   0.0625]]
    :param kernel_size: the size of the gaussian kernel (an odd integer).
    :return: a 2-D gaussian kernel of the given size.
    """
    kernel = gaussian_kernel(kernel_size)
    kernel_x, kernel_y = np.meshgrid(kernel, kernel)
    return kernel_x * kernel_y


def blur_spatial(im, kernel_size, factor=1):
    """
    Performs image blurring using 2D convolution between the image f and a
    gaussian kernel g
    :param im: image to be blurred (grayscale float64 image).
    :param kernel_size: size of the gaussian kernel in each dimension (an
                        odd integer).
    :param factor: Factor by which to multiply the kernel.
    :return: blurry image (grayscale float64 image).
    """
    kernel2D = gaussian_kernel_2D(kernel_size) * factor
    return convolve2d(im, kernel2D, mode='same')


def reduce_image(im, kernel_size):
    """
    Reduce the given image by blurring and sub-smapling.
    Blurs the image with the given kernel, then sub-samples every 2nd pixel
    in every 2nd row.
    :param im: The image to reduce
    :param kernel_size: The kernel to blur with.
    :return: A reduced image of size im/4.
    """
    return blur_spatial(im, kernel_size)[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP]


def expand_image(im, shape, kernel_size):
    """
    Expands the given image to the given shape by zero-padding and blurring.
    Pads every 2nd pixel with zero, then Blurs the image with the given
    kernel.
    :param im: The image to reduce
    :param kernel_size: The kernel to blur with.
    :return: An expanded image of size im*4.
    """
    padded = np.zeros(shape)
    padded[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP] = im
    return blur_spatial(padded, kernel_size, 4)


def smart_max_levels(im, max_levels):
    """
    Returns the real maximal number of levels without scaling below 16*16.
    :param max_levels: desired maximal levels
    :return: actual maximal levels.
    """
    return int(min(np.log2(min(im.shape)) - PYRAMID_SMALLEST_LEVEL,
                   max_levels))


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid of the given image.
    :param im:  A grayscale image with double values in [0, 1] (e.g. the
                output of ex1’s read_image with the representation set to 1).
    :param max_levels: The maximal number of levels in the resulting pyramid.
    :param filter_size: The size of the Gaussian filter (an odd scalar that
                        represents a squared filter) to be used in
                        constructing the pyramid filter (e.g for
                        filter_size = 3 we get [0.25, 0.5, 0.25])
    :return: (pyr, filter_vec)
    :param pyr: A standard python array of images. Its length is max_levels.
                The pyramid levels are arranged in order of descending
                resolution s.t. pyr[0] has the resolution of the given input
                image im.
    :param filter_vec:  1D-row of size filter_size used for the pyramid
                        construction. This filter is built using a
                        consequent 1D convolutions of [1 1] with itself in
                        order to derive a row of the binomial coefficients
                        which is a good approximation to the Gaussian
                        profile. The filter_vec is normalized.
    """
    filter_vec = gaussian_kernel(filter_size)
    levels = smart_max_levels(im, max_levels)
    pyr = [np.copy(im)] * levels
    for level in range(1, levels):
        pyr[level] = reduce_image(pyr[level - 1], filter_size)

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid of the given image.
    :param im:  A grayscale image with double values in [0, 1] (e.g. the
                output of ex1’s read_image with the representation set to 1).
    :param max_levels: The maximal number of levels in the resulting pyramid.
    :param filter_size: The size of the Gaussian filter (an odd scalar that
                        represents a squared filter) to be used in
                        constructing the pyramid filter (e.g for
                        filter_size = 3 you should get [0.25, 0.5, 0.25])
    :return: (pyr, filter_vec)
    :param pyr: A standard python array of images. Its length is max_levels.
                The pyramid levels are arranged in order of descending
                resolution s.t. pyr[0] has the resolution of the given input
                image im.
    :param filter_vec:  1D-row of size filter_size used for the pyramid
                        construction. This filter is built using a
                        consequent 1D convolutions of [1 1] with itself in
                        order to derive a row of the binomial coefficients
                        which is a good approximation to the Gaussian
                        profile. The filter_vec is normalized.
    """
    pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(pyr) - 1):
        pyr[i] -= expand_image(pyr[i + 1], pyr[i].shape, filter_size)
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Reconstructs an image from its Laplacian Pyramid.
    :param lpyr:    A standard python array of images. Its length is
                    max_levels. The pyramid levels are arranged in order of
                    descending resolution s.t. pyr[0] has the resolution of
                    the given input image im.
    :param filter_vec:  1D-row of size filter_size used for the pyramid
                        construction. This filter is built using a
                        consequent 1D convolutions of [1 1] with itself in
                        order to derive a row of the binomial coefficients
                        which is a good approximation to the Gaussian
                        profile. The filter_vec is normalized.
    :param coeff:   A vector. The vector size is the same as the number of
                    levels in the pyramid lpyr.
                    Before reconstructing the image img, each level i of the
                    laplacian pyramid is multiplied by its corresponding
                    coefficient coeff[i].
                    Only when this vector is all ones we get the original
                    image (up to a negligible floating error, e.g. maximal
                    absolute difference around 10−12).
                    When some values are different than 1 we will get
                    filtering effects.
    :return: The image reconstructed from the Laplacian Pyramid.
    """
    clpyr = [lpyr[i]*coeff[i] for i in range(len(lpyr))]
    im = clpyr[-1]
    for level in range(len(clpyr) - 1, 0, -1):
        expanded = expand_image(im, clpyr[level - 1].shape, len(filter_vec))
        im = expanded + clpyr[level - 1]
    return im


def render_pyramid(pyr, levels):
    """
    Returns a rendered image visualization of the given pyramid.
    :param pyr: Either a Gaussian or a Laplacian pyramid.
                A standard python array of images. Its length is max_levels.
                The pyramid levels are arranged in order of descending
                resolution s.t. pyr[0] has the resolution of the given input
                image im.
    :param levels: The number of levels to present in the result ≤ max_levels.
    :return:    A single black image in which the pyramid levels of the given
                pyramid pyr are stacked horizontally (after stretching the
                values to [0, 1]).
    """
    rows = pyr[0].shape[0]
    cols = sum([level.shape[1] for level in pyr[:levels]])
    render = np.zeros((rows, cols))
    left = 0
    for level in range(min(len(pyr),levels)):
        bottom = pyr[level].shape[0]
        right = left + pyr[level].shape[1]
        render[0:bottom, left:right] = stretch(pyr[level])
        left = right

    return render


def stretch(im):
    """
    Returns the given image stretched to the range [0,1]
    :param im: image to stretch
    :return: image after stretching to [0,1]
    """
    minv, maxv = im.min(), im.max()
    return (im - minv) / (maxv - minv)


def display_pyramid(pyr, levels):
    """
    Displays the given pyramid.
    :param pyr: Either a Gaussian or a Laplacian pyramid.
                A standard python array of images. Its length is max_levels.
                The pyramid levels are arranged in order of descending
                resolution s.t. pyr[0] has the resolution of the given input
                image im.
    :param levels: The number of levels to present in the result ≤ max_levels.
    """
    rendered_pyr = render_pyramid(pyr, levels)
    plt.imshow(rendered_pyr)
    plt.show()


def main():
    """
    Tests for sol2.py
    """
    image = read_image(TEST_IMAGE, MODE_GRAY)
    kernel = gaussian_kernel(5)
    if False:
        plt.imshow(image, plt.cm.gray)
        plt.show()
    if False:
        print(kernel)
        gpyr, vec = build_gaussian_pyramid(image, 30, 5)
        for im in gpyr:
            plt.imshow(im, plt.cm.gray)
            plt.show()
    if False:
        plt.imshow(expand_image(image, 10), plt.cm.gray)
        plt.show()
    if False:
        lpyr, vec = build_laplacian_pyramid(image, 30, 5)
        for im in lpyr:
            plt.imshow(im, plt.cm.gray)
            plt.show()
    if True:
        lpyr, vec = build_laplacian_pyramid(image, 30, 5)
        im2 = laplacian_to_image(lpyr, vec, [1-(0.5 ** i) for i in range(len(
            lpyr))])
        plt.imshow(im2, plt.cm.gray)
        plt.show()
    if False:
        gpyr, vec = build_gaussian_pyramid(image, 30, 5)
        plt.imshow(gpyr[0], plt.cm.gray)
        plt.show()
        render = render_pyramid(gpyr, 3)
        plt.imshow(render, plt.cm.gray)
        plt.show()
    if False:
        lpyr, vec = build_laplacian_pyramid(image, 4, 2)
        plt.imshow(lpyr[0], plt.cm.gray)
        plt.show()
        render = render_pyramid(lpyr, 10)
        plt.imshow(render, plt.cm.gray)
        plt.show()


if (__name__ == "__main__"):
    main()
