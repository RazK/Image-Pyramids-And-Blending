#!/usr/bin/env python3
#############################################################
# FILE : sol3.py
# WRITER : Raz Karl , razkarl , 311143127
# EXERCISE : Image Processing Project 3 2017-2018
# DESCRIPTION:
#############################################################
import numpy as np
import sys, os
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

# RGB / RGBA Constants
RED = 0
GREEN = 1
BLUE = 2
ALPHA = 3
PIXEL_CHANNELS_RGB = [RED, GREEN, BLUE]
PIXEL_CHANNELS_RGBA = [RED, GREEN, BLUE, ALPHA]

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

# Images
TEST_IMAGE = "jerusalem.jpg"
TEST_IM1 = "external/black360.jpg"
TEST_IM2 = "external/white360.jpg"
TEST_MASK = "external/mask360.jpg"
IMAGE_MICHALI = "external/michali3.jpg"
IMAGE_ELLA = "external/ella2.jpg"
MASK_MICHELLA = "external/mask_michella4.jpg"
MASK_MICHELLA2 = "external/michella_mask2.jpg"
IMAGE_APPLE1 = "external/apple.jpg"
IMAGE_APPLE2 = "external/apple2.jpg"
MASK_APPLE = "external/mask_apple.jpg"
IMAGE_SPACE = "external/space2.jpg"
IMAGE_LANDSCAPE = "external/landscape.jpg"
MASK_LANDSPACE = "external/mask_landspace.jpg"


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


def blur(im, kernel, factor=1):
    """
    Performs image blurring using 2D convolution between the image f and a
    gaussian kernel g
    :param im: image to be blurred (grayscale float64 image).
    :param kernel_size: size of the gaussian kernel in each dimension (an
                        odd integer).
    :param factor: Factor by which to multiply the kernel.
    :return: blurry image (grayscale float64 image).
    """
    kernel2D = np.expand_dims(kernel, axis=0) * factor
    blurY = fconvolve(im, kernel2D.transpose())
    return fconvolve(blurY, kernel2D)


def reduce_image_spatial(im, kernel_size):
    """
    Reduce the given image by blurring and sub-smapling.
    Blurs the image with the given kernel, then sub-samples every 2nd pixel
    in every 2nd row.
    :param im: The image to reduce
    :param kernel_size: The kernel to blur with.
    :return: A reduced image of size im/4.
    """
    return blur_spatial(im, kernel_size)[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP]


def reduce_image(im, kernel):
    """
    Reduce the given image by blurring and sub-smapling.
    Blurs the image with the given kernel, then sub-samples every 2nd pixel
    in every 2nd row.
    :param im: The image to reduce
    :param kernel_size: The kernel to blur with.
    :return: A reduced image of size im/4.
    """
    return blur(im, kernel)[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP]


def expand_image_spatial(im, kernel_size, shape=None):
    """
    Expands the given image to the given shape by zero-padding and blurring.
    Pads every 2nd pixel with zero, then Blurs the image with the given
    kernel.
    :param im: The image to reduce
    :param kernel_size: The kernel to blur with.
    :param shape:   The shape to expand the image to.
                    Set to twice im.shape by default.
    :return: An expanded image of size im*4.
    """
    if shape == None:
        shape = tuple(dim * 2 for dim in im.shape)
    padded = np.zeros(shape)
    padded[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP] = im
    return blur_spatial(padded, kernel_size, 4)


def expand_image(im, kernel, shape=None):
    """
    Expands the given image to the given shape by zero-padding and
    blurring.
    Pads every 2nd pixel with zero, then Blurs the image with the given
    kernel.
    :param im: The image to reduce
    :param kernel_size: The kernel to blur with.
    :param shape:   The shape to expand the image to.
                    Set to twice im.shape by default.
    :return: An expanded image of size im*4.
    """
    if shape == None:
        shape = tuple(dim * 2 for dim in im.shape)
    padded = np.zeros(shape)
    padded[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP] = im
    return blur(padded, kernel, 2)


def smart_max_levels(im, max_levels):
    """
    Returns the real maximal number of levels without scaling below 16*16.
    :param max_levels: desired maximal levels
    :return: actual maximal levels.
    """
    return int(min(np.ceil(np.log2(min(im.shape)) - PYRAMID_SMALLEST_LEVEL),
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
        pyr[level] = reduce_image(pyr[level - 1], filter_vec)

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
        pyr[i] -= expand_image(pyr[i + 1], filter_vec, pyr[i].shape)
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff=None):
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
    # Default coeff = all ones
    if coeff == None:
        coeff = [1] * len(lpyr)
    clpyr = [lpyr[i] * coeff[i] for i in range(len(lpyr))]
    im = clpyr[-1]
    for level in range(len(clpyr) - 1, 0, -1):
        expanded = expand_image(im, filter_vec, clpyr[level - 1].shape)
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
    for level in range(min(len(pyr), levels)):
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


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
    Blends two images of the same size 2^(max_levels−1)
    :param im1: Input grayscale image to be blended with im2
    :param im2: Input grayscale image to be blended with im1
    :param mask:    A boolean (i.e. dtype == np.bool) mask containing True and
                    False representing which parts of im1 and im2 should
                    appear in the resulting im_blend.
                    Note that a value of True corresponds to 1, and False
                    corresponds to 0.
    :param max_levels:  The max_levels parameter used when generating the
                        Gaussian and Laplacian pyramids.
    :param filter_size_im:  Size of the Gaussian filter (an odd scalar that
                            represents a squared filter) which defining the
                            filter used in the construction of the Laplacian
                            pyramids of im1 and im2.
    :param filter_size_mask:    Size of the Gaussian filter(an odd scalar that
                                represents a squared filter) which defines
                                the filter used in the construction of the
                                Gaussian pyramid of mask.
    :return:
    """
    # Validate shapes for images and mask
    if not (im1.shape == im2.shape and im2.shape == mask.shape):
        raise AttributeError("Images and mask must have the same shape")

    # Construct Laplacian pyramids for images, Gaussian pyramid for mask
    L1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    Gm, filterm = build_gaussian_pyramid(mask.astype(np.float64),
                                         max_levels, filter_size_mask)
    # Construct blended Laplacian pyramid
    levels = min(len(L1), len(L2), len(Gm), max_levels)
    Lout = [np.copy(im1)] * levels
    for k in range(levels):
        Lout[k] = Gm[k] * L1[k] + (1 - Gm[k]) * L2[k]
    return laplacian_to_image(Lout, filter1)


def relpath(filename):
    """
    Relate the path given by filename to the current directory.
    For example, loading the file test.jpg in the subdirectory externals:
    im = read_image(relpath(’externals/test.jpg’), 1)
    :param filename: file path relative to current working directory
    :return: Absolute path to filename
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blendRGB(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    blend = np.zeros(im1.shape)
    for channel in PIXEL_CHANNELS_RGB:
        imc1, imc2 = im1[:, :, channel], im2[:, :, channel]
        blend[:, :, channel] = pyramid_blending(imc1, imc2, mask, max_levels,
                                                filter_size_im,
                                                filter_size_mask)
    return np.clip(blend[..., :], 0, 1)


def show_blending_example(im1, im2, mask, blend):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col',
                                               sharey='row')
    ax1.imshow(im1)
    ax2.imshow(im2)
    ax3.imshow(mask, plt.cm.gray)
    ax4.imshow(blend)
    plt.show()


def blending_example(im1_name, im2_name, mask_name):
    """
    Blends two images using a mask and returns them and their blend.
    :return: im1, im2, mask, im_blend
    """
    im1 = read_image(relpath(im1_name), MODE_RGB)
    im2 = read_image(relpath(im2_name), MODE_RGB)
    mask = read_image(relpath(mask_name), MODE_GRAY)
    blend = blendRGB(im1, im2, mask, 10, 3, 13)
    show_blending_example(im1, im2, mask, blend)
    return im1, im2, mask, blend


def blending_example0():
    """
    Blends two images using a mask and returns them and their blend.
    :return: im1, im2, mask, im_blend
    """
    return blending_example(TEST_IM1, TEST_IM2, TEST_MASK)


def blending_example1():
    """
    Blends two images using a mask and returns them and their blend.
    :return: im1, im2, mask, im_blend
    """
    return blending_example(IMAGE_MICHALI, IMAGE_ELLA, MASK_MICHELLA)


def blending_example11():
    """
    Blends two images using a mask and returns them and their blend.
    :return: im1, im2, mask, im_blend
    """
    return blending_example(IMAGE_ELLA, IMAGE_MICHALI, MASK_MICHELLA)


def blending_example2():
    """
    Blends two images using a mask and returns them and their blend.
    :return: im1, im2, mask, im_blend
    """
    return blending_example(IMAGE_LANDSCAPE, IMAGE_SPACE, MASK_LANDSPACE)


def blending_example3():
    """
    Blends two images using a mask and returns them and their blend.
    :return: im1, im2, mask, im_blend
    """
    return blending_example(IMAGE_APPLE1, IMAGE_APPLE2, MASK_APPLE)
