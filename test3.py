from sol3 import *


def testImshow(image):
    plt.imshow(image, plt.cm.gray)
    plt.show()


def testKernel(kernel):
    print(kernel)


def testBuildGaussianPyramid(image):
    gpyr, vec = build_gaussian_pyramid(image, 30, 9)
    for im in gpyr:
        plt.imshow(im, plt.cm.gray)
        plt.show()

def testBuildGaussianPyramid4Mask(mask):
    mask = read_image(mask, MODE_GRAY)
    for i in [3,5,7,9,11]:
        gpyr, fltr = build_gaussian_pyramid(mask.astype(np.float64),
                           10, i)
        display_pyramid(gpyr, 10)

def testExpandImageSpatial(image):
    plt.imshow(expand_image_spatial(image, 10), plt.cm.gray)
    plt.show()


def testBuildLaplacianPyramid(image):
    lpyr, vec = build_laplacian_pyramid(image, 30, 9)
    for im in lpyr:
        plt.imshow(im, plt.cm.gray)
        plt.show()


def testLaplacianToImage(image):
    lpyr, vec = build_laplacian_pyramid(image, 30, 4)
    im2 = laplacian_to_image(lpyr, vec, [i%3 for i in range(len(
        lpyr))])
    plt.imshow(im2, plt.cm.gray)
    plt.show()


def testRenderGaussianPyramid(image):
    gpyr, vec = build_gaussian_pyramid(image, 30, 9)
    plt.imshow(gpyr[0], plt.cm.gray)
    plt.show()
    render = render_pyramid(gpyr, 3)
    plt.imshow(render, plt.cm.gray)
    plt.show()


def testRenderLaplacianPyramid(image):
    lpyr, vec = build_laplacian_pyramid(image, 4, 2)
    plt.imshow(lpyr[0], plt.cm.gray)
    plt.show()
    render = render_pyramid(lpyr, 10)
    plt.imshow(render, plt.cm.gray)
    plt.show()


def testRelPath():
    print(relpath(TEST_IMAGE))


def testBlending():
    im1 = read_image(IMAGE_ELLA, MODE_RGB)
    im2 = read_image(IMAGE_MICHALI, MODE_RGB)
    mask = read_image(MASK_MICHELLA, MODE_GRAY)
    for i in [1,2,3,4,5,6,7,8]:
        blend = blendRGB(im1, im2, mask, i, 5, 5)
        show_blending_example(im1, im2, mask, blend)


def testBlendingExample1():
    im1, im2, mask, blend = blending_example1()


def testBlendingExample2():
    im1, im2, mask, blend = blending_example2()


def testBlendingExample11():
    im1, im2, mask, blend = blending_example11()


def testBlendingExample3():
    im1, im2, mask, blend = blending_example3()


def main():
    """
    Tests for sol3.py
    """
    image = read_image(IMAGE_LANDSCAPE, MODE_GRAY)

    # testImshow(image)
    # testKernel(kernel)
    # testBuildGaussianPyramid(image)
    #testBuildGaussianPyramid4Mask(MASK_APPLE)
    # testExpandImageSpatial(image)
    # testBuildLaplacianPyramid(image)
    # testLaplacianToImage(image)
    # testRenderGaussianPyramid(image)
    # testRenderLaplacianPyramid(image)
    # testRelPath()
    #testBlending()
    # testBlendingExample1()
    #testBlendingExample3()
    testBlendingExample2()
    #testBlendingExample11()


if (__name__ == "__main__"):
    main()
