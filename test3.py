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


def testExpandImageSpatial(image):
    plt.imshow(expand_image_spatial(image, 10), plt.cm.gray)
    plt.show()


def testBuildLaplacianPyramid(image):
    lpyr, vec = build_laplacian_pyramid(image, 30, 9)
    for im in lpyr:
        plt.imshow(im, plt.cm.gray)
        plt.show()

def testLaplacianToImage(image):
    lpyr, vec = build_laplacian_pyramid(image, 30, 9)
    im2 = laplacian_to_image(lpyr, vec)
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
    im1 = read_image(IMAGE_ELLA, MODE_GRAY)
    im2 = read_image(IMAGE_MICHALI, MODE_GRAY)
    mask = read_image(MASK_MICHELLA, MODE_GRAY)
    blend = pyramid_blending(im1, im2, mask, 10, 7, 7)
    plt.imshow(blend, plt.cm.gray)
    plt.show()

def testBlendingExample1():
    im1, im2, mask, blend = blending_example1()
    plt.imshow(blend)
    plt.show()


def testBlendingExample2():
    im1, im2, mask, blend = blending_example2()
    plt.imshow(blend)
    plt.show()

def testBlendingExample11():
    im1, im2, mask, blend = blending_example11()
    plt.imshow(blend)
    plt.show()

def main():
    """
    Tests for sol3.py
    """
    image = read_image(IMAGE_ELLA, MODE_GRAY)
    kernel = gaussian_kernel(5)

    #testImshow(image)
    #testKernel(kernel)
    #testBuildGaussianPyramid(image)
    #testExpandImageSpatial(image)
    #testBuildLaplacianPyramid(image)
    #testLaplacianToImage(image)
    #testRenderGaussianPyramid(image)
    #testRenderLaplacianPyramid(image)
    #testRelPath()
    #testBlending()
    #testBlendingExample1()
    #testBlendingExample11()
    testBlendingExample2()

if (__name__ == "__main__"):
    main()
