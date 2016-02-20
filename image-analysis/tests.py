import numpy as np
import matplotlib.pylab as plt
from sero import *
from random import randint
from serodraw import *

test_xdim = 50
test_ydim = 50

def create_test_images():
    image0 = np.zeros([test_xdim,test_ydim]) # Evenly spaced squares
    for i in range(0, test_xdim - 10, 10):
        for j in range(0,test_ydim - 10, 10):
            for x in range(0,5):
                for y in range(0,5):
                    image0[i+x][j+y] = Config.hard_max_pixel_value

    image1 = np.zeros([test_xdim,test_ydim]) # Evenly spaced squares with varying intensity per pixel
    for i in range(0,  test_xdim - 10, 10):
        for j in range(0,test_ydim - 10, 10):
            for x in range(0,5):
                for y in range(0,5):
                    image1[i+x][j+y] = 245 + randint(0, 10)

    image2 = np.zeros([test_xdim,test_ydim]) # Random static
    for i in range(0,test_xdim):
        for j in range(0, test_ydim):
            image2[i][j] = Config.min_val_threshold - 1 + randint(0,2)

    image3 = np.zeros([test_xdim,test_ydim])# Lines
    for i in range(0, test_xdim, 5):
        for j in range(0, test_ydim):
            image3[i][j] = Config.hard_max_pixel_value
            image3[j][i] = 245 + randint(0, 10)




    images = [image0, image1, image2, image3]
    slides = []
    # for index,im in enumerate(images):
    #     slides.append(Slide(matrix=im, height=index))
    s0 = Slide(matrix=image0, height=0)

    s1 = Slide(matrix=image1, height=1)

    # s2 = Slide(matrix=image2, height=2)

    s3 = Slide(matrix=image3, height=3)

    # plt.imshow(image2, cmap='rainbow', interpolation='none')
    # plt.imshow(image2, interpolation='none')
    # plt.colorbar()
    # plt.show()

    # showSlide(s1)

    plotBlob2ds(list(b2d for b2d in Blob2d.all.values()), edge=False)#, images_and_heights=[(image0, 0), (image1, 1)])
    # plot_plotly(list(Blob2d.all.values()), b2ds=True)

    # plt.imshow(image1, cmap='rainbow', interpolation='none')
    # plt.imshow(image1, interpolation='none')
    # plt.colorbar()
    # plt.show()

def single_test():
    identity = np.zeros([25,25])
    identity = np.asarray(
        [
        [1,1,1,0,0],
        [0,0,1,1,0],
        [0,1,0,1,1],
        [0,1,1,1,1],
        [1,1,1,1,1]]
    )
    identity *= 255
    identity = np.transpose(identity)
    # for i in range(identity.shape[0]):
    #     for j in range(identity.shape[1]):
    #         if abs(i - j) <= 1:
    #             identity[i][j] = Config.hard_max_pixel_value

    s0 = Slide(matrix=identity, height=0)
    plotBlob2ds(list(b2d for b2d in Blob2d.all.values()), edge=False)



def showSlide(slide):
    import matplotlib.pylab as plt
    print('Showing Slide:')
    if len(slide.alive_pixels) > 0:
        maxx = max(Blob2d.get(b2d).maxx for b2d in slide.blob2dlist)
        maxy = max(Blob2d.get(b2d).maxy for b2d in slide.blob2dlist)
        minx = min(Blob2d.get(b2d).minx for b2d in slide.blob2dlist)
        miny = min(Blob2d.get(b2d).miny for b2d in slide.blob2dlist)
        array = np.zeros([maxx - minx + 1, maxy - miny + 1])
        for pixel in slide.alive_pixels:
            array[pixel.x - minx][pixel.y - miny] = pixel.val
        plt.imshow(array, cmap='rainbow', interpolation='none')
        # plt.matshow(array)
        plt.show()
    else:
        print('Cannot show slide with no pixels:' + str(slide))





filter_available_colors()

create_test_images()
# single_test()