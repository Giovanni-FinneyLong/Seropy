__author__ = 'gio'
''' This script is used to tested to efficiency (specifically time but also memory), of different execution methods'''

import timeit
import pdb

from sero import debug, runShell
import numpy as np


def timeListVsArray():
    setup = '''
import glob
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
from numpy import zeros
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import numpy as np
from sero import Pixel
from sero import KMeansClusterIntoLists

all_images = glob.glob('..\\data\\Swell*.tif')
all_images = [all_images[0]] # HACK

for imagefile in all_images:
    imagein = Image.open(imagefile)
    imarray = np.array(imagein)
    slices = []
    (xdim, ydim, zdim) = imarray.shape
    slices = []
    pixels = []
    sorted_pixels = []
    image_channels = imagein.split()

    for s in range(len(image_channels)):  # Better to split image and use splits for arrays than to split an array
        buf = np.array(image_channels[s])
        slices.append(buf)

    (xdim, ydim) = slices[0].shape
    for pcol in range(xdim):
        for pix in range(ydim):
            pixel_value = slices[0][pcol][pix]
            # print('Pixel #' + str(pcol * xdim + pix) + ' = ' + str(pixel_value))
            if (pixel_value != 0):  # Can use alternate min threshold and <=
                pixels.append((pixel_value, pcol, pix))
    # Now to sort by 3rd element/2nd index = pixel value
    sorted_pixels = sorted(pixels, key=lambda tuplex: tuplex[0], reverse=True)

    endmax = 0
    while (sorted_pixels[endmax][0] >= 250):
        endmax += 1

    max_tuple_list = sorted_pixels[0:endmax]  # Pixels with value 255
    max_pixel_list = [Pixel(float(i[0]), float(i[1]), float(i[2])) for i in max_tuple_list]
    max_tuples_as_arrays = np.asarray([(float(i[0]), float(i[1]), float(i[2])) for i in max_tuple_list])
    max_float_array = zeros([xdim, ydim])
    max_tuple_array = np.empty([xdim, ydim], dtype=object)
    for pixel in max_tuple_list:
        max_float_array[pixel[1]][pixel[2]] = pixel[0]  # Note Remember that these are pointers!
        max_tuple_array[pixel[1]][pixel[2]] = pixel

    # Now have labels in centLabels for each of the max_pixels
    cluster_count = 20
    cluster_lists = KMeansClusterIntoLists(max_tuples_as_arrays, cluster_count)

try: # Declare the counter IF they don't yet exist
    lcount
    acount
except NameError:
    lcount = 0
    acount = 0

'''


    listApproach = '''
print('List iter:' + str(lcount))
cluster_arrays = []  # Each entry is an array, filled only with the maximal values from the corresponding
for cluster in range(cluster_count):
    cluster_arrays.append(zeros([xdim, ydim]))  # (r,c)
    for pixel in cluster_lists[cluster]:
        cluster_arrays[cluster][pixel[1]][pixel[2]] = int(pixel[0])
lcount += 1
'''

    arrayApproach = ''' # Use slicing instead of list indices
print('Array iter:' + str(acount)))
clusters = zeros([1600, 1600, cluster_count])
for cluster range(cluster_count):
    for pixel in cluster_lists[cluster]:
        clusters[pixel[1]][pixel[2]][cluster] = int(pixel(0))
acount += 1
'''
# Repeat (x,y) means x full cycles (including setup), with y runs per cycle
    ncycles = 1
    nruns = 10
    la = [] # List-Approach
    aa = [] # Array-Approach
    la.append(timeit.Timer(listApproach, setup=setup).repeat(ncycles, nruns))
    aa.append(timeit.Timer(listApproach, setup=setup).repeat(ncycles, nruns))

    la_avg = np.mean(la)
    aa_avg = np.mean(aa)
    print('Avg time for la: ' + str(la_avg))
    print('Avg time for aa: ' + str(aa_avg))
    #debug()
    runShell()

