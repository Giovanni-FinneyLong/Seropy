__author__ = 'gio'
#import matplotlib.pyplot as plt
from PIL import Image

#import matplotlib

import readline
import code
import rlcompleter
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
from numpy import zeros
import math
import sys

import time
import glob




from serodraw import *
#import serodraw
#__import__('serodraw')
import pdb


from mpl_toolkits.mplot3d import Axes3D
import pickle # Note uses cPickle automatically ONLY IF python 3
from sklearn.preprocessing import normalize
from PIL import ImageFilter
from collections import OrderedDict

def debug():
    pdb.set_trace()

def runShell():
    gvars = globals()
    gvars.update(locals())
    readline.set_completer(rlcompleter.Completer(gvars).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(gvars)
    shell.interact()

def KMeansClusterIntoLists(array_in, num_clusters):
    'Take an array of floats and returns a list of lists, each of which contains all the pixels of a cluster'
    cluster_lists = [[] for i in range(num_clusters)]
    # for c in range(num_clusters):
    #     cluster_arrays.append(zeros([xdim, ydim])) # (r,c)
    (bookC, distortionC)  = kmeans(array_in, num_clusters)
    (centLabels, centroids) = vq(array_in, bookC)
    for pixlabel in range(len(centLabels)):
        cluster = centLabels[pixlabel]
        # pix = max_pixel_array_floats[pixlabel]
        # cluster_arrays[cluster][pix[1]][pix[2]] = pix[0]
        cluster_lists[cluster].append(array_in[pixlabel])
    return cluster_lists


class Pixel:
    '''This class is being used to hold the coordinates, base info and derived info of a pixle of a single image\'s layer'''
    def __init__(self, value, xin, yin):
        self.x = xin # The x coordinate, int
        self.y = yin # The y coordinate, int
        self.val = value # float
        self.nz_neighbors = 0
        self.maximal_neighbors = 0
        self.neighbor_sum = 0
        self.neighbors_checked = 0
        self.neighbors_set = False # For keeping track later, incase things get nasty

    def setNeighborValues(self, non_zero_neighbors, max_neighbors, neighbor_sum, neighbors_checked):
        self.nz_neighbors = non_zero_neighbors # The number out of the 8 surrounding pixels that are non-zero
        self.maximal_neighbors = max_neighbors
        self.neighbor_sum = neighbor_sum # The sum of the surrounding 8 pixels
        self.neighbors_checked = neighbors_checked
        self.neighbors_set = True
    def toTuple(self):
        return (self.val, self.x, self.y)

    def __str__(self):
        '''Method used to convert Pixel to string, generall for printing'''
        return str('P{[v:' + str(self.val) + ', x:' + str(self.x) + ', y:' + str(self.y) + '], [nzn:' + str(self.nz_neighbors) + ', mn:' + str(self.maximal_neighbors) + ', ns:' + str(self.neighbor_sum ) + ', nc:' + str(self.neighbors_checked) + ']}')
    __repr__ = __str__



def main():
    all_images = glob.glob('..\\data\\Swell*.tif')
    print(all_images)

    all_images = [all_images[0]] # HACK
    print(all_images)

    for imagefile in all_images:
        imagein = Image.open(imagefile)
        print('Starting on image: ' + imagefile)
        imarray = np.array(imagein)
        slices = []
        (xdim, ydim, zdim) = imarray.shape
        # np.set_printoptions(threshold=np.inf)
        print('The are ' + str(zdim) + ' channels')
        image_channels = imagein.split()
        slices = []
        non_zero_count = 0
        pixels = []
        sum_pixels = 0
        sorted_pixels = []

        for s in range(len(image_channels)): # Better to split image and use splits for arrays than to split an array
            buf = np.array(image_channels[s])
            slices.append(buf)
            if (np.amax(slices[s]) == 0):
                print('Slice #' + str(s) + ' is an empty slice')
        # im = Image.fromarray(np.uint8(cm.jet(slices[0])*255))
        # out = im.filter(ImageFilter.MaxFilter)
        # Can use im.load as needed to access Image pixels
        # Opting to use numpy, I expect this will be faster to operate and easier to manipulate

        (xdim, ydim) = slices[0].shape
        pixels_for_hist = []
        for pcol in range(xdim):
            for pix in range(ydim):
                pixel_value = slices[0][pcol][pix]
                # print('Pixel #' + str(pcol * xdim + pix) + ' = ' + str(pixel_value))
                if(pixel_value != 0): # Can use alternate min threshold and <=
                    pixels.append((pixel_value, pcol, pix))
                    pixels_for_hist.append(pixel_value) #Hack due to current lack of a good way to slice tuple list quickly
                    sum_pixels += pixel_value
        print('The are ' + str(len(pixels)) + ' non-zero pixels from the original ' + str(xdim * ydim) + ' pixels')
        # Now to sort by 3rd element/2nd index = pixel value
        sorted_pixels = sorted(pixels, key=lambda tuplex: tuplex[0], reverse=True)
        # Lets go further and grab the maximal pixels, which are at the front
        endmax = 0
        while(sorted_pixels[endmax][0] >=  250):
            endmax += 1
        print('There are  ' + str(endmax) + ' maximal pixels')
        # Time to pre-process the maximal pixels; try and create groups/clusters

        max_tuple_list = sorted_pixels[0:endmax] # Pixels with value 255
        max_pixel_list = [Pixel(float(i[0]), float(i[1]), float(i[2])) for i in max_tuple_list]


        max_tuples_as_arrays = np.asarray([(float(i[0]), float(i[1]), float(i[2])) for i in max_tuple_list])
            # NOTE: Is an array of shape (#pixels, 3), where each element is an array representing a tuple.
            # NOTE: This is the required format for kmeans/vq



        #FIXME, creating flat array (14k,)
        max_float_array = zeros([xdim, ydim])
        max_tuple_array = np.empty([xdim,ydim], dtype=object)
        for pixel in max_tuple_list:
            max_float_array[pixel[1]][pixel[2]] = pixel[0] # Note Remember that these are pointers!
            max_tuple_array[pixel[1]][pixel[2]] = pixel
        #asarray([i.val for i in max_pixel_list])


        # Now have labels in centLabels for each of the max_pixels
        # For fun, going to make an array for each cluster


        cluster_count = 20
        cluster_lists = KMeansClusterIntoLists(max_tuples_as_arrays, cluster_count) #max_tuples_as_arrays is replacing max_float_array which was not workking, TODO figure out why
        for i in range(len(cluster_lists)):
            print('Index:' + str(i) + ', size:' + str(len(cluster_lists[i]))) # + ' pixels:' + str(cluster_lists[i]))

        # findBestClusterCount(0, 100, 5)
        # MeanShiftCluster(max_float_array)
        # AffinityPropagationCluster(max_float_array):




        cluster_arrays = [] # Each entry is an array, filled only with the maximal values from the corresponding
        for cluster in range(cluster_count):
            cluster_arrays.append(zeros([xdim, ydim])) # (r,c)
            for pixel in cluster_lists[cluster]:
                cluster_arrays[cluster][pixel[1]][pixel[2]] = int(pixel[0])

        # PlotListofClusterArraysColor(cluster_arrays, 1)





        # TODO may want to use numpy 3d array over a list of 2d arrays; remains to be checked for speed/memory
        # Note, am writing the pixel filters below, so that images can be compared to their unfilterd counterparts (which are already gen)..

        dead_pixels = [] # Still in other list
        alive_pixels = [] # Could do set difference later, but this should be faster..

        for (pixn, pixel) in enumerate(max_pixel_list): #pixel_number and the actual pixel (value, x-coordinate, y-coordinate)
            col = pixel.y # HACK
            row = pixel.x # TODO FIXME! - The naming scheme is incorrect; refactor
            # Keep track of nz-neighbors, maximal-neighbors, neighbor sum
            buf_nzn = 0
            buf_maxn = 0
            buf_sumn = 0.
            neighbors_checked = 0
            for left_shift in range(-1, 2, 1): # NOTE CURRENTLY 1x1
                for up_shift in range(-1, 2, 1): # NOTE CURRENTLY 1x1
                    if(left_shift != 0 or up_shift != 0): # Don't measure the current pixel
                        if(row + left_shift < xdim and row + left_shift >= 0 and col + up_shift < ydim and col + up_shift >= 0): # Boundary check.
                            neighbors_checked += 1
                            cur_neighbor_val = max_float_array[row + left_shift][col + up_shift]
                            if(cur_neighbor_val > 0):
                                buf_nzn += 1
                                if(cur_neighbor_val == 255):
                                    buf_maxn += 1
                                buf_sumn += cur_neighbor_val
            #if buf_nzn != 0:
                #print('Setting pixel vals to: nzn:' + str(buf_nzn) + ', maxn:' + str(buf_maxn) + ', sumn:' + str(buf_sumn))
            pixel.setNeighborValues(buf_nzn, buf_maxn, buf_sumn, neighbors_checked)
            if buf_nzn == 0 :
                dead_pixels.append(pixel)
            else:
                alive_pixels.append(pixel)
        print('There are ' + str(len(dead_pixels)) + ' dead pixels & ' + str(len(alive_pixels)) + ' still alive')

        alive_float_array = zeros([xdim, ydim])
        for pixel in alive_pixels:
            alive_float_array[pixel.x][pixel.y] = pixel.val


        plotMatrixColor(alive_float_array)


        PlotListofClusterArraysColor(cluster_arrays, 1)
        pdb.set_trace()

        sub_cluster_count = 10









# TODO time to switch to sparse matrices, it seems that there are indeed computational optimizations

# TODO Look into DBSCAN from Sklearn as an alternate way to cluster
# TODO & other sklearn clustering techniques: http://scikit-learn.org/stable/modules/clustering.html


# plt.imsave for saving
#TODO recover and use more of the data from kmeans, like the centroids, which can be used to relate blobs?

# Now to start working on each cluster, and see if we can generate some blobs!!! :D
'''
Rules:
    A max pixel (mp) has value 255
    Around a pixel means the 8 pixels that are touching it in the 2d plane
        123
        4.5
        678
    Any mp next to each other belong together
    Any mp that has no mp around it is removed as noise
    TODO: https://books.google.com/books?id=ROHaCQAAQBAJ&pg=PA287&lpg=PA287&dq=python+group+neighborhoods&source=bl&ots=f7Vuu9CQdg&sig=l6ASHdi27nvqbkyO_VvztpO9bRI&hl=en&sa=X&ei=4COgVbGFD8H1-QGTl7aABQ&ved=0CCUQ6AEwAQ#v=onepage&q=python%20group%20neighborhoods&f=false
        Info on neighborhoods
'''


if __name__ == '__main__':
    main() # Run the main function






####NOTE: intersting to note that sparse groups of pixels, including noise in a general area are often grouped together, probably due to their comparable lack of grouping improvements.
    # May be able to exploit this when removing sparse pixels.

#NOTE: What would happen if did iterative k-means until all pixels of all groups were touching, or there was a group consisting of a single pixel?