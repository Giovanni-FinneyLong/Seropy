__author__ = 'gio'
# import matplotlib.pyplot as plt


# import matplotlib


from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
from numpy import zeros
import sys
import collections
from serodraw import *
from config import *


from mpl_toolkits.mplot3d import Axes3D
import pickle  # Note uses cPickle automatically ONLY IF python 3
from sklearn.preprocessing import normalize
from PIL import ImageFilter
from collections import OrderedDict
import readline
import code
import rlcompleter



def KMeansClusterIntoLists(array_in, num_clusters):
    'Take an array of floats and returns a list of lists, each of which contains all the pixels of a cluster'
    cluster_lists = [[] for i in range(num_clusters)]
    # for c in range(num_clusters):
    #     cluster_arrays.append(zeros([xdim, ydim])) # (r,c)
    (bookC, distortionC) = kmeans(array_in, num_clusters)
    (centLabels, centroids) = vq(array_in, bookC)
    for pixlabel in range(len(centLabels)):
        cluster = centLabels[pixlabel]
        # pix = max_pixel_array_floats[pixlabel]
        # cluster_arrays[cluster][pix[1]][pix[2]] = pix[0]
        cluster_lists[cluster].append(array_in[pixlabel])
    return cluster_lists


class Pixel:
    '''This class is being used to hold the coordinates, base info and derived info of a pixle of a single image\'s layer'''

    id_num = 0 # ***STARTS AT ZERO
    def __init__(self, value, xin, yin):
        self.x = xin  # The x coordinate, int
        self.y = yin  # The y coordinate, int
        self.val = value  # float
        self.nz_neighbors = 0
        self.maximal_neighbors = 0
        self.neighbor_sum = 0
        self.neighbors_checked = 0
        self.neighbors_set = False  # For keeping track later, incase things get nasty
        self.blob_id = 0 # 0 means that it is unset

    @staticmethod
    def getNextBlobId(): # Starts at 0
        Pixel.id_num += 1
        #print('New id count:' + str(Pixel.id_count))
        return Pixel.id_num - 1 #HACK this -1 is so that id's start at zero

    def setNeighborValues(self, non_zero_neighbors, max_neighbors, neighbor_sum, neighbors_checked):
        self.nz_neighbors = non_zero_neighbors  # The number out of the 8 surrounding pixels that are non-zero
        self.maximal_neighbors = max_neighbors
        self.neighbor_sum = neighbor_sum  # The sum of the surrounding 8 pixels
        self.neighbors_checked = neighbors_checked
        self.neighbors_set = True

    def setBlobID(self, new_val):
        self.blob_id = new_val

    def toTuple(self):
        return (self.val, self.x, self.y)

    def toArray(self):
        return np.array([self.val, self.x, self.y])

    def __str__(self):
        '''Method used to convert Pixel to string, generall for printing'''
        return str('P{[v:' + str(self.val) + ', x:' + str(self.x) + ', y:' + str(self.y) + '], [nzn:' + str(
            self.nz_neighbors) + ', mn:' + str(self.maximal_neighbors) + ', ns:' + str(
            self.neighbor_sum) + ', nc:' + str(self.neighbors_checked) + ']}')

    __repr__ = __str__

    def __lt__(self, other): # Used for sorting; 'less than'
        # Sort by y then x, so that (1,0) comes before (0,1) (x,y)
        if self.y < other.y:
            return True
        elif self.y == other.y:
            return self.x < other.x
        else:
            return False


def main():

    debug_pixel_ops = False
    remap_ids_by_group_size = True
    min_val_threshold = 250
    max_val_step = 5 # The maximim amount that two neighboring pixels can differ in val and be grouped by blob_id



    all_images = glob.glob(DATA_DIR + 'Swell*.tif')
    all_images = [all_images[0]]  # HACK

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
        tuples = []
        sum_pixels = 0

        for s in range(len(image_channels)):  # Better to split image and use splits for arrays than to split an array
            buf = np.array(image_channels[s])
            slices.append(buf)
            if np.amax(slices[s]) == 0:
                print('Slice #' + str(s) + ' is an empty slice')
        # im = Image.fromarray(np.uint8(cm.jet(slices[0])*255))
        # out = im.filter(ImageFilter.MaxFilter)

        (xdim, ydim) = slices[0].shape # HACK

        for pcol in range(xdim):
            for pix in range(ydim):
                pixel_value = slices[0][pcol][pix]
                # print('Pixel #' + str(pcol * xdim + pix) + ' = ' + str(pixel_value))
                if (pixel_value != 0):  # Can use alternate min threshold and <=
                    tuples.append((pixel_value, pcol, pix))
                    sum_pixels += pixel_value
        print('The are ' + str(len(tuples)) + ' non-zero pixels from the original ' + str(xdim * ydim) + ' pixels')

        tuples.sort(key=lambda tuplex: tuplex[0], reverse=True) # Note that sorting is being done like so to sort based on value not position as is normal with pixels. Sorting is better as no new list; saves 10MB!

        # Lets go further and grab the maximal pixels, which are at the front
        endmax = 0
        while (endmax < len(tuples) and tuples[endmax][0] >= min_val_threshold ):
            endmax += 1
        print('There are  ' + str(endmax) + ' maximal pixels')

        # Time to pre-process the maximal pixels; try and create groups/clusters
        max_tuple_list = tuples[0:endmax]  # Pixels with value 255
        max_pixel_list = [Pixel(float(i[0]), float(i[1]), float(i[2])) for i in max_tuple_list]
        max_tuples_as_arrays = np.asarray([(float(i[0]), float(i[1]), float(i[2])) for i in max_tuple_list])
        # NOTE: Is an array of shape (#pixels, 3), where each element is an array representing a tuple.
        # NOTE: This is the required format for kmeans/vq

        max_float_array = zeros([xdim, ydim])
        max_tuple_array = np.empty([xdim, ydim], dtype=object)
        for pixel in max_tuple_list:
            max_float_array[pixel[1]][pixel[2]] = pixel[0]  # Note Remember that these are pointers!
            max_tuple_array[pixel[1]][pixel[2]] = pixel

        # Now have labels in centLabels for each of the max_pixels
        cluster_count = 20
        cluster_lists = KMeansClusterIntoLists(max_tuples_as_arrays, cluster_count)
        # NOTE max_tuples_as_arrays is replacing max_float_array which was not working, kmeans takes arrays of arrays not arrays of tuples
        for i in range(len(cluster_lists)):
            print('Index:' + str(i) + ', size:' + str(len(cluster_lists[i])))  # + ' pixels:' + str(cluster_lists[i]))

        cluster_arrays = []  # Each entry is an array, filled only with the maximal values from the corresponding
        for cluster in range(cluster_count):
            cluster_arrays.append(zeros([xdim, ydim]))  # (r,c)
            for pixel in cluster_lists[cluster]:
                cluster_arrays[cluster][pixel[1]][pixel[2]] = int(pixel[0])

        # PlotListofClusterArraysColor(cluster_arrays, 1)
        dead_pixels = []  # Still in other list
        alive_pixels = []  # Could do set difference later, but this should be faster..
        for (pixn, pixel) in enumerate(
                max_pixel_list):  # pixel_number and the actual pixel (value, x-coordinate, y-coordinate)
            col = pixel.x  # Note: The naming scheme has been repaired
            row = pixel.y
            # Keep track of nz-neighbors, maximal-neighbors, neighbor sum
            buf_nzn = 0
            buf_maxn = 0
            buf_sumn = 0.
            neighbors_checked = 0
            for horizontal_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1
                for vertical_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1
                    if (vertical_offset != 0 or horizontal_offset != 0):  # Don't measure the current pixel
                        if (col + vertical_offset < xdim and col + vertical_offset >= 0 and row + horizontal_offset < ydim and row + horizontal_offset >= 0):  # Boundary check.
                            neighbors_checked += 1
                            cur_neighbor_val = max_float_array[col + vertical_offset][row + horizontal_offset]
                            if (cur_neighbor_val > 0):
                                buf_nzn += 1
                                if (cur_neighbor_val == 255):
                                    buf_maxn += 1
                                buf_sumn += cur_neighbor_val
                                # if buf_nzn != 0:
                                # print('Setting pixel vals to: nzn:' + str(buf_nzn) + ', maxn:' + str(buf_maxn) + ', sumn:' + str(buf_sumn))
            pixel.setNeighborValues(buf_nzn, buf_maxn, buf_sumn, neighbors_checked)
            if buf_nzn == 0:
                dead_pixels.append(pixel)
            else:
                alive_pixels.append(pixel)
        print('There are ' + str(len(dead_pixels)) + ' dead pixels & ' + str(len(alive_pixels)) + ' still alive')
        alive_float_array = zeros([xdim, ydim])
        alive_pixel_array = np.zeros([xdim, ydim], dtype=object) # Can use zeros instead of empty; moderately slower, but better to have non-empty entries incase of issues
        for pixel in alive_pixels:
            alive_float_array[pixel.x][pixel.y] = pixel.val
            alive_pixel_array[pixel.x][pixel.y] = pixel # Pointer :) Modifications to the pixels in the list affect the array

        derived_count = 0
        derived_pixels = []
        derived_ids = []
        pixel_id_groups = []

        conflict_differences = []
        alive_pixels.sort() # Sorted here so that still in order

        # NOTE Up increases downwards, across increaes to the right. (Origin top left)
        # Order of neighboring pixels visitation:
        # 0 1 2
        # 3 X 4
        # 5 6 7
        vertical_offsets   = [-1, -1 , -1,  0]
        horizontal_offsets = [-1,  0,   1, -1]

        for pixel in alive_pixels: # Need second iteration so that all of the pixels of the array have been set
            if pixel.blob_id == 0: # Value not yet set
                col = pixel.x
                row = pixel.y
                for (vertical_offset, horizontal_offset) in zip(horizontal_offsets, vertical_offsets):
                    if (col + vertical_offset < xdim and col + vertical_offset >= 0 and row + horizontal_offset < ydim and row + horizontal_offset >= 0):  # Boundary check.
                        neighbor = alive_pixel_array[col + vertical_offset][row + horizontal_offset]
                        if (neighbor != 0):
                            if abs(pixel.val - neighbor.val) <= max_val_step: # Within acceptrable bound to be grouped by id
                                if neighbor.blob_id != 0:
                                    if(pixel.blob_id != 0):
                                        if debug_pixel_ops:
                                            print('Pixel:' + str(pixel) + ' conflicts on neighbor with non-zero blob_id:' + str(neighbor))
                                        conflict_differences.append(abs(neighbor.val - pixel.val))
                                    else: # Pixel hasn't yet set it's id; give it the id of its neighbor
                                        if debug_pixel_ops:
                                            print('Assigning the derived id:' + str(neighbor.blob_id) + ' to pixel:' + str(pixel))
                                        pixel.blob_id = neighbor.blob_id
                                        derived_pixels.append(pixel)
                                        derived_ids.append(pixel.blob_id)
                                        derived_count += 1
                                        pixel_id_groups[pixel.blob_id].append(pixel)
                                elif pixel.blob_id != 0:
                                    # neighboring blob is a zero, and the current pixel has an id, so we can assign this id to the neighbor
                                    if debug_pixel_ops:
                                        print('Derived a neighbor\'s id: assigning id:' + str(pixel.blob_id) + ' to neigbor:' + str(neighbor) + ' from pixel:' + str(pixel))
                                    neighbor.blob_id = pixel.blob_id
                                    derived_pixels.append(neighbor)
                                    derived_ids.append(neighbor.blob_id)
                                    derived_count += 1
                                    pixel_id_groups[neighbor.blob_id].append(neighbor)
            if pixel.blob_id == 0:
                pixel.blob_id = Pixel.getNextBlobId()
                pixel_id_groups.append([pixel])
                derived_ids.append(pixel.blob_id) # Todo should refactor 'derived_ids' to be more clear
                #print('Never derived a value for pixel:' + str(pixel) + ', assigning it a new one:' + str(pixel.blob_id))


        counter = collections.Counter(derived_ids)

        print('Total Derived Count:' + str(derived_count))
        print('There were: ' + str(len(alive_pixels)) + ' alive pixels assigned to ' + str(Pixel.id_num) + ' ids')

        # top_common_id_count = Pixel.id_num# HACK Grabbing all for now, +1 b/c we start at 0

        most_common_ids = counter.most_common()#top_common_id_count) # NOTE Stored as (id, count)
        top_common_id_count = len(most_common_ids)

        id_arrays = []  # Each entry is an array, filled only with the maximal values from the corresponding
        remap = [None] * (top_common_id_count)
        # print(counter)
        # print(len(most_common_ids))
        # print(most_common_ids)


        #TODO DEBUG strange error where getting the most common from the counter is only getting a subset of the entries (540/564).

        for id in range(top_common_id_count): # Supposedly up to 2.5x faster than using numpy's .tolist()
            id_arrays.append(zeros([xdim, ydim]))  # (r,c)
            #print(id)
            remap[most_common_ids[id][0]] = id

        if remap_ids_by_group_size:
            for pixel in alive_pixels:
                id_arrays[remap[pixel.blob_id]][pixel.x][pixel.y] = int(pixel.val)
        else:
            for pixel in alive_pixels:
                id_arrays[pixel.blob_id][pixel.x][pixel.y] = int(pixel.val)


        AnimateClusterArraysGif(id_arrays, imagefile, 0)
        #PlotListofClusterArraysColor2D(id_arrays, 20)
        #PlotListofClusterArraysColor(id_arrays, 0)
        debug()
        pdb.set_trace()
        #AnimateClusterArrays(id_arrays, imagefile, 0)
        # AnimateClusterArraysGif(id_arrays, imagefile, 0)
        runShell()

        # NOTE 504 Ids generated using new neighbor filtering approach, but not yet using the new method of connected component labeling
        # NOTE there are some interesting pixel disparties, where a pixel

        sub_cluster_count = 10
        # findBestClusterCount(0, 100, 5)
        # MeanShiftCluster(max_float_array)
        # AffinityPropagationCluster(max_float_array):

# NOTE Now to start working on each cluster, and see if we can generate some blobs!!! :D
'''
My informal Rules:
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
    main()  # Run the main function

# TODO time to switch to sparse matrices, it seems that there are indeed computational optimizations
# TODO Look into DBSCAN from Sklearn as an alternate way to cluster
# TODO & other sklearn clustering techniques: http://scikit-learn.org/stable/modules/clustering.html
# TODO recover and use more of the data from kmeans, like the centroids, which can be used to relate blobs?
# TODO may want to use numpy 3d array over a list of 2d arrays; remains to be checked for speed/memory
# NOTE: intersting to note that sparse groups of pixels, including noise in a general area are often grouped together, probably due to their comparable lack of grouping improvements.
# May be able to exploit this when removing sparse pixels.
# NOTE: What would happen if did iterative k-means until all pixels of all groups were touching, or there was a group consisting of a single pixel?