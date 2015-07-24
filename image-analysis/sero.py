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
        self.blob_id = -1 # 0 means that it is unset
        self.new_id = -1 # TODO this can be removed, just want to keep old value for debug reasons

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
        return str('P{[v:' + str(self.val) + ', x:' + str(self.x) + ', y:' + str(self.y) + '],' + str(self.blob_id) + '}')
            # '[nzn:' + str(
            # self.nz_neighbors) + ', mn:' + str(self.maximal_neighbors) + ', ns:' + str(
            # self.neighbor_sum) + ', nc:' + str(self.neighbors_checked) + ']}')

    __repr__ = __str__

    def __lt__(self, other): # Used for sorting; 'less than'
        # Sort by y then x, so that (1,0) comes before (0,1) (x,y)
        if self.y < other.y:
            return True
        elif self.y == other.y:
            return self.x < other.x
        else:
            return False

def filterSparsePixelsFromList(listin):
    max_float_array = zeros([xdim, ydim])
    for pixel in listin:
        max_float_array[pixel.x][pixel.y] = pixel.val  # Note Remember that these are pointers!
    filtered_pixels = []
    for (pixn, pixel) in enumerate(listin):  # pixel_number and the actual pixel (value, x-coordinate, y-coordinate)
        xpos = pixel.x  # Note: The naming scheme has been repaired
        ypos = pixel.y
        # Keep track of nz-neighbors, maximal-neighbors, neighbor sum
        buf_nzn = 0
        buf_maxn = 0
        buf_sumn = 0.
        neighbors_checked = 0
        for horizontal_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1
            for vertical_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1
                if (vertical_offset != 0 or horizontal_offset != 0):  # Don't measure the current pixel
                    if (xpos + horizontal_offset < xdim and xpos + horizontal_offset >= 0 and ypos + vertical_offset < ydim and ypos + vertical_offset >= 0):  # Boundary check.
                        neighbors_checked += 1
                        cur_neighbor_val = max_float_array[xpos + horizontal_offset][ypos + vertical_offset]
                        if (cur_neighbor_val > 0):
                            buf_nzn += 1
                            if (cur_neighbor_val == 255):
                                buf_maxn += 1
                            buf_sumn += cur_neighbor_val
                            # if buf_nzn != 0:
                            # print('Setting pixel vals to: nzn:' + str(buf_nzn) + ', maxn:' + str(buf_maxn) + ', sumn:' + str(buf_sumn))
        pixel.setNeighborValues(buf_nzn, buf_maxn, buf_sumn, neighbors_checked)
        if buf_nzn != 0:
            filtered_pixels.append(pixel)
    print('There are ' + str(len(listin) - len(filtered_pixels)) + ' dead pixels & ' + str(len(filtered_pixels)) + ' still alive')
    return filtered_pixels

def KMeansClusterIntoLists(listin, num_clusters):

    def doClustering(array_in, num_clusters):
        'Take an array of tuples and returns a list of lists, each of which contains all the pixels of a cluster'
        cluster_lists = [[] for i in range(num_clusters)]
        (bookC, distortionC) = kmeans(array_in, num_clusters)
        (centLabels, centroids) = vq(array_in, bookC)
        for pixlabel in range(len(centLabels)):
            cluster = centLabels[pixlabel]
            # pix = max_pixel_array_floats[pixlabel]
            # cluster_arrays[cluster][pix[1]][pix[2]] = pix[0]
            cluster_lists[cluster].append(array_in[pixlabel])
        return cluster_lists

    max_tuples_as_arrays = np.asarray([(float(pixel.val), float(pixel.x), float(pixel.y)) for pixel in listin])
    # NOTE: Is an array of shape (#pixels, 3), where each element is an array representing a tuple.
    # NOTE: This is the required format for kmeans/vq
    tuple_array = np.asarray([(float(pixel.val), float(pixel.x), float(pixel.y)) for pixel in listin])
    return doClustering(tuple_array, num_clusters)


def PlotClusterLists(list_of_lists):
    '''
    Takes a list of lists, each list is a the pixels of the corresponding cluster
    '''
    cluster_count = len(list_of_lists)
    cluster_arrays = []  # Each entry is an array, filled only with the maximal values from the corresponding
    for cluster in range(cluster_count):
        cluster_arrays.append(zeros([xdim, ydim]))  # (r,c)
        for pixel in list_of_lists[cluster]:
            cluster_arrays[cluster][pixel[1]][pixel[2]] = int(pixel[0])
    PlotListofClusterArraysColor(cluster_arrays, 1)

def firstPass(pixel_list):

    # NOTE Up increases downwards, across increaes to the right. (Origin top left)
    # Order of neighboring pixels visitation:
    # 0 1 2
    # 3 X 4
    # 5 6 7
    # For 8 way connectivity, should check NE, N, NW, W (2,1,0,3)
    horizontal_offsets = [-1,  0,   1, -1] #, 1, -1, 0, 1]
    vertical_offsets   = [-1, -1, -1,  0] #, 0, 1, 1, 1]


    derived_count = 0
    derived_pixels = []
    derived_ids = []
    pixel_id_groups = []
    conflict_differences = []

    pixel_array = np.zeros([xdim, ydim], dtype=object) # Can use zeros instead of empty; moderately slower, but better to have non-empty entries incase of issues
    for pixel in pixel_list:
        pixel_array[pixel.x][pixel.y] = pixel # Pointer :) Modifications to the pixels in the list affect the array
    for pixel in pixel_list: # Need second iteration so that all of the pixels of the array have been set
        if pixel.blob_id == -1: # Value not yet set
            xpos = pixel.x
            ypos = pixel.y
            if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                print('New cursor pixel:' + str(pixel))
            for (horizontal_offset, vertical_offset) in zip(horizontal_offsets, vertical_offsets):
                # if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                #     print(' Trying offsets:' + str(horizontal_offset) + ':' + str(vertical_offset))
                if (ypos + vertical_offset < ydim and ypos + vertical_offset >= 0 and xpos + horizontal_offset < xdim and xpos + horizontal_offset >= 0):  # Boundary check.
                    neighbor = pixel_array[xpos + horizontal_offset][ypos + vertical_offset]
                    # print('  Checking neigbor:' + str(neighbor) + 'at offsets:(' + str(horizontal_offset) + ',' + str(vertical_offset) +')')
                    if (neighbor != 0):
                        if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                            print('   Pixel:' + str(pixel) + ' found a nzn:' + str(neighbor))
                        if abs(pixel.val - neighbor.val) <= max_val_step: # Within acceptrable bound to be grouped by id
                            # if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                                # print('   DEBUG: Neighbor was within range.')
                            if neighbor.blob_id != -1:
                                if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                                    print('   DEBUG: Neighbor already has an id.')
                                    print('   ' + str(pixel))
                                    print('   ' + str(neighbor))
                                if(pixel.blob_id != -1 and pixel.blob_id != neighbor.blob_id):
                                    if debug_pixel_ops:
                                        print('  Pixel:' + str(pixel) + ' conflicts on neighbor with non-zero blob_id:' + str(neighbor))
                                    conflict_differences.append(abs(neighbor.val - pixel.val))
                                else: # Pixel hasn't yet set it's id; give it the id of its neighbor
                                    if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG:
                                        print('**Assigning the derived id:' + str(neighbor.blob_id) + ' to pixel:' + str(pixel))
                                    pixel.blob_id = neighbor.blob_id
                                    derived_pixels.append(pixel)
                                    derived_ids.append(pixel.blob_id)
                                    derived_count += 1
                                    pixel_id_groups[pixel.blob_id].append(pixel)
                            else:
                                if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                                    print('   DEBUG: neighbor didnt have an id')
                                if pixel.blob_id != -1:
                                    # neighboring blob is a zero, and the current pixel has an id, so we can assign this id to the neighbor
                                    if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG:
                                        print('**Derived a neighbor\'s id: assigning id:' + str(pixel.blob_id) + ' to neigbor:' + str(neighbor) + ' from pixel:' + str(pixel))
                                    neighbor.blob_id = pixel.blob_id
                                    derived_pixels.append(neighbor)
                                    derived_ids.append(neighbor.blob_id)
                                    derived_count += 1
                                    pixel_id_groups[neighbor.blob_id].append(neighbor)
        else:
            if debug_pixel_ops:
                print('Pixel:' + str(pixel) + ' already had an id when the cursor reached it')
        if pixel.blob_id == -1:
            pixel.blob_id = Pixel.getNextBlobId()
            pixel_id_groups.append([pixel])
            derived_ids.append(pixel.blob_id) # Todo should refactor 'derived_ids' to be more clear
            print('**Never derived a value for pixel:' + str(pixel) + ', assigning it a new one:' + str(pixel.blob_id))
    return (derived_ids, derived_count)













def main():


    global xdim # Must be declared global so that their values can adjusted, otherwise new locals are created when trying to mod
    global ydim # See: http://stackoverflow.com/questions/10588317/python-function-global-variables
    global zdim




    if(test_instead_of_data):
        all_images = glob.glob(TEST_DIR + '*.png')
    else:
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
        pixels = []
        sum_pixels = 0

        for s in range(len(image_channels)):  # Better to split image and use splits for arrays than to split an array
            buf = np.array(image_channels[s])
            slices.append(buf)
            if np.amax(slices[s]) == 0:
                print('Slice #' + str(s) + ' is an empty slice')
        # im = Image.fromarray(np.uint8(cm.jet(slices[0])*255))
        # out = im.filter(ImageFilter.MaxFilter)

        for curx in range(xdim):
            for cury in range(ydim):
                pixel_value = slices[0][curx][cury] # HACK
                if (pixel_value != 0):  # Can use alternate min threshold and <=
                    pixels.append(Pixel(pixel_value, curx, cury))
                    sum_pixels += pixel_value
        print('The are ' + str(len(pixels)) + ' non-zero pixels from the original ' + str(xdim * ydim) + ' pixels')
        pixels.sort(key=lambda pix: pix.val, reverse=True)# Note that sorting is being done like so to sort based on value not position as is normal with pixels. Sorting is better as no new list

        # Lets go further and grab the maximal pixels, which are at the front
        endmax = 0
        while (endmax < len(pixels) and pixels[endmax].val >= min_val_threshold ):
            endmax += 1
        print('There are ' + str(endmax) + ' maximal pixels')

        # Time to pre-process the maximal pixels; try and create groups/clusters
        max_pixel_list = pixels[0:endmax]
        #PlotClusterLists(KMeansClusterIntoLists(max_pixel_list, 20))

        alive_pixels = filterSparsePixelsFromList(max_pixel_list)
        alive_pixels.sort() # Sorted here so that still in order

        (derived_ids, derived_count) = firstPass(alive_pixels)


        counter = collections.Counter(derived_ids)
        print('Total Derived Count:' + str(derived_count))
        print('There were: ' + str(len(alive_pixels)) + ' alive pixels assigned to ' + str(Pixel.id_num) + ' ids')

        most_common_ids = counter.most_common()# HACK Grabbing all for now, +1 b/c we start at 0 # NOTE Stored as (id, count)
        top_common_id_count = len(most_common_ids)

        id_arrays = []  # Each entry is an array, filled only with the maximal values from the corresponding
        remap = [None] * (top_common_id_count)

        for id in range(top_common_id_count): # Supposedly up to 2.5x faster than using numpy's .tolist()
            id_arrays.append(zeros([xdim, ydim]))  # (r,c)
            remap[most_common_ids[id][0]] = id

        if remap_ids_by_group_size:
            for pixel in alive_pixels:
                id_arrays[remap[pixel.blob_id]][pixel.x][pixel.y] = int(pixel.val)
        else:
            for pixel in alive_pixels:
                id_arrays[pixel.blob_id][pixel.x][pixel.y] = int(pixel.val)


        # AnimateClusterArraysGif(id_arrays, imagefile, 0)
        PlotListofClusterArraysColor2D(id_arrays, 30, xdim, ydim)
        #PlotListofClusterArraysColor(id_arrays, 0)
        pdb.set_trace()
        #AnimateClusterArrays(id_arrays, imagefile, 0)
        # AnimateClusterArraysGif(id_arrays, imagefile, 0)


        # NOTE 504 Ids generated using new neighbor filtering approach, but not yet using the new method of connected component labeling
        # NOTE there are some interesting pixel disparties, where a pixel

        # TODO NOTICED THAT BOTTOM LEFT CORNER IS THE ORIGIN IN PYPLOT, whereas I expected it to be in the top left like in images


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