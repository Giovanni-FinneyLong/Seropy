__author__ = 'gio'

import sys
import collections
from serodraw import *
from config import *

from PIL import ImageFilter
from collections import OrderedDict
import readline
import code
import rlcompleter
# from pympler import asizeof
from munkres import Munkres
import pickle # Note uses cPickle automatically ONLY IF python 3


def setglobaldims(x, y, z):
    def setserodims(x, y, z):
        global xdim
        global ydim
        global zdim
        xdim = x
        ydim = y
        zdim = z
    setserodims(x, y, z) # Need both calls, one to set the global vars in each file, otherwise they don't carry
    setseerodrawdims(x, y, z) # Even when using 'global'; one method needs to be in each file

class Blob2d:
    '''
    This class contains a list of pixels, which comprise a 2d blob on a single image
    '''

    # equivalency_set = set() # Used to keep track of touching blobs, which can then be merged.
    total_blobs = 0 # Note that this is set AFTER merging is done, and so is not modified by blobs



    def __init__(self, idnum, list_of_pixels, master_array, slide):
        self.id = idnum
        # TODO may want to sort list here...
        self.pixels = list_of_pixels
        self.num_pixels = len(list_of_pixels)
        sum_vals = 0
        self.avgx = 0
        self.avgy = 0
        self.sum_vals = 0
        self.master_array = master_array
        self.slide = slide
        self.possible_partners = [] # A list of blobs which MAY be part of the same blob3d as this blob2d
        self.partner_costs = [] # The minimal cost for the corresponding blob2d in possible_partners
        self.partner_indeces = [] # A list of indeces (row, column) for each

        self.partner_subpixels = [] # Each element is a list of pixels, corresponding to a subset of the edge pixels from the partner blob
                                    # The value of each element in the sublist for each partner is the index of the pixel from the corresponding partner
        self.my_subpixels = []      # Set of own subpixels, with each list corresponding to a list from partner_subpixels

        # TODO make sure the list is sorted
        self.minx = list_of_pixels[0].x
        self.miny = list_of_pixels[0].y
        self.maxx = self.minx
        self.maxy = self.miny
        min_nzn_pixel = list_of_pixels[0]
        for pixel in list_of_pixels:
            self.sum_vals += pixel.val
            self.avgx += pixel.x
            self.avgy += pixel.y
            if pixel.x < self.minx:
                self.minx = pixel.x
            if pixel.y <= self.miny:
                self.miny = pixel.y
            if pixel.x > self.maxx:
                self.maxx = pixel.x
            if pixel.y > self.maxy:
                self.maxy = pixel.y
            if pixel.nz_neighbors < min_nzn_pixel.nz_neighbors:
                self.min_nzn_pixel = pixel
        # Note for now, will use the highest non-zero-neighbor count pixel
        self.avgx /= self.num_pixels
        self.avgy /= self.num_pixels
        self.setEdge()
        self.setTouchingBlobs()
        self.center = (self.avgx, self.avgy) # TODO
        self.max_width = self.maxx-self.minx # TODO
        # self.min_width = -1 # TODO
        self.max_height = self.maxy-self.miny # TODO
        # self.min_height = -1 # TODO
        self.edge_radius = 0
        for pixel in self.edge_pixels:
            self.edge_radius += math.sqrt(math.pow(pixel.x - self.avgx, 2) + math.pow(pixel.y - self.avgy, 2))

         # TODO, derive from the edge pixels, using center from all pixels in blob

    def setEdge(self):
        self.edge_pixels = [pixel for pixel in self.pixels if pixel.nz_neighbors < 8]
        self.edge_pixels.sort()


    def setTouchingBlobs(self):
        '''
        Updates the equivalency set of the slide containing the given blob.
        This is used to combine blobs that are touching.
        '''
        self.touching_blobs = set()
        for pixel in self.edge_pixels:
            neighbors = pixel.getNeighbors(self.master_array)
            for curn in neighbors:
                if(curn != 0 and curn.blob_id != self.id):
                    self.touching_blobs.add(curn.blob_id) # Not added if already in the set.
                    if (curn.blob_id < self.id):
                        self.slide.equivalency_set.add((curn.blob_id, self.id))
                    else:
                        self.slide.equivalency_set.add((self.id, curn.blob_id))


    def updateid(self, newid):
        '''
        Update the blob's id and the id of all pixels in the blob
        Better this was, as we dont have to check each pixel's id, just each blobs (externally!) before chaging pixels
        '''
        self.id = newid
        for pix in self.pixels:
            pix.blob_id = newid


    def setPossiblePartners(self, slide):
        '''
        Finds all blobs in the given slide that COULD overlap with the given blob.
        These blobs could be part of the same blob3D (partners)
        '''
        # A blob is a possible partner to another blob if they are in adjacent slides, and they overlap in area
        # Overlap cases (minx, maxx, miny, maxy at play)
        #  minx2 <= (minx1 | max1) <= maxx2
        #  miny2 <= (miny1 | maxy1) <= maxy2

        # print('DEBUG Checking blob for possible partners:' + str(self) + ' xrange: (' + str(self.minx) + ',' + str(self.maxx) + '), yrange: (' + str(self.miny) + ',' + str(self.maxy) + ')')
        for blob in slide.blob2dlist:
            # print('DEBUG  Comparing against blob:' + str(blob) + ' xrange: (' + str(blob.minx) + ',' + str(blob.maxx) + '), yrange: (' + str(blob.miny) + ',' + str(blob.maxy) + ')')
            inBounds = False
            partnerSmaller = False

            if (blob.minx <= self.minx <= blob.maxx) or (blob.minx <= self.maxx <= blob.maxx): # Covers the case where the blob on the above slide is larger
                # Overlaps in the x axis; a requirement even if overlapping in the y axis
                if (blob.miny <= self.miny <= blob.maxy) or (blob.miny <= self.maxy <= blob.maxy):
                    inBounds = True
                    partnerSmaller = False
            if not inBounds:
                if (self.minx <= blob.minx <= self.maxx) or (self.minx <= blob.maxx <= self.maxx):
                    if (self.miny <= blob.miny <= self.maxy) or (self.miny <= blob.maxy <= self.maxy):
                        inBounds = True
                        partnerSmaller = True
            # If either of the above was true, then one blob is within the bounding box of the other
            if inBounds:
                self.possible_partners.append(blob)
                # print('DEBUG  Inspected blob was added to current blob\'s possible partners')
                # TODFO here we find a subset of the edge pixels from the potential partner that correspond to the area
                # NOTE use self.avgx, self.avgy
                if partnerSmaller:
                    # Use partner's (blob) midpoints, and expand a proportion of minx, maxx, miny, maxy
                    midx = blob.avgx
                    midy = blob.avgy
                    left_bound = midx - ((blob.avgx - blob.minx) * overscan_coefficient)
                    right_bound = midx + ((blob.maxx - blob.avgx) * overscan_coefficient)
                    down_bound = midy - ((blob.avgy - blob.miny) * overscan_coefficient)
                    up_bound = midy + ((blob.maxy - blob.avgy) * overscan_coefficient)
                else:
                    # Use partner's (blob) midpoints, and expand a proportion of minx, maxx, miny, maxy
                    midx = self.avgx
                    midy = self.avgy
                    left_bound = midx - ((self.avgx - self.minx) * overscan_coefficient)
                    right_bound = midx + ((self.maxx - self.avgx) * overscan_coefficient)
                    down_bound = midy - ((self.avgy - self.miny) * overscan_coefficient)
                    up_bound = midy + ((self.maxy - self.avgy) * overscan_coefficient)
                partner_subpixel_indeces = []
                my_subpixel_indeces = []
                for p_num, pixel in enumerate(blob.edge_pixels): # TODO TODO TODO IMPORTANT, need two way setup, for blob and self. FIXME
                    if left_bound <= pixel.x <= right_bound and down_bound <= pixel.y <= up_bound:
                        partner_subpixel_indeces.append(p_num)
                for p_num, pixel in enumerate(self.edge_pixels): # TODO TODO TODO IMPORTANT, need two way setup, for blob and self. FIXME
                    if left_bound <= pixel.x <= right_bound and down_bound <= pixel.y <= up_bound:
                        my_subpixel_indeces.append(p_num)
                self.partner_subpixels.append(partner_subpixel_indeces)
                self.my_subpixels.append(my_subpixel_indeces)


        self.partner_costs = [0] * len(self.possible_partners)
        # TODO update this method to do better filtering, like checking if the blobs are within each other etc


    def setShapeContexts(self, num_bins):
        '''
        Uses the methods described here: https://www.cs.berkeley.edu/~malik/papers/BMP-shape.pdf
        to set a shape context histogram (with num_bins), for each edge pixel in the blob.
        Note that only the edge_pixels are used to determine the shape context,
        and that only edge points derive a context.

        num_bins is the number of bins in the histogram for each point
        '''
        # Note making the reference point for each pixel itself
        # Note that angles are NORMALLY measured COUNTER-clockwise from the +x axis,
        # Note  however the += 180, used to remove the negative values,
        # NOTE  makes it so that angles are counterclockwise from the NEGATIVE x-axis
        edgep = len(self.edge_pixels)
        self.context_bins = np.zeros((edgep , num_bins)) # Each edge pixel has rows of num_bins each
        # First bin is 0 - (360 / num_bins) degress
        for (pix_num, pixel) in enumerate(self.edge_pixels):
            for (pix_num2, pixel2) in enumerate(self.edge_pixels):
                if pix_num != pix_num2: # Only check against other pixels.
                    distance = math.sqrt(math.pow(pixel.x - pixel2.x, 2) + math.pow(pixel.y - pixel2.y, 2))
                    angle = math.degrees(math.atan2(pixel2.y - pixel.y, pixel2.x - pixel.x)) # Note using atan2 handles the dy = 0 case
                    angle += 180
                    if not 0 <= angle <= 360:
                        print('\n\n\n--ERROR: Angle=' + str(angle))
                    # Now need bin # and magnitude for histogram
                    bin_num = math.floor((angle / 360.) * (num_bins - 1)) # HACK PSOE from -1
                    value = math.log(distance, 10)
                    # print('DB: Pixel:' + str(pixel) + ' Pixel2:' + str(pixel2) + ' distance:' + str(distance) + ' angle:' + str(angle) + ' bin_num:' + str(bin_num))
                    self.context_bins[pix_num][bin_num] += value


    def __str__(self):
        return str('B{id:' + str(self.id) + ', #P:' + str(self.num_pixels)) + ', #EP:' + str(len(self.edge_pixels)) + '}'

    __repr__ = __str__

    def totalBlobs(self):
        '''
        TODO depreciated, need to updated blob2d.total_blobs as merging blobs
        This is currently being updated at the end of the slide_creation
        '''
        return Blob2d.total_blobs


    @staticmethod
    def mergeblobs(bloblist):
        '''
        Returns a NEW list of blobs, which have been merged after having their ids updated (externally, beforehand)
        Use the global variable 'debug_set_merge' to control output
        '''
        newlist = []
        copylist = list(bloblist) # Hack, fix by iterating backwards: http://stackoverflow.com/questions/2612802/how-to-clone-or-copy-a-list-in-python
        if debug_set_merge:
            print('Blobs to merge:' + str(copylist))
        while len(copylist) > 0:
            blob1 = copylist[0]
            newpixels = []
            merged = False
            if debug_set_merge:
                print('**Curblob:' + str(blob1))
            for (index2, blob2) in enumerate(copylist[1:]):
                if blob2.id == blob1.id:
                    if debug_set_merge:
                        print('   Found blobs to merge: ' + str(blob1) + ' & ' + str(blob2))
                    merged = True
                    newpixels = newpixels + blob2.pixels
            if merged == False:
                if debug_set_merge:
                    print('--Never merged on blob:' + str(blob1))
                newlist.append(blob1)
                del copylist[0]
            else:
                if debug_set_merge:
                    print(' Merging, newlist-pre:' + str(newlist))
                if debug_set_merge:
                    print(' Merging, copylist-pre:' + str(copylist))
                index = 0
                while index < len(copylist):
                    if debug_set_merge:
                        print(' Checking to delete:' + str(copylist[index]))
                    if copylist[index].id == blob1.id:
                        if debug_set_merge:
                            print('  Deleting:' + str(copylist[index]))
                        del copylist[index]
                        index -= 1
                    index += 1
                newlist.append(Blob2d(blob1.id, blob1.pixels + newpixels, blob1.master_array, blob1.slide))
                if debug_set_merge:
                    print(' Merging, newlist-post:' + str(newlist))
                if debug_set_merge:
                    print(' Merging, copylist-post:' + str(copylist))
        if debug_set_merge:
            print('Merge result' + str(newlist))
        return newlist

class Stitches:
    """
    Only created when it is expected that two blobs from different slides belong to the same blob3d
    Contains the cost, and point information from stitching 2 blobs together.
    Contains 2 sets of mappings to their edge pixels
    As stitches are created before it is confirmed that two blobs may overlap or be suitable partners, the internal vars
        isConnected and isPartners indicate whether the Stitches are valid, in the sense of overlap and viable partners respectively
    """

    def pixelsInBounds(self, subsetblob, boundaryblob):
        """
        :param subsetblob: A blob2d from which we will find a subset of edge_pixels which are within an acceptable bound
        :param boundaryblob: A blob2d which is used to set the boundary constraints for pixels from subsetblob
        :param scale: Advised 1-1.2, the amount to expand the boundary of the boundary blob
            to contain the subset of the other blob
        :return: A list of pixels, chosen from subsetblob.edge_pixels,
            which are within the scaled boundary defined by boundaryblob
        """
        assert 0 < self.overscan_scale < 2
        left_bound = boundaryblob.avgx - ((boundaryblob.avgx - boundaryblob.minx) * self.overscan_scale)
        right_bound = boundaryblob.avgx + ((boundaryblob.maxx - boundaryblob.avgx) * self.overscan_scale)
        down_bound = boundaryblob.avgy - ((boundaryblob.avgy - boundaryblob.miny) * self.overscan_scale)
        up_bound = boundaryblob.avgy + ((boundaryblob.maxy - boundaryblob.avgy) * self.overscan_scale)
        boundedpixels = []
        for p_num, pixel in enumerate(subsetblob.edge_pixels): # TODO TODO TODO IMPORTANT, need two way setup, for blob and self. FIXME
            if left_bound <= pixel.x <= right_bound and down_bound <= pixel.y <= up_bound:
                boundedpixels.append(pixel)
        return boundedpixels


    def setShapeContexts(self, num_bins):
        '''
        Uses the methods described here: https://www.cs.berkeley.edu/~malik/papers/BMP-shape.pdf
        to set a shape context histogram (with num_bins), for each edge pixel in the blob.
        Note that only the edge_pixels are used to determine the shape context,
        and that only edge points derive a context.

        num_bins is the number of bins in the histogram for each point
        '''
        # Note making the reference point for each pixel itself
        # Note that angles are NORMALLY measured COUNTER-clockwise from the +x axis,
        # Note  however the += 180, used to remove the negative values,
        # NOTE  makes it so that angles are counterclockwise from the NEGATIVE x-axis
        ledgep = len(self.lowerpixels)
        uedgep = len(self.upperpixels)

        self.lower_context_bins = np.zeros((ledgep , num_bins)) # Each edge pixel has rows of num_bins each
        self.upper_context_bins = np.zeros((uedgep , num_bins)) # Each edge pixel has rows of num_bins each
        # First bin is 0 - (360 / num_bins) degress
        for (pix_num, pixel) in enumerate(self.lowerpixels):
            for (pix_num2, pixel2) in enumerate(self.lowerpixels):
                if pix_num != pix_num2: # Only check against other pixels.
                    distance = math.sqrt(math.pow(pixel.x - pixel2.x, 2) + math.pow(pixel.y - pixel2.y, 2))
                    angle = math.degrees(math.atan2(pixel2.y - pixel.y, pixel2.x - pixel.x)) # Note using atan2 handles the dy = 0 case
                    angle += 180
                    if not 0 <= angle <= 360:
                        print('\n\n\n--ERROR: Angle=' + str(angle))
                    # Now need bin # and magnitude for histogram
                    bin_num = math.floor((angle / 360.) * (num_bins - 1)) # HACK PSOE from -1
                    value = math.log(distance, 10)
                    # print('DB: Pixel:' + str(pixel) + ' Pixel2:' + str(pixel2) + ' distance:' + str(distance) + ' angle:' + str(angle) + ' bin_num:' + str(bin_num))
                    self.lower_context_bins[pix_num][bin_num] += value
        for (pix_num, pixel) in enumerate(self.upperpixels):
            for (pix_num2, pixel2) in enumerate(self.upperpixels):
                if pix_num != pix_num2: # Only check against other pixels.
                    distance = math.sqrt(math.pow(pixel.x - pixel2.x, 2) + math.pow(pixel.y - pixel2.y, 2))
                    angle = math.degrees(math.atan2(pixel2.y - pixel.y, pixel2.x - pixel.x)) # Note using atan2 handles the dy = 0 case
                    angle += 180
                    if not 0 <= angle <= 360:
                        print('\n\n\n--ERROR: Angle=' + str(angle))
                    # Now need bin # and magnitude for histogram
                    bin_num = math.floor((angle / 360.) * (num_bins - 1)) # HACK PSOE from -1
                    value = math.log(distance, 10)
                    # print('DB: Pixel:' + str(pixel) + ' Pixel2:' + str(pixel2) + ' distance:' + str(distance) + ' angle:' + str(angle) + ' bin_num:' + str(bin_num))
                    self.upper_context_bins[pix_num][bin_num] += value


    def munkresCost(self):
        """
        Finds the minimal cost of pairings between the chosen subsets of pixels and their generated context bins using their cost array
        Generates self.cost_array based on the subsets of pixels
        Then used the hungarian method (Munkres) to find the optimal subset of indeces & total cost from the given subset of edge_pixels
        :return:
        """
        def costBetweenPoints(bins1, bins2):
            assert len(bins1) == len(bins2)
            cost = 0
            for i in range(len(bins1)):
                if (bins1[i] + bins2[i]) != 0:
                    cost += math.pow(bins1[i] - bins2[i], 2) / (bins1[i] + bins2[i])
            return cost / 2


        def makeCostArray():
            """
            Generates a cost array (self.cost_array) from the subset of pixels
            :return:
            """
            # Setting up cost array with the costs between respective points
            ndim = min(len(self.lowerpixels), len(self.upperpixels)) # HACK HACK min for now, in the hopes that munkres can handle non-square matrices.
            # cost_array = np.zeros([len(blob1.edge_pixels), len(blob2.edge_pixels)])
            self.cost_array = np.zeros([ndim, ndim])
            for i in range(ndim):
                for j in range(ndim):
                    self.cost_array[i][j] = costBetweenPoints(self.lower_context_bins[i], self.upper_context_bins[j])
            return self.cost_array

        makeCostArray()
        munk = Munkres()
        # print('Upper blob:' + str(self.upperblob) + ' lowerblob:' + str(self.lowerblob))
        # print('Lower pix:' + str(self.lowerpixels) + ' upper pix:' + str(self.upperpixels))
        # print('Lower slide:' + str(self.lowerslidenum) + ' ' + ' upper slide' + str(self.upperslidenum))
        # print('About to find indeces for cost array:' + str(self.cost_array))
        self.indeces = munk.compute(np.copy(self.cost_array))
        self.cost = 0
        for row, col in self.indeces:
            self.cost += self.cost_array[row][col]

    def __str__(self):
        return str('Stitching slides:(' + str(self.lowerslidenum) + ',' + str(self.upperslidenum) + ') with blobs (' +
                   str(self.lowerblob.id) + ',' + str(self.upperblob.id) + '). Chose:' + str(len(self.lowerpixels)) +
                   '/' + str(len(self.lowerblob.edge_pixels)) + ' lower blob pixels and ' + str(len(self.upperpixels)) +
                   '/' + str(len(self.upperblob.edge_pixels)) + ' upper blob pixels. ' + 'Cost:' + str(self.cost))

    def __init__(self, lowerblob, upperblob, overscan_scale, num_bins):
        self.overscan_scale = overscan_scale
        self.num_bins = num_bins
        self.lowerslidenum = lowerblob.slide.id_num
        self.upperslidenum = upperblob.slide.id_num
        self.lowerblob = lowerblob
        self.upperblob = upperblob
        self.upperpixels = None
        self.lowerpixels = self.pixelsInBounds(lowerblob, upperblob) # TODO psoe on the order of lower and upper
        self.cost = -1
        self.isReduced = False # True when have chosen a subset of the edge pixels to reduce computation
        if len(self.lowerpixels) != 0: # Optimization
            self.upperpixels = self.pixelsInBounds(upperblob, lowerblob)
        if self.upperpixels is not None and len(self.upperpixels) != 0 and len(self.lowerpixels) != 0:
            # HACK
            # NOTE planning to reduce to a subset
            # NOTE 1:28 for (203,301) pre-opt, :37 for (174, 178), 66mins for (640, 616) -> 4 mins after optimization (picking half of each) -> 59 seconds with selective[::3]
            # NOTE After ::2 opt, total time for [:3] data slides = 10 mins 19 seconds, instead of ~ 2 hours, after selective[::3], total time = 6mins 49 seconds
            # selective [::3] with 5 slides = 36 mins
            if len(self.upperpixels) > 200 and len(self.lowerpixels) > 200:
                print('-->Too many pixels in below stitch, reducing to a subset, originally was: ' + str(len(self.lowerpixels)) +
                   '/' + str(len(self.lowerblob.edge_pixels)) + ' lower blob pixels and ' + str(len(self.upperpixels)) +
                   '/' + str(len(self.upperblob.edge_pixels)) + ' upper blob pixels.')
                pickoneover = 2
                self.isReduced = True
                if len(self.upperpixels) > 500 and len(self.lowerpixels) > 500:
                    pickoneover = 3

                self.upperpixels = self.upperpixels[::pickoneover] # Every pickoneover'th element
                self.lowerpixels = self.lowerpixels[::pickoneover] # HACK this is a crude way of reducing the number of pixels


            self.isConnected = True
            self.setShapeContexts(num_bins) # Set lower and upper context bins
            print('   ', end='') # Fo
            print(self)
            self.munkresCost() # Now have set self.cost and self.indeces and self.connect
            # TODO grade the stitching based on the cost and num of pixels, and then set isPartners
        else:
            self.isConnected = False

class Pixel:
    '''
    This class is being used to hold the coordinates, base info and derived info of a pixel of a single image\'s layer
    '''

    id_num = 0
    def __init__(self, value, xin, yin):
        self.x = xin  # The x coordinate, int
        self.y = yin  # The y coordinate, int
        self.val = value  # float
        self.nz_neighbors = 0
        self.maximal_neighbors = 0
        self.neighbor_sum = 0
        self.neighbors_checked = 0
        self.neighbors_set = False  # For keeping track later, in case things get nasty
        self.blob_id = -1 # 0 means that it is unset

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
        return str('P{[v:' + str(self.val) + ', x:' + str(self.x) + ', y:' + str(self.y) + '], id:' + str(self.blob_id) + '}')
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

    def getNeighbors(self, master_array): # TODO depreciated
        neighbors = []
        xpos = self.x
        ypos = self.y
        for horizontal_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1
            for vertical_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1
                if (vertical_offset != 0 or horizontal_offset != 0):  # Don't measure the current pixel
                    if (xpos + horizontal_offset < xdim and xpos + horizontal_offset >= 0 and ypos + vertical_offset < ydim and ypos + vertical_offset >= 0):  # Boundary check.
                        neighbors.append(master_array[xpos + horizontal_offset][ypos + vertical_offset])
        return neighbors

class Slide:
    ''''
    Each slide holds the Blob2d's from a single scan image.
    Slides are compared to create 3d blobs.
    '''

    total_slides = 0

    def __init__(self, filename):


        self.id_num = Slide.total_slides
        Slide.total_slides += 1


        self.t0 = time.time()
        self.filename = filename
        self.equivalency_set = set() # Used to keep track of touching blobs, which can then be merged. # NOTE, moving from blob2d
        imagein = Image.open(filename)
        print('Starting on image: ' + filename)
        imarray = np.array(imagein)
        (im_xdim, im_ydim, im_zdim) = imarray.shape
        setglobaldims(im_xdim, im_ydim, im_zdim) # TODO FIXME
        print('The are ' + str(zdim) + ' channels')
        image_channels = imagein.split()
        slices = []
        pixels = []
        self.sum_pixels = 0
        for s in range(len(image_channels)):  # Better to split image and use splits for arrays than to split an array
            buf = np.array(image_channels[s])
            slices.append(buf)
            if np.amax(slices[s]) == 0:
                print('Channel #' + str(s) + ' is empty')
        for curx in range(xdim):
            for cury in range(ydim):
                pixel_value = slices[0][curx][cury] # CHANGED back,  FIXME # reversed so that orientation is the same as the original when plotted with a reversed y.
                if (pixel_value != 0):  # Can use alternate min threshold and <=
                    pixels.append(Pixel(pixel_value, curx, cury))
                    self.sum_pixels += pixel_value
        print('The are ' + str(len(pixels)) + ' non-zero pixels from the original ' + str(xdim * ydim) + ' pixels')
        pixels.sort(key=lambda pix: pix.val, reverse=True)# Note that sorting is being done like so to sort based on value not position as is normal with pixels. Sorting is better as no new list

        # Lets go further and grab the maximal pixels, which are at the front
        endmax = 0
        while (endmax < len(pixels) and pixels[endmax].val >= min_val_threshold ):
            endmax += 1
        print('There are ' + str(endmax) + ' maximal pixels')

        # Time to pre-process the maximal pixels; try and create groups/clusters
        self.alive_pixels = filterSparsePixelsFromList(pixels[0:endmax])
        self.alive_pixels.sort() # Sorted here so that in y,x order instead of value order

        alive_pixel_array = zeros([xdim, ydim], dtype=object)
        for pixel in self.alive_pixels:
            alive_pixel_array[pixel.x][pixel.y] = pixel


        (derived_ids, derived_count, num_ids_equiv) = self.firstPass(self.alive_pixels)
        counter = collections.Counter(derived_ids)
        total_ids = len(counter.items())
        print('There were: ' + str(len(self.alive_pixels)) + ' alive pixels assigned to ' + str(total_ids) + ' blobs')
        most_common_ids = counter.most_common()# HACK Grabbing all for now, +1 b/c we start at 0 # NOTE Stored as (id, count)
        id_lists = getIdLists(self.alive_pixels, remap=remap_ids_by_group_size, id_counts=most_common_ids) # Hack, don't ned to supply id_counts of remap is false; just convenient for now
        self.blob2dlist = [] # Note that blobs in the blob list are ordered by number of pixels, not id, this makes merging faster

        for (blobnum, blobslist) in enumerate(id_lists):
            self.blob2dlist.append(Blob2d(blobslist[0].blob_id, blobslist, alive_pixel_array, self))

        # Note that we can now sort the Blob2d.equivalency_set b/c all blobs have been sorted
        self.equivalency_set = sorted(self.equivalency_set)
        print('Touching blobs: ' + str(self.equivalency_set))

        equiv_sets = []
        for (index, tuple) in enumerate(self.equivalency_set):
            found = 0
            found_set_indeces = []
            for (eqindex, eqset) in enumerate(equiv_sets):
                t1 = tuple[0] in eqset
                t2 = tuple[1] in eqset
                if not (t1 and t2):
                    if t1:
                        eqset.add(tuple[1])
                        found += 1
                        found_set_indeces.append(eqindex)
                    elif t2:
                        eqset.add(tuple[0])
                        found += 1
                        found_set_indeces.append(eqindex)
            if found == 0:
                equiv_sets.append(set([tuple[0], tuple[1]]))
            elif found > 1:
                superset = set([])
                for mergenum in found_set_indeces:
                    superset = superset | equiv_sets[mergenum]
                for delset in reversed(found_set_indeces):
                    del equiv_sets[delset]
                equiv_sets.append(superset)

        # Sets to lists for indexing,
        # Note that in python, sets are always unordered, and so a derivative list must be sorted.
        for (index,stl) in enumerate(equiv_sets):
            equiv_sets[index] = sorted(stl) # See note
        # print('Equivalency Sets after turned to lists: ' + str(equiv_sets))

        for blob in self.blob2dlist: # NOTE Merging sets
            for equivlist in equiv_sets:
                if blob.id != equivlist[0] and blob.id in equivlist: # Not the base and in the list
                    # print('old:' + str(blob) + ':' + str(blob.pixels[0]))
                    # print('  Found id:' + str(blob.id) + ' in eq:' + str(equivlist))
                    blob.updateid(equivlist[0])
                    # print('new:' + str(blob) + ':' + str(blob.pixels[0]))

        # print('Before Merging: ' + str(self.blob2dlist))
        # print('Equiv set:' + str(self.equivalency_set))
        self.blob2dlist = Blob2d.mergeblobs(self.blob2dlist) # NOTE, by assigning the returned Blob2d list to a new var, the results of merging can be demonstrated
        # print('After Merging: ' + str(self.blob2dlist))
        self.edge_pixels = []
        edge_lists = []
        for (blobnum, blobslist) in enumerate(self.blob2dlist):
            edge_lists.append(self.blob2dlist[blobnum].edge_pixels)
            self.edge_pixels.extend(self.blob2dlist[blobnum].edge_pixels)
        Blob2d.total_blobs += len(self.blob2dlist)
        self.tf = time.time()
        printElapsedTime(self.t0, self.tf)
        print('')

    def getNextBlobId(self): # Starts at 0
        self.id_num += 1
        return self.id_num - 1 # this -1 is so that id's start at zero

    def totalBlobs(self):
        ''' Allows access to class vars without class declaration'''
        return Blob2d.total_blobs
    def totalSlides(self):
        ''' Allows access to class vars without class declaration'''
        return Slide.total_slides


    def firstPass(self, pixel_list):

        # NOTE Vertical increases downwards, horizontal increases to the right. (Origin top left)
        # Order of neighboring pixels visitation:
        # 0 1 2
        # 3 X 4
        # 5 6 7
        # For 8 way connectivity, should check NE, N, NW, W (2,1,0,3)
        # For origin in the top left, = SE,S,SW,W

        # Note scanning starts at top left, and increases down, until resetting to the top and moving +1 column right
        # Therefore, the ONLY examined neighbors of any pixel are : 0, 3, 5 ,1
        # 0 = (-1, -1)
        # 3 = (-1, 0)
        # 5 = (-1, 1)
        # 1 = (0, -1


        vertical_offsets  = [-1, -1, -1, 0]#[1, 0, -1, -1]#,  0,   1, -1] #, 1, -1, 0, 1]
        horizontal_offsets = [-1, 0, 1, -1]#[-1, -1, -1, 0]#, 1, 1,  0] #, 0, 1, 1, 1]

        derived_count = 0
        derived_pixels = []
        derived_ids = []
        pixel_id_groups = []
        conflict_differences = []

        equivalent_labels = []

        pixel_array = np.zeros([xdim, ydim], dtype=object) # Can use zeros instead of empty; moderately slower, but better to have non-empty entries incase of issues
        for pixel in pixel_list:
            pixel_array[pixel.x][pixel.y] = pixel # Pointer :) Modifications to the pixels in the list affect the array
        for pixel in pixel_list: # Need second iteration so that all of the pixels of the array have been set
            if pixel.blob_id == -1: # Value not yet set
                xpos = pixel.x
                ypos = pixel.y
                # if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                #     print('New cursor pixel:' + str(pixel))
                for (horizontal_offset, vertical_offset) in zip(horizontal_offsets, vertical_offsets):
                    # if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                    #     print(' Trying offsets:' + str(horizontal_offset) + ':' + str(vertical_offset))
                    if (ypos + vertical_offset < ydim and ypos + vertical_offset >= 0 and xpos + horizontal_offset < xdim and xpos + horizontal_offset >= 0):  # Boundary check.
                        neighbor = pixel_array[xpos + horizontal_offset][ypos + vertical_offset]
                        # print('  Checking neigbor:' + str(neighbor) + 'at offsets:(' + str(horizontal_offset) + ',' + str(vertical_offset) +')')
                        if (neighbor != 0):
                            # if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                            #     print('   Pixel:' + str(pixel) + ' found a nzn:' + str(neighbor))
                            difference = abs(float(pixel.val) - float(neighbor.val)) # Note: Need to convert to floats, otherwise there's an overflow error due to the value range being int8 (0-255)
                            if difference <= max_val_step: # Within acceptrable bound to be grouped by id
                                # if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                                    # print('   DEBUG: Neighbor was within range.')
                                if neighbor.blob_id != -1:
                                    # if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG
                                    #     print('   DEBUG: Neighbor already has an id.')
                                    #     print('   curpixel:' + str(pixel))
                                    #     print('   neighbor:' + str(neighbor))
                                    if pixel.blob_id != -1 and pixel.blob_id != neighbor.blob_id:
                                        if debug_pixel_ops:
                                            print('\n*****Pixel:' + str(pixel) + ' conflicts on neighbor with non-zero blob_id:' + str(neighbor))
                                        conflict_differences.append(difference)

                                        if pixel.blob_id < neighbor.blob_id:
                                            pair = (pixel.blob_id, neighbor.blob_id)
                                        else:
                                            pair = (neighbor.blob_id, pixel.blob_id)
                                        # pair is (lower_id, higher_id); want lower id to dominate
                                        base = pair[0]
                                        while equivalent_labels[base] != base: # Id maps to a lower id
                                            base = equivalent_labels[base]
                                        equivalent_labels[pair[1]] = base # Remapped the larger value to the end of the chain of the smaller

                                    elif pixel.blob_id != neighbor.blob_id:
                                        # if debug_pixel_ops and pixel.y < debug_pixel_ops_y_depth: # DEBUG:
                                        #     print('**Assigning the derived id:' + str(neighbor.blob_id) + ' to pixel:' + str(pixel))
                                        pixel.blob_id = neighbor.blob_id
                                        derived_pixels.append(pixel)
                                        derived_ids.append(pixel.blob_id)
                                        derived_count += 1
                                        # print('DEBUG adding to group:' + str(pixel.blob_id) + '/' + str(len(pixel_id_groups)))
                                        pixel_id_groups[pixel.blob_id].append(pixel)

            else:
                if debug_pixel_ops:
                    print('****Pixel:' + str(pixel) + ' already had an id when the cursor reached it')
            if pixel.blob_id == -1: # Didn't manage to derive an id_num from the neighboring pixels
                pixel.blob_id = len(pixel_id_groups) # This is used to assign the next id to a pixel, using an id that is new


                pixel_id_groups.append([pixel])
                derived_ids.append(pixel.blob_id) # Todo should refactor 'derived_ids' to be more clear
                equivalent_labels.append(pixel.blob_id) # Map the new pixel to itself until a low equivalent is found
                if debug_pixel_ops:
                    print('**Never derived a value for pixel:' + str(pixel) + ', assigning it a new one:' + str(pixel.blob_id))
        if debug_pixel_ops:
            print('EQUIVALENT LABELS: ' + str(equivalent_labels))
        # Time to clean up the first member of each id group-as they are skipped from the remapping

        #TODO TODO need to do more fixes to the equivalent labels; basically condense them, to remove the issue of a trailing larger number
        print('Number of initial pixel ids before deriving equivalencies:' + str(self.id_num))
        id_to_reuse = []

        for id in range(self.id_num):
            if id not in equivalent_labels:
                if debug_blob_ids:
                    print('ID #' + str(id) + ' wasnt in the list, adding to ids_to _replace')
                id_to_reuse.append(id)
            else:
                if(len(id_to_reuse) != 0):
                    buf = id_to_reuse[0]
                    if debug_blob_ids:
                        print('Replacing ' + str(id) + ' with ' + str(buf) + ' and adding ' + str(id) + ' to the ids to be reused')
                    id_to_reuse.append(id)
                    for id_fix in range(len(equivalent_labels)):
                        if equivalent_labels[id_fix] == id:
                            equivalent_labels[id_fix] = buf
                    id_to_reuse.pop(0)
            if debug_blob_ids:
                print('New equiv labels:' + str(equivalent_labels))
        # DEBUG if debug_blob_ids:
        #   print('**********Remaining id_to_reuse: ' + str(id_to_reuse))


        for pixel in pixel_list:
            pixel.blob_id = equivalent_labels[pixel.blob_id]
        for id in range(len(derived_ids)):
            derived_ids[id] = equivalent_labels[derived_ids[id]]

        removed_id_count = 0
        for id in range(len(equivalent_labels)):
            if equivalent_labels[id] != id:
                removed_id_count += 1
        print('There were ' + str(removed_id_count) + ' removed ids')

        # TODO: See if we can reverse the adjusting of the actual pixel ids until after the equivalent labels are cleaned up, to reflect the merged labels

        return (derived_ids, derived_count, removed_id_count)


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
        for horizontal_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1 # TODO rteplace with getneighbors
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
        if buf_nzn >= minimal_nonzero_neighbors:
            filtered_pixels.append(pixel)
    print('There are ' + str(len(listin) - len(filtered_pixels)) + ' dead pixels & ' + str(len(filtered_pixels)) + ' still alive')
    return filtered_pixels


def getIdLists(pixels, **kwargs):
    '''
    Returns a list of lists, each of which corresponds to an id. If remapped, the first list is the largest
    KWArgs:
        remap=True => Remap blobs from largest to smallest pixel count
            Requires id_counts
        id_counts=Counter(~).most_common()
    '''
    do_remap = kwargs.get('remap', False)
    id_counts =  kwargs.get('id_counts',None)
    kwargs_ok = True
    if do_remap:
        if id_counts is None:
            print('>>>ERROR, if remapping, must supply id_counts (the n-most_common elements of a counter')
            kwargs_ok = False
    if kwargs_ok:
        id_lists = [[] for i in range(len(id_counts))]
        if do_remap:
            # remap = [None] * len(id_counts)
            remap = dict()

            for id_index, id in enumerate(range(len(id_counts))): # Supposedly up to 2.5x faster than using numpy's .tolist()
                # print(' id=' + str(id) + ' id_counts[id]=' + str(id_counts[id]) + ' id_counts[id][0]=' + str(id_counts[id][0]))
                remap[id_counts[id][0]] = id
            for pixel in pixels:
                id_lists[remap[pixel.blob_id]].append(pixel)
        else:
            for pixel in pixels:
                if pixel.blob_id >= len(id_counts):
                    print('DEBUG: About to fail:' + str(pixel)) # DEBUG
                id_lists[pixel.blob_id].append(pixel)
        return id_lists
    else:
        print('Issue with kwargs in call to getIdLists!!')


def munkresCompare(blob1, blob2):
    '''
    Uses the Hungarian Algorithm implementation from the munkres package to find an optimal combination of points
    between blob1 and blob2 as well as deriving the point->point relationships and storing them in indeces
    '''
    # TODO try this with and without manual padding; change min/max for ndim and change the end padding portion of makeCostArray

    def costBetweenPoints(bins1, bins2):
        assert len(bins1) == len(bins2)
        cost = 0
        for i in range(len(bins1)):
            debug_cost = cost
            if (bins1[i] + bins2[i]) != 0:
                cost += math.pow(bins1[i] - bins2[i], 2) / (bins1[i] + bins2[i])
            # if math.isnan(cost) and not math.isnan(debug_cost):
            #     print('Became nan, old val=' + str(debug_cost) + ' Pow part:' + str(math.pow(bins1[i] - bins2[i], 2)) + ' denomerator:' + str(bins1[i] + bins2[i]))
            #     print(' bins1:' + str(bins1[i]) + ' bins2:' + str(bins2[i]))
            #     buf = bins1[i] - bins2[i]
            #     print(' Buf=' + str(buf) + ' pow:' + str(math.pow(bins1[i] - bins2[i], 2)))
        return cost / 2

    def makeCostArray(blob1, blob2):
        # Setting up cost array with the costs between respective points
        ndim = max(len(blob1.edge_pixels), len(blob2.edge_pixels)) # HACK HACK min for now, in the hopes that munkres can handle non-square matrices.
        # cost_array = np.zeros([len(blob1.edge_pixels), len(blob2.edge_pixels)])
        cost_array = np.zeros([len(blob1.edge_pixels), len(blob2.edge_pixels)])

        for i in range(len(blob1.edge_pixels)):
            for j in range(len(blob2.edge_pixels)):
                cost_array[i][j] = costBetweenPoints(blob1.context_bins[i], blob2.context_bins[j])
        i = len(blob1.context_bins)
        j = len(blob2.context_bins)

        return cost_array

    # TODO run this on some simple examples provided online to confirm it is accurate.
    print('DB starting munkres compare')
    cost_array = makeCostArray(blob1, blob2)
    # print('Result from makeCostArray:' + str(cost_array))
    print('DB done making cost array')
    munk = Munkres()
    # print('Working on cost_array:' + str(np.copy(cost_array).tolist()))

    indeces = munk.compute(np.copy(cost_array).tolist())
    print('Done computing indeces')
    total_cost = 0
    # print(cost_array)
    for row, col in indeces:
        # print('Row=' + str(row) + ' Col=' + str(col))
        value = cost_array[row][col]
        total_cost += value
        # print ('(%d, %d) -> %d' % (row, col, value))
    # print('Total Cost = ' + str(total_cost))
    return total_cost, indeces


def doPickle(slidelist, stitchlist, filename):
    pickledict = dict()
    pickledict['slides'] = slidelist
    pickledict['stitches'] = stitchlist
    pickledict['xdim'] = xdim
    pickledict['ydim'] = ydim
    pickledict['zdim'] = zdim
    pickle.dump(pickledict, open(filename, "wb"))


def unPickle(filename):
        print('Loading from pickle')
        pickledict = pickle.load(open(filename, "rb"))
        slidelist = pickledict['slides']
        stitchlist = pickledict['stitches']
        xdim = pickledict['xdim']
        ydim = pickledict['ydim']
        zdim = pickledict['zdim']
        setglobaldims(xdim, ydim, zdim)
        return slidelist, stitchlist


def setAllPossiblePartners(slidelist):
    for slide_num, slide in enumerate(slidelist[:-1]): # All but the last slide
        for blob in slide.blob2dlist:
            blob.setPossiblePartners(slidelist[slide_num + 1])


def setAllShapeContexts(slidelist):
    # Use the shape contexts approach from here: http://www.cs.berkeley.edu/~malik/papers/mori-belongie-malik-pami05.pdf
    # The paper uses 'Representative Shape Contexts' to do inital matching; I will do away with this in favor of checking bounds for possible overlaps
    for slide in slidelist:
        for blob in slide.blob2dlist:
            blob.setShapeContexts(36)


def stitchAllBlobs(slidelist):
    stitchlist = []
    print('Beginning to stitch together blobs')
    for slide_num, slide in enumerate(slidelist):
        print('Starting slide #' + str(slide_num) + ', len(blob2dlist)=' + str(len(slide.blob2dlist)))
        for blob1 in slide.blob2dlist:
            print('  Starting on a new blob from bloblist:' + str(blob1) + ' which has:' + str(len(blob1.possible_partners)) + ' possible partners')
            # print('  Blob1 current parter_costs:' + str(blob1.partner_costs))

            for b2_num, blob2 in enumerate(blob1.possible_partners):
                print('   Comparing to blob2:' + str(blob2))
                t0 = time.time()
                bufStitch = Stitches(blob1, blob2, 1.1, 36)
                if bufStitch.isConnected:
                    stitchlist.append(bufStitch)
                    tf = time.time()
                    print('    ', end='') # Formatting output
                    printElapsedTime(t0, tf)
                    ''' # Code normally used to generate munkres costs for entire blob edge_pixels instead of subset
                    t0 = time.time()
                    total_cost, indeces = munkresCompare(blob1, blob2)
                    tf = time.time()
                    printElapsedTime(t0, tf)
                    print('Total_cost=' + str(total_cost))
                    blob1.partner_costs[b2_num] = total_cost
                    print('Indeces=' + str(indeces))
                    blob1.partner_indeces.append(indeces)
                    '''
    return stitchlist


def main():

    stitchlist = []
    if test_instead_of_data:
        picklefile = 'pickletest.pickle'
    else:
        picklefile = 'pickledata.pickle'


    if not dePickle:
        setMasterStartTime()
        if test_instead_of_data:
            dir = TEST_DIR
            extension = '*.png'
        else:
            dir = DATA_DIR
            extension = 'Swell*.tif'
        all_images = glob.glob(dir + extension)

        # # HACK
        # if not test_instead_of_data:
        #     all_images = all_images[:10]

        print(all_images)
        all_slides = []
        for imagefile in all_images:
            all_slides.append(Slide(imagefile)) # Pixel computations are done here, as the slide is created.
        # Note now that all slides are generated, and blobs are merged, time to start mapping blobs upward, to their possible partners

        setAllPossiblePartners(all_slides)
        setAllShapeContexts(all_slides)
        t_start_munkres = time.time()
        stitchlist = stitchAllBlobs(all_slides)
        t_finish_munkres = time.time()
        print('Done stitching together blobs, total time for all: ', end='')
        printElapsedTime(t_start_munkres, t_finish_munkres)
    else:
        all_slides, stitchlist = unPickle(picklefile)

    for stitch in stitchlist:
        print(stitch)


    anim_orders = [
    ('y+', 90+360, 60+360),
    ('x+', 360, 90+360) ]

    # plotSlidesVC(all_slides, stitchlist, stitches=True, edges=True, color='slides', subpixels=False, midpoints=True, context=False, animate=False, orders=anim_orders, canvas_size=(1000, 1000), gif_size=(400,400))#, color=None)
    if not dePickle:
        doPickle(all_slides, stitchlist, picklefile)
    plotSlidesVC(all_slides, stitchlist, stitches=True, edges=True, color='slides', subpixels=False, midpoints=False, context=False, animate=False, orders=anim_orders, canvas_size=(1000, 1000), gif_size=(400,400))#, color=None)
    debug()

    # plotSlidesVC(all_slides, edges=True, color='slides', midpoints=True, possible=True, context=True, canvas_size=(1000, 1000))#, color=None)
    # TODO had a memory error adding to view when midpoints = True
    debug()

    # Note took 10 mins 19 seconds for [:3] with ::2 opt



'''
    TODO: https://books.google.com/books?id=ROHaCQAAQBAJ&pg=PA287&lpg=PA287&dq=python+group+neighborhoods&source=bl&ots=f7Vuu9CQdg&sig=l6ASHdi27nvqbkyO_VvztpO9bRI&hl=en&sa=X&ei=4COgVbGFD8H1-QGTl7aABQ&ved=0CCUQ6AEwAQ#v=onepage&q=python%20group%20neighborhoods&f=false
        Info on neighborhoods
'''



if __name__ == '__main__':
    main()  # Run the main function

# TODO time to switch to sparse matrices, it seems that there are indeed computational optimizations
# TODO other sklearn clustering techniques: http://scikit-learn.org/stable/modules/clustering.html
