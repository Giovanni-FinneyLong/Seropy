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
        self.possible_partners = []
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
        self.touching_blobs = set()
        for pixel in self.edge_pixels:
            neighbors = pixel.getNeighbors(self.master_array)
            for curn in neighbors:
                if(curn != 0 and curn.blob_id != self.id):
                    # print('DEBUG: Found a neighboring blob:' + str(curn.blob_id))
                    self.touching_blobs.add(curn.blob_id) # Not added if already in the set.
                    if (curn.blob_id < self.id):
                        self.slide.equivalency_set.add((curn.blob_id, self.id))
                    else:
                        self.slide.equivalency_set.add((self.id, curn.blob_id))
        # if(len(self.touching_blobs)):
        #     print('Blob #' + str(self.id) + ' is touching blobs with ids:' + str(self.touching_blobs))

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
        Finds all blobs in the given slide that overlap with the given blob, and so could be part of the same blob (partners)
        '''
        # A blob is a possible partner to another blob if they are in adjacent slides, and they overlap in area
        # Overlap cases (minx, maxx, miny, maxy at play)
        #  minx2 <= (minx1 | max1) <= maxx2
        #  miny2 <= (miny1 | maxy1) <= maxy2

        for blob in slide.blob2dlist:
            if (blob.minx <= self.minx <= blob.maxx) or (blob.minx <= self.maxx <= blob.maxx):
                # Overlaps in the x axis; a requirement even if overlapping in the y axis
                if (blob.miny <= self.miny <= blob.maxy) or (blob.miny <= self.maxy <= blob.maxy):
                    self.possible_partners.append(blob)

    def setShapeContexts(self, num_bins):
        '''
        num_bins is the number of pins PER polar layer
        '''
        # Note making bin depth = 2
        # Note making the reference point for each pixel itself
        # Note that angles are NORMALLY measured COUNTER-clockwise from the +x axis,
        # Note  however the += 180, used to remove the negative values, makes it so that angles are counterclockwise from the NEGATIVE x-axis
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
        return Blob2d.total_blobs


    @staticmethod
    def mergeblobs(bloblist):
        '''
        Returns a NEW list of blobs, which have been merged after having their ids updated (externally, beforehand)
        '''
        newlist = []
        copylist = list(bloblist) # Hack, fix by iterating backwards: http://stackoverflow.com/questions/2612802/how-to-clone-or-copy-a-list-in-python
        if debug_set_merge:
            print('Blobs to merge:' + str(copylist))

        while len(copylist) > 0:
            blob1 = copylist[0]
            newpixels = []
            # killindeces = []
            merged = False
            # print(len(copylist))
            # print(copylist[1:])
            if debug_set_merge:
                print('**Curblob:' + str(blob1))
            for (index2, blob2) in enumerate(copylist[1:]):
                # if debug_set_merge:
                #     print('  DEBUG: checking blob:' + str(blob1) + ' against blob:' + str(blob2))
                if blob2.id == blob1.id:
                    if debug_set_merge:
                        print('   Found blobs to merge: ' + str(blob1) + ' & ' + str(blob2))
                    merged = True
                    newpixels = newpixels + blob2.pixels
                    # killindeces.append(index2 + 1) # Note +1 b/c offset by 1
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
                # print('Deleting ' + str(len(killindeces)) + ' elements with id:' + str(blob1.id))
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

    def getNeighbors(self, master_array):
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
        Slide.total_slides += 1

        self.id_num = 0
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
                print('Slice #' + str(s) + ' is an empty slice')
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
        print('There were: ' + str(len(self.alive_pixels)) + ' alive pixels assigned to ' + str(total_ids) + ' ids')
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
        print('Equivalency Sets after turned to lists: ' + str(equiv_sets))

        for blob in self.blob2dlist: # NOTE Merging sets
            for equivlist in equiv_sets:
                if blob.id != equivlist[0] and blob.id in equivlist: # Not the base and in the list
                    # print('old:' + str(blob) + ':' + str(blob.pixels[0]))
                    # print('  Found id:' + str(blob.id) + ' in eq:' + str(equivlist))
                    blob.updateid(equivlist[0])
                    # print('new:' + str(blob) + ':' + str(blob.pixels[0]))

        print('Before Merging: ' + str(self.blob2dlist))
        print('Equiv set:' + str(self.equivalency_set))
        self.blob2dlist = Blob2d.mergeblobs(self.blob2dlist) # NOTE, by assigning the returned Blob2d list to a new var, the results of merging can be demonstrated
        print('After Merging: ' + str(self.blob2dlist))
        self.edge_pixels = []
        edge_lists = []
        for (blobnum, blobslist) in enumerate(self.blob2dlist):
            edge_lists.append(self.blob2dlist[blobnum].edge_pixels)
            self.edge_pixels.extend(self.blob2dlist[blobnum].edge_pixels)
        Blob2d.total_blobs += len(self.blob2dlist)
        self.tf = time.time()
        printElapsedTime(self.t0, self.tf)
        # debug()
        # PlotClusterLists(newclusterlists, markersize=5)
        # PlotClusterLists(id_lists, dim='2d', markersize=5)#, numbered=True)
        # PlotClusterLists(edge_lists, dim='2d', markersize=10)#, numbered=True)
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
                                        pixel_id_groups[pixel.blob_id].append(pixel)

            else:
                if debug_pixel_ops:
                    print('****Pixel:' + str(pixel) + ' already had an id when the cursor reached it')
            if pixel.blob_id == -1:
                pixel.blob_id = self.getNextBlobId()
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


def getIdArrays(pixels, id_counts):
    '''
    Returns a list of filled arrays, each of which corresponds to an id. If remapped, the first array is most dense
    '''
    id_arrays = [zeros([xdim, ydim]) for _ in range(len(id_counts))]  # Each entry is an (r,c) array, filled only with the maximal values from the corresponding

    if remap_ids_by_group_size:
        remap = [None] * len(id_counts)
        for id in range(len(id_counts)): # Supposedly up to 2.5x faster than using numpy's .tolist()
            remap[id_counts[id][0]] = id
        for pixel in pixels:
            id_arrays[remap[pixel.blob_id]][pixel.x][pixel.y] = int(pixel.val)
    else:
        for pixel in pixels:
            if pixel.blob_id >= id_counts:
                print('DEBUG: About to fail:' + str(pixel))
            id_arrays[pixel.blob_id][pixel.x][pixel.y] = int(pixel.val)
    return id_arrays


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
            remap = [None] * len(id_counts)
            for id in range(len(id_counts)): # Supposedly up to 2.5x faster than using numpy's .tolist()
                remap[id_counts[id][0]] = id
            for pixel in pixels:
                id_lists[remap[pixel.blob_id]].append(pixel)
        else:
            for pixel in pixels:
                if pixel.blob_id >= len(id_counts):
                    print('DEBUG: About to fail:' + str(pixel)) # DEBUG
                id_lists[pixel.blob_id].append(pixel)
        return id_lists




def hungarianCompare(blob1, blob2):
    '''
    Uses the Hungarian Alg to find the optimal pairings of points between two Blob2D's
    '''
    # Each blob has its shape contexts (context_bins) set by here. Now for each point in b1 there is a row, and each in b2 a column.
    # One blob may have more points than the other, and so rows OR columns containing the max value of the matrix are added so that the matrix is nxn
    # TODO create a nested function to calculate cost between 2 points based on their histograms (contained in context_bins[point#)

    global debugval

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

    def printCostArrayLined(cost_array, row_lines, col_lines):
        print('**Lined Cost Array=')
        ndim = len(cost_array)
        for r in range(ndim):
            print('[', end='')
            for c in range(ndim):
                buf = []
                if r in row_lines:
                    buf.append('r')
                if c in col_lines:
                    buf.append('c')
                if len(buf) == 2:
                    print('+ ', end='')
                elif len(buf) == 1:
                    if 'r' in buf:
                        print('- ', end='')
                    else:
                        print('| ', end='')
                else:
                    print(cost_array[r][c], end=' ')
            print(']')


    ndim = max(len(blob1.edge_pixels), len(blob2.edge_pixels))
    cost_array = np.zeros([ndim, ndim])

    ''' # TEMP for DEBUG

    for i in range(len(blob1.edge_pixels)):
        for j in range(len(blob2.edge_pixels)):
            cost_array[i][j] = costBetweenPoints(blob1.context_bins[i], blob2.context_bins[j])
    i = len(blob1.context_bins)
    j = len(blob2.context_bins)
    print('i=' + str(i) + ' j=' + str(j))
    if i != j:
        # Need to find max value currently in the array, and use it to add rows or cols so that the matrix is square
        maxcost = np.amax(cost_array)
        if i < j:
            for r in range(i,j):
                for s in range(j):
                    cost_array[r][s] = maxcost # By convention
        else:
            for r in range(j,i):
                for s in range(i):
                    cost_array[r][s] = maxcost # By convention
    # Here the cost array is completely filled.
    # TODO run this on some simple examples provided online to confirm it is accurate.

    '''

    #HACK DEBUG
    cost_array = np.array([[10,19,8,15,19], # Note that after 10 major iterations, the generated matrix is almost exactly the same as the final matrix (which can be lined), the only difference is that the twos in the
                                            #Middle column should be ones
                  [10,18,7,17,19],
                  [13,16,9,14,19],
                  [12,19,8,18,19],
                  [14,17,10,19,19]])
    # cost_array = np.array([ # This gets the correct result as the prime/starred version
    #     [1,2,3],
    #     [2,4,6],
    #     [3,6,9]
    # ]
    # )
    # cost_array = np.array([ # From http://www.math.harvard.edu/archive/20_spring_05/handouts/assignment_overheads.pdf
    #                         # NOTE reduces correctly in 1 major iteration and 2 minor
    #     [250,400,350],
    #     [400,600,350],
    #     [200,400,250]
    # ])
    cost_array = np.array([ # From http://www.math.harvard.edu/archive/20_spring_05/handouts/assignment_overheads.pdf
        [90,75,75,80],
        [35,85,55,65],
        [125,95,90,105],
        [45,110,95,115]
    ])
    #HACK HACK
    wiki_not_harvard = False # NOTE If true, use the method from http://www.wikihow.com/Use-the-Hungarian-Algorithm else use method from: http://www.math.harvard.edu/archive/20_spring_05/handouts/assignment_overheads.pdf

    # cost_array = np.array([
    #     [0,1,0,0,5],
    #     [1,0,3,4,5],
    #     [7,0,0,4,5],
    #     [9,0,3,4,5],
    #     [3,0,3,4,5]
    # ])
    ndim = len(cost_array)
    original_cost_array = np.copy(cost_array)
    #HACK
    print('NDIM=' + str(ndim))
    print('Starting cost_array=\n' + str(cost_array)) # DEBUG



    # First subtract the min of each row from that row
    row_mins = np.amin(cost_array, axis=1) # An array where the nth element is the largest number in the nth row of cost_array
    print('Row mins found to be:' + str(row_mins))
    for row in range(len(cost_array)):
        cost_array[row] -= row_mins[row]
    print('After min per row subtracted cost_array=\n' + str(cost_array)) # DEBUG

    # Now if there are any cols without a zero, subtract the min of that column from the entire column (therefore creating a zero)
    # This is the equivalent of subtracting the min of a column by itself in all cases, as all values are non-negative
    col_mins = np.amin(cost_array, axis=0) # An array where the nth element is the largest number in the nth column of cost_array
    cost_array -= col_mins
    print('After min per col subtracted cost_array=\n' + str(cost_array)) # DEBUG


    # Now cover all zero elements with a minimal number of vertical/horizontal lines
    # Maintain a list of the number of zeros in each row/col




    iteration = 0

    lines_used = 0
    while lines_used != ndim:

        col_zeros = np.zeros([ndim])
        row_zeros = np.zeros([ndim])
        for row in range(ndim):
            for col in range(ndim):
                if not cost_array[row][col]:
                    col_zeros[col] += 1
                    row_zeros[row] += 1
        print('============================================\nIteration #' + str(iteration))

        # print('DB col_zeros:' + str(col_zeros))
        # print('DB row_zeros:' + str(row_zeros))
        total_zeros = sum(col_zeros) #len(col_zeros) + len(row_zeros)
        print('DB total_zeros=' + str(total_zeros))

        if iteration > 25: # DEBUG
            debug()
        lines_used = 0
        zeros_covered = 0
        row_lines = [] # Holds the indeces of lines drawn through rows
        col_lines = [] # Holds the indeces of lines drawn through columns
        last_line_horizontal = None # To be T/F
        next_line_horizontal = None
        last_line_index = -1
        next_line_index = -1

        print('About to start setting lines, total zeros=' + str(total_zeros) + ' zeros_covered=' + str(zeros_covered))
        # Now start setting the lines
        line_drawing_iteration = -1 # DEBUG
        while total_zeros != zeros_covered:
            line_drawing_iteration += 1
            # print(' Setting lines iteration #' + str(line_drawing_iteration) + ' zeros_covered/total_zeros=' + str(zeros_covered) + '/' + str(total_zeros))
            # print(' Cost_array=\n' + str(cost_array))
            # printCostArrayLined(cost_array, row_lines, col_lines)

            # print(' Col_zeros:' + str(col_zeros))
            # print(' Row_zeros:' + str(row_zeros))
            # print(' RowLines:' + str(row_lines))
            # print(' ColLines:' + str(col_lines))
            most_zeros_row = np.argmax(row_zeros) # An index not a value
            most_zeros_col = np.argmax(col_zeros) # An index not a value
            # print(' Most zeros (r,c) = (' + str(most_zeros_row) + ',' + str(most_zeros_col) + ')')
            if line_drawing_iteration == 6:
                print('hit 6 iterations, debugging')
                debug()

            max_covered = -1
            next_line_index = -1
            max_covered_r = -1
            max_covered_c = -1
            next_line_index_c = -1
            next_line_index_r = -1
            for r in range(ndim):# http://stackoverflow.com/questions/14795111/hungarian-algorithm-how-to-cover-0-elements-with-minimum-lines
                if(row_zeros[r] > max_covered_r or (row_zeros[r] == max_covered_r and last_line_horizontal == True)):
                    next_line_index_r = r
                    next_line_horizontal = True
                    max_covered_r = row_zeros[r]
            for c in range(ndim):
                if(col_zeros[c] > max_covered_c or (col_zeros[c] == max_covered_c and last_line_horizontal == False)):
                    next_line_index_c = c
                    next_line_horizontal = False
                    max_covered_c = col_zeros[c]
            # TODO fix the above by making it so that there is a preference for vertical if just did horizontal and vice versa.
            # Should involve separate max counters for each direction (already have in most_zeros_row/col)
            # So just need max_covered both verticall and horizontally, then compare and then set.
            print(' MAX COVERED R/C=' + str(max_covered_r) + ', ' + str(max_covered_c))

            if max_covered_r == max_covered_c:
                if not last_line_horizontal: # Prefer column as used column last time
                    next_line_index = next_line_index_c
                    next_line_horizontal = False
                    max_covered = max_covered_c
                else:
                    next_line_index = next_line_index_r
                    next_line_horizontal = True
                    max_covered = max_covered_r
            else:
                if max_covered_r > max_covered_c:
                    next_line_index = next_line_index_r
                    next_line_horizontal = True
                else:
                    next_line_index = next_line_index_c
                    next_line_horizontal = False
                max_covered = max(max_covered_c, max_covered_r)
                #TODO set max_covered, and next_line_index and next_line_horizontal

            # TODO now is done setting line early when it should be, although looks to be making progress, as drew the horizontal line at pos 2 after the vertical at pos 2
            # NOTE Current issue is that continue adding lines even when all elements are covered correctly
            # DEBUG NOTE STEP 5 of the online slides says to subtract from each uncovered row and add to each covered column,
            #       whereas the wikihow says subtract the min element from every element in the matrix
            print(' Max_covered_r=' + str(max_covered_r) + ', Max_covered_c=' + str(max_covered_c))
            print(' After iterating r/c found the best line index to be:' + str(next_line_index) + ' and next_line_horizontal=' + str(next_line_horizontal) + ', max_covered=' + str(max_covered))
            if next_line_horizontal:
                row_zeros[next_line_index] = 0
                for c in range(ndim):
                    if cost_array[next_line_index][c] == 0:
                        col_zeros[c] -= 1
                row_lines.append(next_line_index)
            else:
                col_zeros[next_line_index] = 0
                for r in range(ndim):
                    if cost_array[r][next_line_index] == 0:
                        row_zeros[r] -= 1
                col_lines.append(next_line_index)
            zeros_covered += max_covered
            last_line_horizontal = next_line_horizontal
            last_line_index = next_line_index

            def oldLinesMethod():
                # if row_zeros[most_zeros_row] > col_zeros[most_zeros_col]:
                #     # Set a 'line' through a row
                #     lines_used += 1
                #     zeros_covered += row_zeros[most_zeros_row]
                #     row_lines.append(most_zeros_row)
                #     row_zeros[most_zeros_row] = 0
                #     for col in range(ndim): # Updating the number of zeros in each column, as we have just removed some by creating the line
                #         if cost_array[most_zeros_row][col] == 0 and col not in col_lines:
                #             col_zeros[col] -= 1
                #             # DEBUG
                #             if col_zeros[col] < 0:
                #                 print('Error!!!! lt zero')
                #                 debug()
                # else:
                #     lines_used += 1
                #     zeros_covered += col_zeros[most_zeros_col]
                #     col_lines.append(most_zeros_col)
                #     col_zeros[most_zeros_col] = 0
                #     for row in range(ndim):
                #         if cost_array[row][most_zeros_col] == 0 and row not in row_lines:
                #             row_zeros[row] -= 1
                #             # DEBUG

                #             if row_zeros[row] < 0:
                #                 print('Error!!! lt zero')
                #                 debug()
                1
            if total_zeros < zeros_covered:
                print('Error, too many zeros covered')
                debug()

        lines_used = len(col_lines) + len(row_lines)
        print('DONE SETTING LINES to cover zeros, next find min uncovered element, lines_used=' + str(lines_used))
        printCostArrayLined(cost_array, row_lines, col_lines)
        # print('RowLines:' + str(row_lines))
        # print('ColLines:' + str(col_lines))
        print('Cost_array:\n' + str(cost_array))

        # Now find the minimal UNCOVERED element, and add it to every COVERED element
        if lines_used != ndim: # Can skip this if we've already got all the lines we need (ndim)
            min_uncovered = np.amax(cost_array)
            for row in range(ndim):
                for col in range(ndim):
                    if row not in row_lines and col not in col_lines:
                        min_uncovered = min(min_uncovered, cost_array[row][col])
            print('The min_uncovered value is:' + str(min_uncovered))


            #HACK DEBUG
            if wiki_not_harvard:
                # Now add the min_uncovered to the COVERED elements
                # Note that if an element is covered twice, we add the minimal element twice
                for row in range(ndim): # TODO this could be a bit more efficient by subtracting from the whole row/col at once
                    for col in range(ndim):
                        if row in row_lines:
                            cost_array[row][col] += min_uncovered
                        if col in col_lines:
                            cost_array[row][col] += min_uncovered
                print('After increasing ONLY covered by min uncovered, cost_array=\n' + str(cost_array))

                # Now subtract the minimal element from every element in the matrix
                arr_min = np.amin(cost_array) # This can never be zero, as all zeros had the minimal uncovered value added to them
                print('Minimal value of all elements=' + str(arr_min))
                # DEBUG
                if not arr_min:
                    print('Error, contained a zero value after all zeros were added to')
                    debug()
                cost_array -= arr_min
                print('Cost_array after subtracting min_element:\n' + str(cost_array))
                # now re-cover the zero elements
            else:
                # NOTE this is the harvard method, found here: http://www.math.harvard.edu/archive/20_spring_05/handouts/assignment_overheads.pdf
                # "Determine the smallest entry not covered by any line. Subtract this entry from each uncovered row, and then add it to each covered column.
                # print('USING HARVARD METHOD!')
                for row in range(ndim):
                    if row not in row_lines:
                        for col in range(ndim):
                            cost_array[row][col] -= min_uncovered
                print('Cost array after subtracting smallest entry from each UNCOVERED row:\n' + str(cost_array))
                for col in range(ndim):
                    if col in col_lines:
                        for row in range(ndim):
                            cost_array[row][col] += min_uncovered
                print('Cost array after adding smallest entry to each COVERED col:\n' + str(cost_array))
            iteration += 1



        else:
            print('DB SKIPPED SECOND PART AS NUM LINES ALREADY PERFECT')


    print('DB Done, now find a cover such that each row or column only has one zero selected ')
    print('Original cost array\n' + str(original_cost_array))
    print('Current cost array\n' + str(cost_array))
    printCostArrayLined(cost_array, row_lines, col_lines)
    print('---SUCCESSFULLY COMPLETED hungarian method')






    # if debugval == 1:
    #     debugval = 0
    #     for row in range(len(cost_array)):
    #         for col in range(len(cost_array[row])):
    #             print(str(cost_array[row][col]) + ', ', end='')
    #         print('')

    # print('Cost array = ' + str(cost_array))
    return 1






def main():
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
    #     all_images = all_images[:6]

    print(all_images)
    all_slides = []
    for imagefile in all_images:
        all_slides.append(Slide(imagefile)) # Computations are done here, as the slide is created.
        cur_slide = all_slides[-1]
    # Note now that all slides are generated, and blobs are merged, time to start mapping blobs upward, to their possible partners

    # TODO plotting midpoints is currently MUCH slower and more cpu instensive.
    for slide_num, slide in enumerate(all_slides[:-1]): # All but the last slide
        for blob in slide.blob2dlist:
            blob.setPossiblePartners(all_slides[slide_num + 1])
            # print(blob.possible_partners)

    anim_orders = [
        ('y+', 120, 120),
        ('x+', 360, 90) ]

    # plotSlidesVC(all_slides, edges=True, color='slides', midpoints=True, possible=True, animate=False, orders=anim_orders, canvas_size=(1000, 1000), gif_size=(500,500))#, color=None)

    # TODO: Match up Blob2Ds against their partners on either adjacent slide
    # Use the shape contexts approach from here: http://www.cs.berkeley.edu/~malik/papers/mori-belongie-malik-pami05.pdf
    # The paper uses 'Representative Shape Contexts' to do inital matching; I will do away with this in favor of checking bounds for possible overlaps
    # Components:
    #   Derive a context for every edge pixel
    #   Minimalize shape combo costs

    for slide in all_slides:
        for blob in slide.blob2dlist:
            print('Blob:' + str(blob))
            blob.setShapeContexts(36)
    for slide in all_slides:
        for blob in slide.blob2dlist:
            costs = []
            for blob2 in blob.possible_partners:
                buf = hungarianCompare(blob, blob2)
                debug()


    # mplfig = plt.figure()
    # for i in range(len(all_slides[0].blob2dlist[0].context_bins)):
    #     print('I=' + str(i))
    #     print(all_slides[0].blob2dlist[0].context_bins[i])
    #     plt.bar(range(36), all_slides[0].blob2dlist[0].context_bins[i])
    #     plt.show()

    # NOTE: Dummy rows/cols should be added to a hungarian matrix so that it is nxn and the Hungarian Method can be used.
    # The dummy rows/cols are filled with the largest value in the matrix
    # Blob1 will consist of N points, blob2 will consist of M points
    # The cost of matching a point Pi on the first shape to the point Qj on the second shape is:
    # Sum(Cost(




    debug()




    # plotSlides(all_slides)


        # findBestClusterCount(0, 100, 5)
        # MeanShiftCluster(max_float_array)
        # AffinityPropagationCluster(max_float_array):

'''
My informal Rules:
    Any pixel next to each other belong together
    Any pixel that has no pixels around it is removed as noise
    TODO: https://books.google.com/books?id=ROHaCQAAQBAJ&pg=PA287&lpg=PA287&dq=python+group+neighborhoods&source=bl&ots=f7Vuu9CQdg&sig=l6ASHdi27nvqbkyO_VvztpO9bRI&hl=en&sa=X&ei=4COgVbGFD8H1-QGTl7aABQ&ved=0CCUQ6AEwAQ#v=onepage&q=python%20group%20neighborhoods&f=false
        Info on neighborhoods
'''



if __name__ == '__main__':
    main()  # Run the main function

# TODO time to switch to sparse matrices, it seems that there are indeed computational optimizations
# TODO other sklearn clustering techniques: http://scikit-learn.org/stable/modules/clustering.html
