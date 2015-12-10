from Pixel import *
import sys
import collections
# from serodraw import *
# from serodraw import setglobaldims
from myconfig import *
from Blob2d import *
import numpy as np

from PIL import Image
# import numpy as np
import time
import math



def setglobaldims(x, y, z):
    def setserodims(x, y, z):
        global xdim
        global ydim
        global zdim
        xdim = int(x)
        ydim = int(y)
        zdim = int(z)
    setserodims(x, y, z) # Need both calls, one to set the global vars in each file, otherwise they don't carry
    setseerodrawdims(x, y, z) # Even when using 'global'; one method needs to be in each file


class Slide:
    ''''
    Each slide holds the Blob2d's from a single scan image.
    Slides are compared to create 3d blobs.
    '''

    total_slides = 0
    sub_slides = 0

    def __init__(self, filename=None, matrix=None, height=None, quiet=False):
        # Note: Must include either filename or matrix
        # When given a matrix instead of a filename of an image, the assumption is that
        # We are computing over blob2ds from within a blob3d,ie experimenting with a subslide
        assert not (matrix is None and filename is None)
        slices = []
        self.t0 = time.time()
        self.isSubslide = False
        self.debugFlag = False
        if matrix is None: # Only done if this is a primary slide # FIXME
            self.id_num = Slide.total_slides
            self.height = Slide.total_slides
            Slide.total_slides += 1
            self.filename = filename
            self.primary_slide = True
            imagein = Image.open(filename)
            if not quiet:
                print('Starting on image: ' + filename)
            imarray = np.array(imagein)
            (self.local_xdim, self.local_ydim, self.local_zdim) =  (im_xdim, im_ydim, im_zdim) = imarray.shape[0],imarray.shape[1], self.height
            # setglobaldims(im_xdim * slide_portion, im_ydim * slide_portion, im_zdim * slide_portion) # TODO FIXME, remove when possible
            if not quiet:
                if len(imarray.shape) > 2:
                    print('The are ' + str(imarray.shape[2]) + ' channels')
                else:
                    print('There is one channel')
            image_channels = imagein.split()
            for s in range(len(image_channels)):  # Better to split image and use splits for arrays than to split an array
                buf = np.array(image_channels[s])
                slices.append(buf)
                if np.amax(slices[s]) == 0:
                    if not quiet:
                        print('Channel #' + str(s) + ' is empty')
        else:
            self.isSubslide = True
            slices = [matrix]
            self.local_xdim, self.local_ydim = matrix.shape
            self.id_num = Slide.sub_slides
            self.height = height
            Slide.sub_slides += 1
            self.primary_slide = False

        self.equivalency_set = set() # Used to keep track of touching blobs, which can then be merged. # NOTE, moving from blob2d
        pixels = []
        self.sum_pixels = 0
        for curx in range(self.local_xdim):
            for cury in range(self.local_ydim):
                pixel_value = slices[0][curx][cury] # CHANGED back,  FIXME # reversed so that orientation is the same as the original when plotted with a reversed y.
                if (pixel_value != 0):  # Can use alternate min threshold and <=
                    pixels.append(Pixel(pixel_value, curx, cury, self.id_num))
                    self.sum_pixels += pixel_value
        if not quiet:
            print('The are ' + str(len(pixels)) + ' non-zero pixels from the original ' + str(self.local_xdim * self.local_ydim) + ' pixels')
        pixels.sort(key=lambda pix: pix.val, reverse=True)# Note that sorting is being done like so to sort based on value not position as is normal with pixels. Sorting is better as no new list

        # Lets go further and grab the maximal pixels, which are at the front
        endmax = 0
        while (endmax < len(pixels) and pixels[endmax].val >= min_val_threshold ):
            endmax += 1
        if not self.isSubslide:
            if not quiet:
                print('There are ' + str(endmax) + ' pixels above the minimal threshold')
        # Time to pre-process the maximal pixels; try and create groups/clusters
        self.alive_pixels = filterSparsePixelsFromList(pixels[0:endmax], (self.local_xdim, self.local_ydim), quiet=quiet)
        self.alive_pixels.sort() # Sorted here so that in y,x order instead of value order
        alive_pixel_array = np.zeros([self.local_xdim, self.local_ydim], dtype=object)
        for pixel in self.alive_pixels:
            alive_pixel_array[pixel.x][pixel.y] = pixel
        (derived_ids, derived_count, num_ids_equiv) = self.assignPixelsToIds(self.alive_pixels, not self.isSubslide) # Note only printing when primary slide
        counter = collections.Counter(derived_ids)
        total_ids = len(counter.items())
        if not quiet:
            print('There were: ' + str(len(self.alive_pixels)) + ' alive pixels assigned to ' + str(total_ids) + ' blobs')
        most_common_ids = counter.most_common()# HACK Grabbing all for now, +1 b/c we start at 0 # NOTE Stored as (id, count)
        id_lists = getIdLists(self.alive_pixels, remap=remap_ids_by_group_size, id_counts=most_common_ids) # Hack, don't ned to supply id_counts of remap is false; just convenient for now
        self.blob2dlist = [] # Note that blobs in the blob list are ordered by number of pixels, not id, this makes merging faster

        for (blobnum, blobslist) in enumerate(id_lists):
            self.blob2dlist.append(Blob2d(blobslist[0].blob_id, blobslist, alive_pixel_array, self))

        # Note that we can now sort the Blob2d.equivalency_set b/c all blobs have been sorted
        self.equivalency_set = sorted(self.equivalency_set)
        if not self.isSubslide:
            if not quiet:
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
        for blob in self.blob2dlist: # NOTE Merging sets
            for equivlist in equiv_sets:
                if blob.id != equivlist[0] and blob.id in equivlist: # Not the base and in the list
                    blob.updateid(equivlist[0])
        self.blob2dlist = Blob2d.mergeblobs(self.blob2dlist) # NOTE, by assigning the returned Blob2d list to a new var, the results of merging can be demonstrated
        self.edge_pixels = []
        edge_lists = []
        for (blobnum, blobslist) in enumerate(self.blob2dlist):
            edge_lists.append(self.blob2dlist[blobnum].edge_pixels)
            self.edge_pixels.extend(self.blob2dlist[blobnum].edge_pixels)
        if not quiet:
            self.tf = time.time()
            printElapsedTime(self.t0, self.tf)
            print('')

    @staticmethod
    def setAllPossiblePartners(slidelist, **kwargs):
        debugflag = kwargs.get('debugflag', -1)
        debug2ds = kwargs.get('debugforb2ds',[])
        debugging = (debugflag == 1)# debug, set value as desired

        max_height = max(slide.height for slide in slidelist)
        slides_by_height = [[] for i in range(max_height + 1)]
        for slide in slidelist:
            slides_by_height[slide.height].append(slide)
        for height,slides_at_height in enumerate(slides_by_height[:-1]): # All but the highest slides
            for slide in slides_at_height:
                for blob in slide.blob2dlist:
                    for above_slide in slides_by_height[height + 1]:
                        blob.setPossiblePartners(above_slide, **kwargs)

    @staticmethod
    def setAllShapeContexts(slidelist):
        # Note Use the shape contexts approach from here: http://www.cs.berkeley.edu/~malik/papers/mori-belongie-malik-pami05.pdf
        # Note The paper uses 'Representative Shape Contexts' to do inital matching; I will do away with this in favor of checking bounds for possible overlaps
        for slide in slidelist:
            for blob in slide.blob2dlist:
                blob.setShapeContexts(36)


    def getNextBlobId(self):
        # Starts at 0
        self.id_num += 1
        return self.id_num - 1 # this -1 is so that id's start at zero

    def totalBlobs(self):
        ''' Allows access to class vars without class declaration'''
        return Blob2d.total_blobs

    def totalSlides(self):
        ''' Allows access to class vars without class declaration'''
        return Slide.total_slides

    @staticmethod
    def assignPixelsToIds(pixel_list, print_info=False):

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

        local_xdim = max(pixel.x for pixel in pixel_list) + 1
        local_ydim = max(pixel.y for pixel in pixel_list) + 1


        # local_xdim, local_ydim = local_dim_tuple
        vertical_offsets  = [-1, -1, -1, 0]#[1, 0, -1, -1]#,  0,   1, -1] #, 1, -1, 0, 1]
        horizontal_offsets = [-1, 0, 1, -1]#[-1, -1, -1, 0]#, 1, 1,  0] #, 0, 1, 1, 1]

        derived_count = 0
        derived_pixels = []
        derived_ids = []
        pixel_id_groups = []
        conflict_differences = []

        equivalent_labels = []

        pixel_array = np.zeros([local_xdim, local_ydim], dtype=object) # Can use zeros instead of empty; moderately slower, but better to have non-empty entries incase of issues
        for pixel in pixel_list:
            pixel_array[pixel.x][pixel.y] = pixel # Pointer :) Modifications to the pixels in the list affect the array
        for pixel in pixel_list: # Need second iteration so that all of the pixels of the array have been set
            if pixel.blob_id == -1: # Value not yet set
                xpos = pixel.x
                ypos = pixel.y
                for (horizontal_offset, vertical_offset) in zip(horizontal_offsets, vertical_offsets):
                    if (ypos + vertical_offset < local_ydim and ypos + vertical_offset >= 0 and xpos + horizontal_offset < local_xdim and xpos + horizontal_offset >= 0):  # Boundary check.
                        neighbor = pixel_array[xpos + horizontal_offset][ypos + vertical_offset]
                        if (neighbor != 0):
                            difference = abs(float(pixel.val) - float(neighbor.val)) # Note: Need to convert to floats, otherwise there's an overflow error due to the value range being int8 (0-255)
                            if difference <= max_val_step: # Within acceptrable bound to be grouped by id
                                if neighbor.blob_id != -1:
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
                                        pixel.blob_id = neighbor.blob_id
                                        derived_pixels.append(pixel)
                                        derived_ids.append(pixel.blob_id)
                                        derived_count += 1
                                        pixel_id_groups[pixel.blob_id].append(pixel)

            else:
                if debug_pixel_ops:
                    print('****Pixel:' + str(pixel) + ' already had an id when the cursor reached it')
            if pixel.blob_id == -1: # Didn't manage to derive an id_num from the neighboring pixels
                # FIXME
                pixel.blob_id = len(pixel_id_groups) # This is used to assign the next id to a pixel, using an id that is new
                # FIXME

                pixel_id_groups.append([pixel])
                derived_ids.append(pixel.blob_id) # Todo should refactor 'derived_ids' to be more clear
                equivalent_labels.append(pixel.blob_id) # Map the new pixel to itself until a low equivalent is found
                if debug_pixel_ops:
                    print('**Never derived a value for pixel:' + str(pixel) + ', assigning it a new one:' + str(pixel.blob_id))
        if debug_pixel_ops:
            print('EQUIVALENT LABELS: ' + str(equivalent_labels))
        # Time to clean up the first member of each id group-as they are skipped from the remapping
        id_to_reuse = []

        maxid = max(pixel.blob_id for pixel in pixel_list)
        # for id in range(self.id_num): # NOTE CHANGED 12/9/15 to allow compatibility without needing slide # TODO
        print('Max id:' + str(maxid))

        for id in range(maxid):
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

        for pixel in pixel_list:
            # print('DB:' + str(pixel.blob_id) + ' len el:' + str(len(equivalent_labels)))
            pixel.blob_id = equivalent_labels[pixel.blob_id]
        for id in range(len(derived_ids)):
            derived_ids[id] = equivalent_labels[derived_ids[id]]

        removed_id_count = 0
        for id in range(len(equivalent_labels)):
            if equivalent_labels[id] != id:
                removed_id_count += 1
        if print_info:
            print('There were ' + str(removed_id_count) + ' removed ids')

        # TODO: See if we can reverse the adjusting of the actual pixel ids until after the equivalent labels are cleaned up, to reflect the merged labels

        return (derived_ids, derived_count, removed_id_count)

    def __str__(self):
        return str('Slide <Id:' + str(self.id_num) + ' Num of Blob2ds:' + str(len(self.blob2dlist)) + '>')


class SubSlide(Slide):
    '''
    A slide that is created from a portion of a blob3d (one of its blob2ds)
    '''

    def __init__(self, sourceBlob2d, sourceBlob3d):
        super().__init__(matrix=sourceBlob2d.gen_saturated_array(), height=sourceBlob2d.slide.id_num, quiet=True) # NOTE can turn off quiet if desired

        assert(isinstance(sourceBlob2d, Blob2d))
        self.parentB3d = sourceBlob3d
        self.parentB2d = sourceBlob2d
        self.offsetx = sourceBlob2d.minx
        self.offsety = sourceBlob2d.miny
        # self.height = sourceBlob2d.slide.id_num
        for pixel in self.alive_pixels: # TODO this is part of the source of error, need to update offsets
            pixel.x += self.offsetx
            pixel.y += self.offsety
            pixel.z = self.height
        for blob2d in self.blob2dlist:
            blob2d.avgx += self.offsetx
            blob2d.avgy += self.offsety
            blob2d.minx += self.offsetx
            blob2d.miny += self.offsety
            blob2d.maxx += self.offsetx
            blob2d.maxy += self.offsety


    def __str__(self):
        return super().__str__() + ' <subslide>: Offset(x,y):(' + str(self.offsetx) + ',' + str(self.offsety) + ')' + ' height:' + str(self.height)

def timeNoSpaces():
    return time.ctime().replace(' ', '_').replace(':', '-')


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

def filterSparsePixelsFromList(listin, local_dim_tuple, quiet=False):
    local_xdim, local_ydim = local_dim_tuple
    max_float_array = np.zeros([local_xdim, local_ydim])
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
                    if (xpos + horizontal_offset < local_xdim and xpos + horizontal_offset >= 0 and ypos + vertical_offset < local_ydim and ypos + vertical_offset >= 0):  # Boundary check.
                        neighbors_checked += 1
                        cur_neighbor_val = max_float_array[xpos + horizontal_offset][ypos + vertical_offset]
                        if (cur_neighbor_val > 0):
                            buf_nzn += 1
                            if (cur_neighbor_val == 255):
                                buf_maxn += 1
                            buf_sumn += cur_neighbor_val
        pixel.setNeighborValues(buf_nzn, buf_maxn, buf_sumn, neighbors_checked)
        if buf_nzn >= minimal_nonzero_neighbors:
            filtered_pixels.append(pixel)
    if not quiet:
        print('There are ' + str(len(listin) - len(filtered_pixels)) + ' dead pixels & ' + str(len(filtered_pixels)) + ' still alive')
    return filtered_pixels

def setseerodrawdims(x,y,z):
    global xdim
    global ydim
    global zdim
    xdim = x
    ydim = y
    zdim = z

def  printElapsedTime(t0, tf, pad=''): # HACK FIXME REMOVE THIS AND IMPORT CORRECTLY
    temp = tf - t0
    m = math.floor(temp / 60)
    plural_minutes = ''
    if m > 1:
        plural_minutes = 's'
    if m > 0:
        print(pad + 'Elapsed Time: ' + str(m) + ' minute' + str(plural_minutes) + ' & %.0f seconds' % (temp % 60))
    else:
        print(pad + 'Elapsed Time: %.2f seconds' % (temp % 60))