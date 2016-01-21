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
from Pixel import Pixel
from serodraw import warn


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
                    pixels.append(Pixel(pixel_value, curx, cury, self.id_num, validate=False))
                    self.sum_pixels += pixel_value
        if not quiet:
            print('The are ' + str(len(pixels)) + ' non-zero pixels from the original ' + str(self.local_xdim * self.local_ydim) + ' pixels')

        # Time to pre-process the maximal pixels; try and create groups/clusters
        self.alive_pixels = filterSparsePixelsFromList(pixels, (self.local_xdim, self.local_ydim), quiet=quiet)
        for pixel in self.alive_pixels:
            pixel.validate()

        alive_pixel_array = np.zeros([self.local_xdim, self.local_ydim], dtype=object)
        for pixel in self.alive_pixels:
            alive_pixel_array[pixel.x][pixel.y] = pixel
        if not quiet:
            print("Assigning pixels to ids")

        #HACK
        group_count = self.firstPass(self.alive_pixels, (self.local_xdim, self.local_ydim), not self.isSubslide) # Note only printing when primary slide
        #HACK

        # self.assignPixelsToIds(self.alive_pixels, True)

        #HACK

        if not quiet:
            print("Done assigning pixels to ids")

        # counter = collections.Counter(derived_ids)
        # total_ids = len(counter.items())
        if not quiet:
            print('There were: ' + str(len(self.alive_pixels)) + ' alive pixels assigned to ' + str(group_count) + ' blobs.')
        print('Getting id lists')
        # most_common_ids = counter.most_common()# HACK Grabbing all for now, +1 b/c we start at 0 # NOTE Stored as (id, count)
        t_idlist_0 = time.time()

        #FIXME this is failing b/c NONE of the pixels in self.alive_pixels have their b2d_id set

        id_lists = getIdLists(self.alive_pixels)
        t_idlist_f = time.time()
        # id_lists = getIdLists(self.alive_pixels) # Hack, don't ned to supply id_counts of remap is false; just convenient for now
        self.blob2dlist = [] # Note that blobs in the blob list are ordered by number of pixels, not id, this makes merging faster

        print('Creating b2ds from number of id_lists:' + str(len(id_lists)))
        for (blobnum, blobslist) in enumerate(id_lists):
            if len(blobslist): # This is now needed, as no longer compressing out unused ids
                newb2d = Blob2d(blobslist, self.height)
                self.blob2dlist.append(newb2d.id)

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

        #db_blob2d_list = [Blob2d.get(b2d) for b2d in self.blob2dlist]

        #print('DB working on slide, w/ blob2list: ' + str(db_blob2d_list))

        t_sort_0 = time.time()
        print('Sorting equiv sets, total of ' + str(len(equiv_sets)))
        for (index,stl) in enumerate(equiv_sets): # TODO this can be changed to be faster after b2ds are statically indexed
            equiv_sets[index] = sorted(stl)
        t_sort_f = time.time()
        t_update_0 = time.time()
        print('Updating ids')
        for blob in self.blob2dlist: # NOTE Merging sets
            for equivlist in equiv_sets:
                if blob != equivlist[0] and blob in equivlist: # Not the base and in the list
                    # blob.updateid(equivlist[0])
                    Blob2d.updateid(blob , equivlist[0])
        t_update_f = time.time()
        t_merge_0 = time.time()
        print('Merging blob2ds, size of current blob2dlist:' + str(len(self.blob2dlist)))
        # DEBUG
        print('First 100 of blob2dlist:')
        for b in self.blob2dlist[:100]:
            print(Blob2d.get(b))

        self.blob2dlist = Blob2d.mergeblobs(self.blob2dlist) # NOTE, by assigning the returned Blob2d list to a new var, the results of merging can be demonstrated
        print('After merging, size of blob2dlist:' + str(len(self.blob2dlist)))
        t_merge_f = time.time()
        print('Time to combine pixels into groups:')
        printElapsedTime(t_idlist_0, t_idlist_f)
        print('Time to sort:')
        printElapsedTime(t_sort_0, t_sort_f)
        print('Time to update:')
        printElapsedTime(t_update_0, t_update_f)
        print('Time to merge:')
        printElapsedTime(t_merge_0, t_merge_f)

        self.edge_pixels = []
        edge_lists = []
        print('Extending edge pixels')
        for (blobnum, blobslist) in enumerate(self.blob2dlist):
            edge_lists.append(Blob2d.get(self.blob2dlist[blobnum]).edge_pixels)
            self.edge_pixels.extend(Blob2d.get(self.blob2dlist[blobnum]).edge_pixels) # TODO remove
        if not quiet:
            self.tf = time.time()
            printElapsedTime(self.t0, self.tf)
            print('')

    @staticmethod
    def setAllPossiblePartners(slidelist):
        max_height = max(slide.height for slide in slidelist)
        slides_by_height = [[] for i in range(max_height + 1)]
        for slide in slidelist:
            slides_by_height[slide.height].append(slide)
        for height,slides_at_height in enumerate(slides_by_height[:-1]): # All but the highest slides
            for slide in slides_at_height:
                for blob in slide.blob2dlist:
                    for above_slide in slides_by_height[height + 1]:
                        Blob2d.get(blob).setPossiblePartners(above_slide.blob2dlist)

    @staticmethod
    def setAllShapeContexts(slidelist):
        # Note Use the shape contexts approach from here: http://www.cs.berkeley.edu/~malik/papers/mori-belongie-malik-pami05.pdf
        # Note The paper uses 'Representative Shape Contexts' to do inital matching; I will do away with this in favor of checking bounds for possible overlaps
        for slide in slidelist:
            for blob in slide.blob2dlist:
                Blob2d.get(blob).setShapeContexts(36)


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
        for p_num, pixel in enumerate(pixel_list): # Need second iteration so that all of the pixels of the array have been set
            # print('DB assigning pixel #' + str(p_num) + ' / ' + str(len(pixel_list)) + ' an id')

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

        # Time to clean up the first member of each id group-as they are skipped from the remapping
        id_to_reuse = []

        print('DB eq labels len (' + str(len(equivalent_labels)) + ')')
        maxid = max(pixel.blob_id for pixel in pixel_list)

        equivalent_labels_set = set(equivalent_labels)

        print('DB number of pixels: ' + str(len(pixel_list)))
        print('DB maxid: ' + str(maxid))
        print('Printing first 1000 of eq_labels:')
        for index,l in enumerate(equivalent_labels):
            if index != l:
                print(' Mismatch- Index:' + str(index) + ', eq_label:' + str(l))

        conversion_dict = {}

        for id_num, id in enumerate(range(maxid)):
            dbprint = False

            if id_num % 100 == 0 and debug_blob_ids:
                print('DB Working on id #' + str(id_num) + ' / ' + str(maxid))
                dbprint = True
            t_test =time.time()
            # if id not in equivalent_labels_set:
            if id != equivalent_labels[id]: # HACK trying this instead of the above for speeed

                id_to_reuse.append(id)
            else:
                if dbprint and debug_blob_ids:
                    printElapsedTime(t_test,time.time())
                    print(' ID #' + str(id) + ' WAS in the list, adding to ids_to _replace')
                    print('  ids entry is' + str(equivalent_labels[id]))
                    if id != equivalent_labels[id]:
                        print('\n\n\n Found an id in eq labels that didnt have a correspondig entry!!!!!')

                if(len(id_to_reuse) != 0):
                    buf = id_to_reuse[0]

                    if dbprint and debug_blob_ids:
                        print('   Replacing ' + str(id) + ' with ' + str(buf) + ' and adding ' + str(id) + ' to the ids to be reused')
                    id_to_reuse.append(id)
                    if dbprint and debug_blob_ids:

                        print(' -> Now updating ids...',flush=True) # TODO do the below outside for speed



                    # conversion_dict[id] = buf # Later we will convert pixels with bid = id to bid = buf

                    print('Normally would update conversion_dict with just ' + str(id) + ' -> ' + str(buf) + ', but chaining them all')
                    # DEBUG
                    cur_buf = id
                    cur_bufs_used = [buf]
                    while cur_buf in conversion_dict.keys():
                        print(' Swapping cur_buf from ' + str(cur_buf) + ' to ' + str(conversion_dict[cur_buf]))
                        cur_buf = conversion_dict[cur_buf]
                        cur_bufs_used.append(cur_buf)
                    if len(cur_bufs_used) > 1:
                        print(' Updating conversion_dict with ' + str(cb) + ' -> ' + str(cur_buf))
                    for cb in cur_bufs_used:
                        conversion_dict[cb] = cur_buf

                    # temp = buf in conversion_dict.keys()
                    # if temp:
                    #     print(' corresponding buf entry:' + str(conversion_dict[buf]))
                    #     print(' Corresponding entry in keys?' + str(conversion_dict[buf] in conversion_dict.keys()))

                    #TODO may have to check for chaining?

                    # update_count = 0
                    # for id_fix in range(len(equivalent_labels)):
                    #     if equivalent_labels[id_fix] == id:
                    #         print('Updating index of eql:' + str(id_fix) + ' from ' + str(id) + ' to ' + str(buf),flush=True)
                    #         equivalent_labels[id_fix] = buf
                    #         update_count+=1
                    # if update_count > 1:
                    #     print('\n\nMORE THAN ONE UPDATE!!!!')

                    # print('DONE UPDATING INDECES, update_count=' + str(update_count), flush=True)
                    id_to_reuse.pop(0)
                    if dbprint and debug_blob_ids:

                        print(' -> Done updating ids',flush=True) # TODO do
                else:
                    if dbprint and debug_blob_ids:
                        print('   no change b/c ids to reuse is empty')


        print('DB about to go through pixels again')
        print('DB conflicts:' + str(conflict_differences))
        for pixel in pixel_list:
            # print('DB:' + str(pixel.blob_id) + ' len el:' + str(len(equivalent_labels)))
            pixel.blob_id = equivalent_labels[pixel.blob_id]
            if pixel.blob_id in conversion_dict:
                # print('Found an id in conversion_dict, converting from ' + str(pixel.blob_id) + ' to ' + str(conversion_dict[pixel.blob_id]))
                pixel.blob_id = conversion_dict[pixel.blob_id]
        for id in range(len(derived_ids)):
            derived_ids[id] = equivalent_labels[derived_ids[id]]

        removed_id_count = 0
        for id in range(len(equivalent_labels)):
            if equivalent_labels[id] != id:
                removed_id_count += 1
        if print_info:
            print('There were ' + str(removed_id_count) + ' removed ids')

        # TODO: See if we can reverse the adjusting of the actual pixel ids until after the equivalent labels are cleaned up, to reflect the merged labels

        # return (derived_ids, derived_count, removed_id_count)

    def __str__(self):
        return str('Slide <Id:' + str(self.id_num) + ' Num of Blob2ds:' + str(len(self.blob2dlist)) + '>')
    def firstPass(self, pixel_list, local_dim_tuple, print_info):

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

            local_xdim, local_ydim = local_dim_tuple
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
            if print_info:
                print('Number of initial pixel ids before deriving equivalencies:' + str(self.id_num))
            id_to_reuse = []

            ############################
            # New code here:
            # Each index of eql contains the new_id that pixels with id = index should get mapped to
            # Instead of this structure, will store indeces that will end up as the same id (as in their contents equal that id)
            # together into a list of lists, which each index of this list of lists
            # Can use the id that they would normally go into to sort into lists
            # Then just just filter out empty lists from within the list of lists..?
            #----
            #Updates to approach:
            #

            maxid = max(pixel.blob_id for pixel in pixel_list)
            print('Entering new code, size of eql:' + str(len(equivalent_labels)))
            print('Maxid = ' + str(maxid))
            id_groups = [[] for i in range(maxid + 1)] # Note may need +1 todo
            print('Creating id groups', flush=True)
            for id,val in enumerate(equivalent_labels):
                # print(' eql[' + str(id) + '] = ' + str(val))
                id_groups[val].append(id)

            print('Eliminating empty entries from id_groups and casting into sets')
            id_groups = [set(idg) for idg in id_groups] # HACK DEBUG trying without remove empties; does this cause a direct mapping?
            # id_groups = [set(idg) for idg in id_groups if len(idg)] # Reduce to the non_empties, sets for faster searches
            # At this point, id_groups holds a list of sets, each set is a set of indeces that belong together
            print('Number of id_groups: ' +str(len(id_groups)))

            found_eq_val = set() #Debugging only
            print('Now moving on to pixel_list, which is of size: ' + str(len(pixel_list)))

            max_eq_val = max(equivalent_labels)
            remap = [-1 for i in range(max_eq_val+1)]
            print('Len of remap:' + str(len(remap)))

            if_calls = 0
            else_calls = 0
            t1_warn = 0
            t2_warn = 0
            t3_warn = 0
            t4_warn = 0

            for pixel_index,pixel in enumerate(pixel_list): # HACK on 100
                dbprint = False
                if pixel_index % 10000 == 0:
                    print('Now working on pixel_index:' + str(pixel_index) + ' / ' + str(len(pixel_list)))
                    print(' Working on pixel: ' + str(pixel))
                    dbprint = True


                # First grab the value in the index at eq_labels
                eq_val = equivalent_labels[pixel.blob_id]
                if dbprint:
                    print(' The corresponding index(the final group #) of eql is: ' + str(eq_val) + ' which we will look in id_groups for')
                    print('TEST:------->                                          ' + str(equivalent_labels[eq_val]))
                if equivalent_labels[eq_val] != eq_val:
                    warn("Found an area that could require fixing")
                    t1_warn += 1
                    # NOTE the direct mapping would be here
                if eq_val >= len(remap): #DEBUG
                    print(' About to fail, eq_val:' + str(eq_val))
                # print('Updating pixel: ' + str(pixel) + '\n with blob_id:' + str(eq_val) + ' and ' + str(remap[eq_val]))





                # pixel.blob_id = eq_val
                # Pixel.all[pixel.id].blob_id = eq_val


                if remap[eq_val] == -1: # Not yet looked up
                    if_calls += 1
                    for index, idg in enumerate(id_groups):
                        if eq_val in idg:
                            pixel.blob_id = index
                            Pixel.all[pixel.id].blob_id = index
                            if dbprint:
                                print('  Found at index of id_groups:' + str(index))
                                print('  eql[index]=' + str(equivalent_labels[index]))
                            if index != equivalent_labels[index]:
                                warn('warning #3!')
                                t3_warn += 1
                            remap[eq_val] = index
                            found_eq_val.add(eq_val)
                            break # No need to continue the for loop
                else:
                    else_calls += 1
                    # Have already looked this up, the correct new blob_id is in remap
                    # if dbprint:
                    #     print('  already had id_groups\'s index:' + str(remap[eq_val]))
                    #     print('  eql[index]=' + str(equivalent_labels[remap[eq_val]]))
                    if remap[eq_val] != eq_val: # DEBUG if this occurs, then have found sltn to be a recursive issue?
                        print('DB setting pixel.blob_id to:' + str(remap[eq_val]))
                        print('With other technique, would have set to: '  + str(eq_val))
                        warn('This means need to fix!!!!') # NOTE this does not occur in test dataset
                        t2_warn +=1
                    if remap[eq_val] != equivalent_labels[remap[eq_val]]:
                        t4_warn += 1

                    pixel.blob_id = remap[eq_val]
                    Pixel.all[pixel.id].blob_id = remap[eq_val]
                if dbprint:
                    print('  *If calls: ' + str(if_calls) + ', else calls: ' + str(else_calls) + ' ratio:' + str((if_calls + 1.0) * 1.0 / (1.0 + else_calls)))
                    print('  *t1_warn: ' +str(t1_warn) + ', t2_warn: ' + str(t2_warn) + ', t3_warn: ' + str(t3_warn) + ', t4_warn:' + str(t4_warn))
                # Now need to find which
            print('Number of unique eq_val found:' + str(len(found_eq_val)) + ' (this is essentially the number of b2ds from this slide')
            print('Final warning counts: t1_warn: ' +str(t1_warn) + ', t2_warn: ' + str(t2_warn) + ', t3_warn: ' + str(t3_warn) + ', t4_warn(linked with t3): ' + str(t4_warn))

            return len(found_eq_val)

            # NOTE for the testing dataset, this above has 216 non_empty_groups, and so seems to be working
            True # This is for breaking on


            # for id in range(self.id_num):
            #     if id != equivalent_labels[id]: # Checking if it maps to itself, otherwise it maps to something else
            #                                     # And so that id (index) is 'unused'
            #         if debug_blob_ids:
            #             print('ID #' + str(id) + ' wasnt in the list, adding to ids_to _replace')
            #         id_to_reuse.append(id)
            #     else:
            #         if(len(id_to_reuse) != 0):
            #             buf = id_to_reuse[0]
            #             if debug_blob_ids:
            #                 print('Replacing ' + str(id) + ' with ' + str(buf) + ' and adding ' + str(id) + ' to the ids to be reused')
            #             id_to_reuse.append(id)
            #             for id_fix in range(len(equivalent_labels)):
            #                 if equivalent_labels[id_fix] == id:
            #                     equivalent_labels[id_fix] = buf
            #             id_to_reuse.pop(0)
            #     if debug_blob_ids:
            #         print('New equiv labels:' + str(equivalent_labels))
            #
            # for pixel in pixel_list:
            #     pixel.blob_id = equivalent_labels[pixel.blob_id]
            # for id in range(len(derived_ids)):
            #     derived_ids[id] = equivalent_labels[derived_ids[id]]
            #
            # removed_id_count = 0
            # for id in range(len(equivalent_labels)):
            #     if equivalent_labels[id] != id:
            #         removed_id_count += 1
            # if print_info:
            #     print('There were ' + str(removed_id_count) + ' removed ids')
            #
            # # TODO: See if we can reverse the adjusting of the actual pixel ids until after the equivalent labels are cleaned up, to reflect the merged labels
            #
            # return (derived_ids, derived_count, removed_id_count)


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


    id_lists = [[] for i in range(max(pixel.blob_id for pixel in pixels) + 1)]
    print('DB len of id_lists: ' + str(len(id_lists)))


    for index,pixel in enumerate(pixels):
        try:
            id_lists[pixel.blob_id].append(pixel)
        except:
            print('DB max:' + str(max(pixel.blob_id for pixel in pixels) + 1))
            print('Index:' + str(index))
            exit(-1)

    print('DB printing each result of id_lists:')
    # for i in id_lists:
    #     print(i)
    return id_lists

def filterSparsePixelsFromList(listin, local_dim_tuple, quiet=False):
    # TODO convert to ids
    local_xdim, local_ydim = local_dim_tuple
    max_float_array = np.zeros([local_xdim, local_ydim])
    for pixel in listin:
        max_float_array[pixel.x][pixel.y] = pixel.val  # Note Remember that these are pointers!
    filtered_pixels = []
    removed_pixel_ids = []
    for (pixn, pixel) in enumerate(listin):  # pixel_number and the actual pixel (value, x-coordinate, y-coordinate)
        xpos = pixel.x  # Note: The naming scheme has been repaired
        ypos = pixel.y
        # Keep track of nz-neighbors, maximal-neighbors, neighbor sum
        buf_nzn = 0
        for horizontal_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1 # TODO rteplace with getneighbors
            for vertical_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1
                if (vertical_offset != 0 or horizontal_offset != 0):  # Don't measure the current pixel
                    if (xpos + horizontal_offset < local_xdim and xpos + horizontal_offset >= 0 and ypos + vertical_offset < local_ydim and ypos + vertical_offset >= 0):  # Boundary check.
                        # neighbors_checked += 1
                        cur_neighbor_val = max_float_array[xpos + horizontal_offset][ypos + vertical_offset]
                        if (cur_neighbor_val > 0):
                            buf_nzn += 1
        if buf_nzn >= minimal_nonzero_neighbors:
            filtered_pixels.append(pixel)
        else:
            removed_pixel_ids.append(pixel.id)
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
        print(pad + 'Elapsed Time: %.5f seconds' % (temp % 60))


