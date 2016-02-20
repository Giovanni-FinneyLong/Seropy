import collections
from Blob2d import Blob2d
import numpy as np
from PIL import Image
import time
from Pixel import Pixel
from myconfig import Config
from util import printElapsedTime, getImages, warn, progressBar
from Stitches import Pairing
from Blob3d import Blob3d

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
            (self.local_xdim, self.local_ydim, self.local_zdim) = imarray.shape[0],imarray.shape[1], self.height
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
            slices = [matrix]
            self.local_xdim, self.local_ydim = matrix.shape
            self.id_num = Slide.sub_slides
            self.height = height
            Slide.sub_slides += 1
            self.primary_slide = False

        pixels = []
        for curx in range(self.local_xdim):
            for cury in range(self.local_ydim):
                pixel_value = slices[0][curx][cury]
                if (pixel_value >= Config.min_val_threshold):
                    pixels.append(Pixel(pixel_value, curx, cury, self.id_num, validate=False)) # No need to validate at this point
        if not quiet:
            print('The are ' + str(len(pixels)) + ' pixels from the original ' + str(self.local_xdim * self.local_ydim) + ' pixels that are above the minimal pixel threshold')
        self.alive_pixels = filterSparsePixelsFromList(pixels, (self.local_xdim, self.local_ydim), quiet=quiet)

        if len(self.alive_pixels) == 0:
            warn('Didn\'t get any alive pixels from a slide!')
        else:
            self.assign_alive_pixels_to_blob2dlist(quiet=quiet)


    def assign_alive_pixels_to_blob2dlist(self, quiet=False):
        self.assignPixelsToIds(self.alive_pixels, Config.debug_pixel_ops) # Note only printing when primary slide
        id_lists = getIdLists(self.alive_pixels)
        self.blob2dlist = [] # Note that blobs in the blob list are ordered by number of pixels, not id, this makes merging faster
        for (blobnum, blobslist) in enumerate(id_lists):
            newb2d = Blob2d(blobslist, self.height)
            self.blob2dlist.append(newb2d.id)
        if not quiet:
            print('There were ' + str(len(self.alive_pixels)) + ' alive pixels assigned to ' + str(len(self.blob2dlist)) + ' blobs.')
            self.tf = time.time()
            print('Creating this slide took', end='')
            printElapsedTime(self.t0, self.tf, prefix='')
            print('')

    @staticmethod
    def assignPixelsToIds(pixel_list, print_info=False):

        # NOTE Vertical increases downwards, horizontal increases to the right. (Origin top left)
        # Order of neighboring pixels visitation:
        # 3 5 8
        # 2 X 7
        # 1 4 6
        # For 8 way connectivity, should check SW, W, NW, S (1,2,3,4)

        # Note scanning starts at top left, and increases down, until resetting to the top and moving +1 column right

        local_xdim = max(pixel.x for pixel in pixel_list) + 1
        local_ydim = max(pixel.y for pixel in pixel_list) + 1
        horizontal_offsets = [-1, -1, -1, 0]
        vertical_offsets = [-1, 0, 1, -1]
        equivalent_labels = []
        number_of_blob_ids = 0
        pixel_array = np.zeros([local_xdim, local_ydim], dtype=object) # Can use zeros instead of empty; moderately slower, but better to have non-empty entries incase of issues
        for pixel in pixel_list:
            pixel_array[pixel.x][pixel.y] = pixel # Pointer :) Modifications to the pixels in the list affect the array
        for pixel in pixel_list: # Need second iteration so that all of the pixels of the array have been set
            possible_ids = set()
            if pixel.blob_id == -1: # Value not yet set
                xpos = pixel.x
                ypos = pixel.y
                for (horizontal_offset, vertical_offset) in zip(horizontal_offsets, vertical_offsets):
                    if (ypos + vertical_offset < local_ydim and ypos + vertical_offset >= 0 and xpos + horizontal_offset < local_xdim and xpos + horizontal_offset >= 0):  # Boundary check.
                        neighbor = pixel_array[xpos + horizontal_offset][ypos + vertical_offset]
                        if (neighbor != 0):
                            # print(' Checking neighbor: ' + str(neighbor) + ' offsets: h:' + str(horizontal_offset) + ', v: ' + str(vertical_offset))
                            difference = abs(float(pixel.val) - float(neighbor.val)) # Note: Need to convert to floats, otherwise there's an overflow error due to the value range being int8 (0-255)
                            if difference <= Config.max_val_step: # Within acceptable bound to be grouped by id
                                if neighbor.blob_id != -1:
                                    possible_ids.add(neighbor.blob_id)
            if len(possible_ids) == 0:
                pixel.blob_id = number_of_blob_ids
                equivalent_labels.append(pixel.blob_id) # Map the new pixel to itself until a low equivalent is found
                number_of_blob_ids += 1
            elif len(possible_ids) == 1:
                pixel.blob_id = possible_ids.pop() # Note that this changes possible_ids!
            else:
                p_ids_list = list(possible_ids)
                eql_visted = set()
                for index, eql in enumerate(p_ids_list):
                    #looking at each id, and finding which one maps to lowest thing
                    while(equivalent_labels[eql] != eql): # Maps to a smaller value
                        eql = equivalent_labels[eql]
                        eql_visted.add(eql)
                    eql_visted.add(eql)
                low_eql = min(eql_visted)
                for eql in eql_visted:
                    equivalent_labels[eql] = low_eql
                pixel.blob_id = low_eql
        if Config.debug_pixel_ops:
            print('\nDOING FINAL ASSIGNMENTS:')
            print(list(enumerate(equivalent_labels)))
        for pixel in pixel_list:
            # print(str(pixel) + ' -> ', end='')
            base = equivalent_labels[pixel.blob_id]
            while(base != equivalent_labels[base]):
                base = equivalent_labels[base]
            pixel.blob_id = equivalent_labels[base]
            # print(' ' + str(pixel.blob_id))

    @staticmethod
    def dataToSlides(stitch=True):
        t_gen_slides_0 = time.time()
        all_images = getImages()
        all_slides = []
        for imagefile in all_images:
            all_slides.append(Slide(imagefile)) # Pixel computations are done here, as the slide is created.
        print('Total # of non-zero pixels: ' + str(Pixel.total_pixels) + ', total number of pixels after filtering: ' + str(len(Pixel.all)))
        print('Total # of blob2ds: ' + str(len(Blob2d.all)))
        print('Generating ' + str(len(all_slides)) + ' slides took', end='')
        printElapsedTime(t_gen_slides_0, time.time(), prefix='')
        print("Pairing all blob2ds with their potential partners in adjacent slides", flush=True)
        Slide.setAllPossiblePartners(all_slides)

        if stitch:
            print('Setting shape contexts for all blob2ds ',flush=True, end="")
            Slide.setAllShapeContexts(all_slides)
            t_start_munkres = time.time()
            stitchlist = Pairing.stitchAllBlobs(all_slides, debug=False) # TODO change this to work with a list of ids or blob2ds
            t_finish_munkres = time.time()
            print('Done stitching together blobs, ', end='')
            printElapsedTime(t_start_munkres, t_finish_munkres)
        else:
            print('\n-> Skipping stitching the slides, this will result in less accurate blob3ds for the time being')
        blob3dlist = Slide.extract_blob3ds(all_slides, stitched=stitch)
        print('There are a total of ' + str(len(blob3dlist)) + ' blob3ds')
        return all_slides, blob3dlist  # Returns slides and all their blob3ds in a list

    @staticmethod
    def extract_blob3ds(all_slides, stitched=True):
        print('Extracting 3d blobs by combining 2d blobs into 3d', flush=True)
        blob3dlist = []
        if not stitched:
            warn('Extracting blob3ds, and have been told that they haven\'t been stitched. This will be inaccurate')
            print('Extracting blob3ds, and have been told that they haven\'t been stitched. This will be inaccurate') #DEBUG

        for slide_num, slide in enumerate(all_slides):
            for blob in slide.blob2dlist:
                if Blob2d.get(blob).b3did == -1:
                    if stitched: # The much better option! ESPECIALLY for recursive_depth = 0
                        buf = [b2d for b2d in Blob2d.get(blob).get_stitched_partners()] #old method
                        # buf = [Blob2d.get(b2d) for b2d in Blob2d.get(blob).getpartnerschain()] # IDEALLY could use this for both... for now, it doesnt work well
                    else:
                        buf = [Blob2d.get(b2d) for b2d in Blob2d.get(blob).getpartnerschain()] # TODO setting partners needs filtering like stitching
                # HACK refresh buf
                buf = [Blob2d.get(b2d.id) for b2d in buf]
                buf = [b2d for b2d in buf if b2d.b3did == -1]
                if len(buf) != 0:
                    blob3dlist.append(Blob3d([b2d.id for b2d in buf]))
        return blob3dlist


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
        t0 = time.time()
        pb = progressBar(max_val=sum(len(Blob2d.get(b2d).edge_pixels) for slide in slidelist for b2d in slide.blob2dlist))
        for slide in slidelist:
            for blob in slide.blob2dlist:
                Blob2d.get(blob).setShapeContexts(36)
                pb.update(len(Blob2d.get(blob).edge_pixels), set=False)
        pb.finish()
        printElapsedTime(t0, time.time(), prefix='took')

    def getNextBlobId(self):
        # Starts at 0, of course!!
        self.id_num += 1
        return self.id_num - 1 # this -1 is so that id's start at zero

    def totalBlobs(self):
        ''' Allows access to class vars without class declaration'''
        return Blob2d.total_blobs

    def totalSlides(self):
        ''' Allows access to class vars without class declaration'''
        return Slide.total_slides


    def __str__(self):
        return str('Slide <Id:' + str(self.id_num) + ' Num of Blob2ds:' + str(len(self.blob2dlist)) + '>')

def getIdLists(pixels):
    '''
    Returns a list of lists, each of which corresponds to an id
    '''
    used_ids = set()
    for pixel in pixels:
        used_ids.add(pixel.blob_id)
    id_lists = [[] for i in range(len(used_ids))]
    id_to_index = dict()
    for index,id in enumerate(used_ids):
        id_to_index[id] = index
    unassigned_pixels = set(pixels)
    for pixel in unassigned_pixels:
        id_lists[id_to_index[pixel.blob_id]].append(pixel)
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
        if buf_nzn >= Config.minimal_nonzero_neighbors:
            filtered_pixels.append(pixel)
        else:
            removed_pixel_ids.append(pixel.id)
    if not quiet:
        print('There are ' + str(len(listin) - len(filtered_pixels)) + ' dead pixels & ' + str(len(filtered_pixels)) + ' still alive')
    return filtered_pixels


