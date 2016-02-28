from myconfig import Config
import numpy as np
import math
from Pixel import Pixel
import copy
from util import warn, printl, printd


class Blob2d:
    '''
    This class contains a list of pixels, which comprise a 2d blob on a single image
    '''
    # equivalency_set = set() # Used to keep track of touching blobs, which can then be merged.
    total_blobs = 0 # Note that this is set AFTER merging is done, and so is not modified by blobs
    blobswithoutstitches = 0
    used_ids = []
    min_free_id = 0

    all = dict() # A dictionary containing ALL Blob2ds. A blob2d's key is it's id

    def __init__(self, list_of_pixels, height, recursive_depth=0, parentID=-1): # CHANGED to height from slide, removed master_array
        assert(recursive_depth == 0 or parentID != -1)
        Blob2d.total_blobs += 1
        for pixel in list_of_pixels:
            pixel.validate();
        self.minx = min(pixel.x for pixel in list_of_pixels)
        self.maxx = max(pixel.x for pixel in list_of_pixels)
        self.miny = min(pixel.y for pixel in list_of_pixels)
        self.maxy = max(pixel.y for pixel in list_of_pixels)
        self.avgx = sum(pixel.x for pixel in list_of_pixels) / len(list_of_pixels)
        self.avgy = sum(pixel.y for pixel in list_of_pixels) / len(list_of_pixels)
        self.b3did = -1

        self.pixels = [pixel.id for pixel in list_of_pixels]
        self.assignedto3d = False # Set to true once a blod2d has been added to a list that will be used to construct a blob3d
        self.recursive_depth = recursive_depth
        self.parentID = parentID
        self.children = []
        self.height = height
        self.possible_partners = [] # A list of blobs which MAY be part of the same blob3d as this blob2d , deleted later
            #Note may want to use this later
        self.pairings = [] # A list of pairings that this blob belongs to
        self.setEdge()
        self.id = -1
        self.validateID() # self is added to Blob2d.all dict here


    @staticmethod
    def get(id):
        return Blob2d.all.get(id)

    @staticmethod
    def getall():
        return Blob2d.all.values()

    @staticmethod
    def getkeys():
        return Blob2d.all.keys()

    def getdescendants(self, include_self=False, rdepth=0):

        if include_self or rdepth != 0:
            res = [self]
        else:
            res = []
        for child in self.children:
            res = res + Blob2d.all[child].getdescendants(rdepth=rdepth+1)
        return res

    def getdirectdescendants(self, include_self=False):
        if include_self:
            res = []
        else:
            res = []
        return res + [Blob2d.get(b2d) for b2d in self.children]
    def getrelated(self, rdepth=0, include_self=False):
        desc = self.getdescendants(include_self=include_self)
        par = self.getparents()
        return desc + par #TODO This does not operate through branching. Not critical currently, but needs fixing or an modified alternative

    def getpartnerschain(self):
        return list(self.getpartnerschain_recur(set()))

    def getpartnerschain_recur(self, partnerset):
        old_set = partnerset.copy()
        partnerset.add(self.id)
        for p in self.possible_partners:
            if p not in partnerset:
                partnerset.update(Blob2d.all[p].getpartnerschain_recur(partnerset))
        return partnerset.difference(old_set)


    def getparents(self): # Excludes self
        buf = self.getparentsrecur([])
        return buf

    def getparentsrecur(self, buf): # Todo nest
        if self.parentID != -1: #Unassigned
            buf.append(Blob2d.all[self.parentID])
            return Blob2d.all[self.parentID].getparentsrecur(buf)
        else:
            return buf

    def setEdge(self):
        pixeldict = Pixel.pixelidstodict(self.pixels)
        self.edge_pixels = [pixel for pixel in self.pixels if len(Pixel.get(pixel).neighborsfromdict(pixeldict)) < 8]

    def printdescendants(self, rdepth=0):
        pad = ''
        for _ in range(rdepth):
            pad += '-'
        printl(pad + str(self))
        for child in self.children:
            b2d = Blob2d.all[child]
            b2d.printdescendants(rdepth=rdepth+1)


    def validateID(self, quiet=True):
        '''
        Checks that a blob2d's id has not been used, and it's id if it has been used
        It then adds the blob to the Blob2d master dictionary 'all'
        :return:
        '''

        def getNextID():
            index = Blob2d.min_free_id
            while(index < len(Blob2d.used_ids) and Blob2d.used_ids[index] == 1):
                index += 1
            if index == len(Blob2d.used_ids):
                Blob2d.used_ids.append(0) # NOTE can alter this value, for now expanding by 50, which will be filled with zeros
            Blob2d.min_free_id = len(Blob2d.used_ids)
            return index

        if self.id >= len(Blob2d.used_ids):
            Blob2d.used_ids.resize([self.id + 50]) # NOTE can alter this value, for now expanding by 50, which will be filled with zeros
            Blob2d.used_ids[self.id] = 1 # 1 for used, no need to check if the value has been used as we are in a new range
            Blob2d.all[self.id] = self
        elif self.id < 0 or Blob2d.used_ids[self.id] == 1: # This id has already been used
            oldid = self.id
            self.id = getNextID()
            if not quiet:
                printl('Updated id from ' + str(oldid) + ' to ' + str(self.id) + '  ' + str(self))
            Blob2d.all[self.id] = self
            Blob2d.used_ids[self.id] = 1
        else: # Fill this id entry for the first time
            if not quiet:
                printl('Updated entry for ' + str(self.id))
            Blob2d.used_ids[self.id] = 1
            Blob2d.all[self.id] = self


    def setTouchingBlobs(self):
        '''
        Updates the equivalency set of the slide containing the given blob.
        This is used to combine blobs that are touching.
        '''
        self.touching_blobs = set()
        for pixel in self.edge_pixels:
            neighbors = pixel.getNeighbors(self.master_array) # TODO NEED TO FIX THIS FOR DICTS
            for curn in neighbors:
                if(curn != 0 and curn.blob_id != self.id):
                    self.touching_blobs.add(curn.blob_id) # Not added if already in the set.
                    # HACK TODO not sure why is a list here, perhaps from pickles???
                    self.slide.equivalency_set = set(self.slide.equivalency_set)


                    if (curn.blob_id < self.id):
                        self.slide.equivalency_set.add((curn.blob_id, self.id))
                    else:
                        self.slide.equivalency_set.add((self.id, curn.blob_id))


    def setPossiblePartners(self, blob2dlist):
        '''
        Finds all blobs in the given slide that COULD overlap with the given blob.
        These blobs could be part of the same blob3D (partners)
        '''
        # A blob is a possible partner to another blob if they are in adjacent slides, and they overlap in area
        # Overlap cases (minx, maxx, miny, maxy at play)
        #  minx2 <= (minx1 | max1) <= maxx2
        #  miny2 <= (miny1 | maxy1) <= maxy2

        my_pixel_coor = set([(Pixel.get(pix).x, Pixel.get(pix).y) for b2d in self.getdescendants(include_self=True) for pix in b2d.pixels])

        for b_num, blob in enumerate(blob2dlist):
            blob = Blob2d.get(blob)
            inBounds = False
            partnerSmaller = False
            # printl('Working on blob #' + str(b_num) + ' / ' + str(len(blob2dlist)) + ' = ' + str(blob))

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
                pair_coor = set((Pixel.get(pix).x, Pixel.get(pix).y) for b2d in blob.getdescendants(include_self=True) for pix in b2d.pixels)
                # printl('DEBUG running extra tests to narrow possible partners')
                # printl('Pair_coor:' + str(pair_coor))
                # printl('Len of my_pixel_coor: ' + str(len(my_pixel_coor)) + ' len of pair_coor: ' + str(len(pair_coor)))
                # printl('Len of difference:' + str(len(my_pixel_coor - pair_coor)))
                overlap_amount = len(my_pixel_coor) - len(my_pixel_coor - pair_coor)

                if len(pair_coor) and len(my_pixel_coor) and ((overlap_amount / len(my_pixel_coor) > Config.minimal_pixel_overlap_to_be_possible_partners  and len(my_pixel_coor) > 7)
                or ((overlap_amount / len(pair_coor) > Config.minimal_pixel_overlap_to_be_possible_partners) and len(pair_coor) > 7)): #HACK
                    # len(my_pixel_coor - pair_coor) != len(my_pixel_coor)): # Overlapping coordinates
                    self.possible_partners.append(blob.id)
                    Blob2d.get(self.id).possible_partners.append(blob.id)
                    if partnerSmaller:
                        # Use partner's (blob) midpoints, and expand a proportion of minx, maxx, miny, maxy
                        midx = blob.avgx
                        midy = blob.avgy
                        left_bound = midx - ((blob.avgx - blob.minx) * Config.overscan_coefficient)
                        right_bound = midx + ((blob.maxx - blob.avgx) * Config.overscan_coefficient)
                        down_bound = midy - ((blob.avgy - blob.miny) * Config.overscan_coefficient)
                        up_bound = midy + ((blob.maxy - blob.avgy) * Config.overscan_coefficient)
                    else:
                        # Use partner's (blob) midpoints, and expand a proportion of minx, maxx, miny, maxy
                        midx = self.avgx
                        midy = self.avgy
                        left_bound = midx - ((self.avgx - self.minx) * Config.overscan_coefficient)
                        right_bound = midx + ((self.maxx - self.avgx) * Config.overscan_coefficient)
                        down_bound = midy - ((self.avgy - self.miny) * Config.overscan_coefficient)
                        up_bound = midy + ((self.maxy - self.avgy) * Config.overscan_coefficient)
                    partner_subpixel_indeces = []
                    my_subpixel_indeces = []
                    for p_num, pixel in enumerate(blob.edge_pixels):
                        pixel = Pixel.get(pixel)
                        if left_bound <= pixel.x <= right_bound and down_bound <= pixel.y <= up_bound:
                            partner_subpixel_indeces.append(p_num)
                    for p_num, pixel in enumerate(self.edge_pixels):
                        pixel = Pixel.get(pixel)
                        if left_bound <= pixel.x <= right_bound and down_bound <= pixel.y <= up_bound:
                            my_subpixel_indeces.append(p_num)
                # else:# DEBUG
                #     printl('-> Avoided setting ' + str(self) + ' and ' + str(blob) + ' as possible partners')
                #     if len(my_pixel_coor) > 20 and len(pair_coor) > 20:
                #         from serodraw import plotBlob2ds
                #         plotBlob2ds([self] + [blob])
        # self.partner_costs = [0] * len(self.possible_partners) # Note: May want to use this later
        # Could this method to do better filtering, like checking if the blobs are within each other etc
        # TODO update entry in Blob2d...?


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
        edge_pixels = list(Pixel.get(pixel) for pixel in self.edge_pixels)
        for (pix_num, pixel) in enumerate(edge_pixels):
            # pixel = Pixel.get(pixel)
            for (pix_num2, pixel2) in enumerate(edge_pixels):
                # pixel2 = Pixel.get(pixel2)
                if pix_num != pix_num2: # Only check against other pixels.
                    distance = math.sqrt(math.pow(pixel.x - pixel2.x, 2) + math.pow(pixel.y - pixel2.y, 2))
                    angle = math.degrees(math.atan2(pixel2.y - pixel.y, pixel2.x - pixel.x)) # Note using atan2 handles the dy = 0 case
                    angle += 180
                    if not 0 <= angle <= 360:
                        printl('\n\n\n--ERROR: Angle=' + str(angle))
                    # Now need bin # and magnitude for histogram
                    bin_num = math.floor((angle / 360.) * (num_bins - 1)) # HACK PSOE from -1
                    value = math.log(distance, 10)
                    # printl('DB: Pixel:' + str(pixel) + ' Pixel2:' + str(pixel2) + ' distance:' + str(distance) + ' angle:' + str(angle) + ' bin_num:' + str(bin_num))
                    self.context_bins[pix_num][bin_num] += value


    def __str__(self):
        pairingidsl = [pairing.lowerblob for pairing in self.pairings if pairing.lowerblob != self.id]
        pairingidsu = [pairing.upperblob for pairing in self.pairings if pairing.upperblob != self.id]
        pairingids = sorted(pairingidsl + pairingidsu)
        return str('B{id:' + str(self.id) + ', #P=' + str(len(self.pixels))) + ', #EP=' + str(len(self.edge_pixels)) + ', recur_depth=' + str(self.recursive_depth) + ', parentID=' + str(self.parentID) + ', b3did=' + str(self.b3did) + ', pairedids=' + str(pairingids)  + ', height=' + str(self.height) + ', (xl,xh,yl,yh)range:(' + str(self.minx) + ',' + str(self.maxx) + ',' + str(self.miny) + ',' + str(self.maxy) +'), Avg(X,Y):(%.1f' % self.avgx + ',%.1f' % self.avgy + ', children=' + str(self.children) +')}'

    __repr__ = __str__

    def bloomInwards(self, depth=0):
        livepix = set(set(self.pixels) - set(self.edge_pixels))
        last_edge = set(self.edge_pixels)
        alldict = Pixel.pixelidstodict(livepix)
        edge_neighbors = set()
        for pixel in last_edge:
            edge_neighbors = edge_neighbors | set(Pixel.get(pixel).neighborsfromdict(alldict)) # - set(blob2d.edge_pixels)
        edge_neighbors = edge_neighbors - last_edge
        bloomstage = livepix
        livepix = livepix - edge_neighbors

        b2ds = Blob2d.pixels_to_blob2ds(bloomstage, parentID=self.id, recursive_depth=self.recursive_depth+1, modify=False) # NOTE making new pixels, rather than messing with existing

        for num,b2d in enumerate(b2ds):
            b2d = Blob2d.get(b2d)
            Blob2d.all[self.id].pixels = list(set(Blob2d.all[self.id].pixels) - set(b2d.pixels))

        # printl(depth_offset + ' After being bloomed the parentID is:' + str(Blob2d.get(blob2d.id)))
        if (len(self.pixels) < len(Blob2d.get(self.id).pixels)):
            warn('Gained pixels!!!! (THIS SHOULD NEVER HAPPEN!)')

        if depth < Config.max_depth:
            if len(livepix) > 1:
                for b2d in b2ds:
                    Blob2d.get(b2d).bloomInwards(depth=depth+1)

    def get_stitched_partners(self, debug=False):
        '''
        Recursively finds all blobs that are directly or indirectly connected to this blob via stitching
        :return: The list of all blobs that are connected to this blob, including the seed blob
            OR [] if this blob has already been formed into a chain, and cannot be used as a seed.
        '''
        # TODO update this documentation
        def followstitches(cursorblob, blob2dlist):
            '''
            Recursive support function for get_stitched_partners
            :param: cursorblob: The blob whose stitching is examined for connected blob2ds
            :param: blob2dlist: The accumulated list of a blob2ds which are connected directly or indirectly to the inital seed blob
            '''
            if hasattr(cursorblob, 'pairings') and len(cursorblob.pairings) != 0:
                if cursorblob not in blob2dlist:
                    if hasattr(cursorblob, 'assignedto3d') and cursorblob.assignedto3d:
                        printl('====> DB Warning, adding a blob to list that has already been assigned: ' + str(cursorblob))
                    cursorblob.assignedto3d = True
                    blob2dlist.append(cursorblob)
                    # printl('    DB going through pairings:')
                    for pairing in cursorblob.pairings:
                        # printl('     Cur pairing: ' + str(pairing))
                        for blob in (pairing.lowerblob, pairing.upperblob):
                            # printl('      Cur blob in pairing: ' + str(blob))
                            followstitches(blob, blob2dlist)
            else:
                 Blob2d.blobswithoutstitches += 1
        if hasattr(self, 'assignedto3d') and self.assignedto3d is True: # hasattr included to deal with outdate pickle files
            # Has already been assigned to a blob3d group, so no need to use as a seed
            return []
        blob2dlist = []
        followstitches(self, blob2dlist)
        #del self.possible_partners # TODO see if theres a safe way to do this later
        return blob2dlist

    @staticmethod
    def mergeblobs(bloblist):
        '''
        Returns a NEW list of blobs, which have been merged after having their ids updated (externally, beforehand)
        Use the global variable 'debug_set_merge' to control output
        '''
        newlist = []
        copylist = list(bloblist) # Hack, fix by iterating backwards: http://stackoverflow.com/questions/2612802/how-to-clone-or-copy-a-list-in-python
        printd('Blobs to merge:' + str(copylist), Config.debug_set_merge)
        while len(copylist) > 0:
            printd('Len of copylist:' + str(len(copylist)), Config.debug_set_merge)
            blob1 = copylist[0]
            newpixels = []
            merged = False
            printd('**Curblob:' + str(blob1), Config.debug_set_merge)
            for (index2, blob2) in enumerate(copylist[1:]):
                if blob2 == blob1:
                    printd('   Found blobs to merge: ' + str(blob1) + ' & ' + str(blob2), Config.debug_set_merge)
                    if Blob2d.get(blob1).recursive_depth != Blob2d.get(blob2).recursive_depth:
                        printl('WARNING merging two blobs of different recursive depths:' + str(blob1) + ' & ' + str(blob2))
                    merged = True
                    newpixels = newpixels + Blob2d.get(blob2).pixels
            if merged == False:
                printd('--Never merged on blob:' + str(blob1) ,Config.debug_set_merge)
                newlist.append(blob1)
                del copylist[0]
            else:
                printd(' Merging, newlist-pre:', Config.debug_set_merge)
                printd(' Merging, copylist-pre:', Config.debug_set_merge)
                index = 0
                while index < len(copylist):
                    printd(' Checking to delete:' + str(copylist[index]), Config.debug_set_merge)
                    if copylist[index] == blob1:
                        printd('  Deleting:' + str(copylist[index]), Config.debug_set_merge)
                        del copylist[index]
                        index -= 1
                    index += 1
                newlist.append(Blob2d(Blob2d.get(blob1).pixels + newpixels, Blob2d.get(blob1).master_array, Blob2d.get(blob1).slide, recursive_depth=Blob2d.get(blob1).recursive_depth, parentID=min(Blob2d.get(blob1).parentID, Blob2d.get(blob2).parentID), direct_children=Blob2d.get(blob1).children + Blob2d.get(blob2).children))
                printd(' Merging, newlist-post:' + str(newlist), Config.debug_set_merge)
                printd(' Merging, copylist-post:' + str(copylist), Config.debug_set_merge)
        printd('Merge result' + str(newlist), Config.debug_set_merge)
        return newlist

    def toArray(self):
        '''
        Converts a blob2d to an array (including its inner pixels)
        :return: (numpy_array, (x_offset, y_offset))
        '''
        # Note self already has minx, miny
        width = self.maxx - self.minx + 1 # HACK + 1
        height = self.maxy - self.miny + 1  # HACK + 1
        arr = np.zeros((width, height))
        for pixel in self.pixels:
            arr[pixel.x - self.minx][pixel.y - self.miny] = pixel.val
        return (arr, (self.minx, self.miny))

    def edgeToXY(self, **kwargs):
        '''
        Converts blob2d's edge pixels to two 1d-arrays, one each for the x & y coordinates
        :return: (Numpy-1D array of X-coordinates, Numpy-1D array of Y-coordinates)
        '''
        compensate_for_offset = kwargs.get('offset', False)
        if compensate_for_offset:
            offsetx = self.minx
            offsety = self.miny
        else:
            offsetx = 0
            offsety = 0
        x = np.zeros(len(self.edge_pixels))
        y = np.zeros(len(self.edge_pixels))
        for pix_num, pixel in enumerate(self.edge_pixels):
            x[pix_num] = pixel.x - offsetx
            y[pix_num] = pixel.y - offsety
        return (x,y)

    def edgeToArray(self, **kwargs):
        compensate_for_offset = kwargs.get('offset', True) # Will almost always want to do this, to avoid a huge array
        buffer = kwargs.get('buffer', 0) # Number of pixels to leave around the outside, good when operating on image

        if compensate_for_offset:
            offsetx = self.minx - buffer
            offsety = self.miny - buffer
        else:
            offsetx = 0
            offsety = 0
        arr = np.zeros((self.maxx-self.minx + 1 + buffer + 1, (self.maxy-self.miny + 1) + buffer + 1))
        for pixel in self.edge_pixels:
            arr[pixel.x - offsetx][pixel.y - offsety] = pixel.val
        return arr

    def bodyToArray(self, **kwargs):
        compensate_for_offset = kwargs.get('offset', True) # Will almost always want to do this, to avoid a huge array
        buffer = kwargs.get('buffer', 0) # Number of pixels to leave around the outside, good when operating on image

        if compensate_for_offset:
            offsetx = self.minx - buffer # HACK + 1 FIXME
            offsety = self.miny - buffer# HACK + 1 FIXME
        else:
            offsetx = 0
            offsety = 0
        arr = np.zeros((self.maxx-self.minx + 1 + buffer + 1, self.maxy-self.miny + 1 + buffer + 1))
        for pixel in self.pixels:
            arr[pixel.x - offsetx][pixel.y - offsety] = pixel.val
        return arr

    def saveImage(self, filename, **kwargs):
        from scipy import misc as scipy_misc
        array_rep = self.edgeToArray(buffer=0)
        img = scipy_misc.toimage(array_rep, cmin=0.0, cmax=255.0)
        savename = Config.FIGURES_DIR + filename
        printl('Saving Image of Blob2d as: ' + str(savename))
        img.save(savename)

    def gen_saturated_array(self):
        '''
        :return: Tuple(Array with all pixels outside of this blob2d's edge_pixels saturated, xoffset, yoffset)
        '''
        body_arr = self.bodyToArray()
        height, width = body_arr.shape

        xy_sat = [(x, y) for x in range(width) for y in range(height)
                  if body_arr[y][x] == 0] # HACK TODO
        saturated = self.edgeToArray()
        for x,y in xy_sat:
            saturated[y][x] = Config.hard_max_pixel_value
        # At this stage, the entire array is reversed, so will invert the value (not an array inv)
        saturated = abs(saturated - Config.hard_max_pixel_value)
        return saturated


    #TODO this needs some fixing, has been causing issues with pixels to dict, will know is fixed once can remove the id to b2d hack in pixelstodict
    @staticmethod
    def pixels_to_blob2ds(pixellist, parentID=-1, recursive_depth=0, modify=False): # Modify true if we want to modify the pixels passed, otherwise make new ones
        #HACK
        # if modify:
        #     pixelistcopy = pixellist
        # else:
        #     pixelistcopy = []# = pixellist #HACK
        #     for pixel in pixellist:
        #         pixel = Pixel.get(pixel)
        #         pixelistcopy.append(Pixel(pixel.val, pixel.x, pixel.y, pixel.z))

        #DEBUG
        pixellistcopy = pixellist
        #DEBUG

        alonepixels = []
        # for pixel in pixellistcopy:
        #     pixel.blob_id = -1
        #hack
        alive = set(pixellistcopy)
        blob2dlists = []
        # printl('DB ORIGINALLY ALIVE IS:' + str(alive))
        while len(alive):
            # printl('DB len of alive is;' + str(len(alive)))
            # printl('    DB CALLING WITHIN pixels to blob2ds with args:' + str(alive))
            alivedict = Pixel.pixelidstodict(alive)
            pixel = next(iter(alive)) # Basically alive[0]
            neighbors = set(Pixel.get(pixel).neighborsfromdict(alivedict))
            index = 1
            done = False
            while (len(neighbors) == 0 or not done) and len(alive) > 0:
                # printl('    Index:' + str(index) + ' len of neighbors:' + str(len(neighbors)) + ' len of alive:' + str(len(alive)))
                if index < len(alive):
                    try:
                        pixel = list(alive)[index] # Basically alive[0] # TODO fix this to get the index set to the next iteration
                        index += 1
                    except:
                        printl('Error encountered')
                        printl('Index:' + str(index))
                        printl('Length of alive:' + str(len(alive)))
                        import pbt
                        pbt.set_trace()
                else:
                    done = True
                    # Note this needs testing!!!!
                    # Assuming that all the remaining pixels are their own blob2ds essentiall, and so are removed
                neighbors = set(Pixel.get(pixel).neighborsfromdict(alivedict))
                if len(neighbors) == 0:
                    # printl('   Found a blob with no neighbors, removing')
                    alive = alive - set([pixel])
                    alonepixels.append(pixel)
                    index = index - 1 # Incase we damaged the index
                    if index < 0: # HACK
                        index = 0
            oldneighbors = set() # TODO can make this more efficient
            while len(oldneighbors) != len(neighbors):
                oldneighbors = set(neighbors)
                newneighbors = set(neighbors)
                # printl(' Iterating through: ' + str(len(neighbors)) + ' neighbors to cursor pixel & its found neighbors')
                for pixel in neighbors:
                    newneighbors = newneighbors | set(pixel.neighborsfromdict(alivedict))
                neighbors = newneighbors
            # printl(' DB found a group which make up a blob2d:' + str(neighbors) + ' \n  consisting of ' + str(len(neighbors)) + ' pixels')
            blob2dlists.append(list(neighbors))

            # printl('DB NEIGHBORS:' + str(neighbors))
            # printl('DB alive:' + str(alive))
            alive = alive - set(n.id for n in neighbors)
            # printl('DB after update at end of loop, alive:' + str(alive))

        b2ds = [Blob2d(blob2dlist, blob2dlist[0].z, parentID=parentID, recursive_depth=recursive_depth) for blob2dlist in blob2dlists if len(blob2dlist) > 0]

        # TODO this update is very expensive, need to separate this lists of children from the blob2ds (into another dict), therefore no need for a deep copy of a blob2d


        # DEBUG DEBUG

        # buff = copy.deepcopy(Blob2d.get(parentID)) # Note confirmed this doesnt change Blob2d.all
        # buff.children += [b2d.id for b2d in b2ds]
        # Blob2d.all[parentID] = buff
        # printl('=======> Updated to: ' + str(buff) )

        Blob2d.all[parentID].children = Blob2d.all[parentID].children + [b2d.id for b2d in b2ds]


        if Blob2d.get(parentID).recursive_depth > 0:
            # printl('  BEFORE:' + str(Blob2d.get(parentID)))
            Blob2d.all[parentID].pixels += [pixel for b2d in b2ds for pixel in b2d.pixels]
            # printl('  AFTER:' + str(Blob2d.get(parentID)))

        b2ds = [b2d.id for b2d in b2ds]

        return b2ds
