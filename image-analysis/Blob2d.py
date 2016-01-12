from myconfig import *
import numpy as np
import math
from Pixel import Pixel
import copy

if mayPlot:
    from scipy import misc as scipy_misc



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

    @staticmethod
    def get(id):
        return Blob2d.all.get(id)

    @staticmethod
    def getall():
        return Blob2d.all.values()

    @staticmethod
    def getkeys():
        return Blob2d.all.keys()

    def getdescendants(self, include_self=True, rdepth=0):

        if include_self or rdepth != 0:
            res = [self]
        else:
            res = []
        for child in self.children:
            res = res + Blob2d.all[child].getdescendants(rdepth=rdepth+1)
        return res

    def getrelated(self, rdepth=0):
        desc = self.getdescendants()
        par = self.getparents()
        return desc + par #TODO This does not operate through branching. Not critical currently, but needs fixing or an modified alternative


    def getparents(self): # Excludes self
        buf = self.getparentsrecur([])
        return buf

    def getparentsrecur(self, buf): # Todo nest
        if self.parentID != -1: #Unassigned
            buf.append(Blob2d.all[self.parentID])
            return Blob2d.all[self.parentID].getparentsrecur(buf)
        else:
            return buf



    def printdescendants(self, rdepth=0):
        pad = ''
        for _ in range(rdepth):
            pad += '-'
        print(pad + str(self))
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

            db = 1
        elif self.id < 0 or Blob2d.used_ids[self.id] == 1: # This id has already been used
            oldid = self.id
            self.id = getNextID()
            if not quiet:
                print('Updated id from ' + str(oldid) + ' to ' + str(self.id) + '  ' + str(self))
            Blob2d.all[self.id] = self
            Blob2d.used_ids[self.id] = 1
            db = 2
        else: # Fill this id entry for the first time
            if not quiet:
                print('Updated entry for ' + str(self.id))
            # if self.id == 0:
            #     print("DB UPDATING ID 0")
            Blob2d.used_ids[self.id] = 1
            Blob2d.all[self.id] = self
        # print('DB validated the id:' + str(self.id) + ' db value is:' + str(db))



    def __init__(self, list_of_pixels, height, offsetx=0, offsety=0, recursive_depth=0, parentID=-1): # CHANGED to height from slide, removed master_array
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

        self.pixels = [pixel.id for pixel in list_of_pixels]
        self.assignedto3d = False # Set to true once a blod2d has been added to a list that will be used to construct a blob3d
        self.offsetx = offsetx
        self.offsety = offsety
        self.recursive_depth = recursive_depth
        self.parentID = parentID
        self.children = []

        self.height = height
        self.possible_partners = [] # A list of blobs which MAY be part of the same blob3d as this blob2d
        self.partner_costs = [] # The minimal cost for the corresponding blob2d in possible_partners
        # self.partner_subpixels = [] # Each element is a list of pixels, corresponding to a subset of the edge pixels from the partner blob
                                    # The value of each element in the sublist for each partner is the index of the pixel from the corresponding partner
        # self.my_subpixels = []      # Set of own subpixels, with each list corresponding to a list from partner_subpixels
        self.pairings = [] # A list of pairings that this blob belongs to

        # Note for now, will use the highest non-zero-neighbor count pixel
        self.setEdge()
        # self.setTouchingBlobs()
        self.max_width = self.maxx-self.minx + 1 # Note +1 to include both endcaps
        self.max_height = self.maxy-self.miny + 1 # Note +1 to include both endcaps //TODO rename misleading compared to self.heigh
        self.id = -1
        self.validateID() # self is added to Blob2d.all dict here

    def setEdge(self):
        pixeldict = Pixel.pixelidstodict(self.pixels)
        # self.edge_pixels = [pixel for pixel in self.pixels if pixel.nz_neighbors < 8]
        self.edge_pixels = [pixel for pixel in self.pixels if len(Pixel.get(pixel).neighborsfromdict(pixeldict)) < 8]
        self.edge_pixels.sort()


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

    @staticmethod
    def updateid(oldid, newid):
        '''
        Update the blob's id and the id of all pixels in the blob
        Better this was, as we dont have to check each pixel's id, just each blobs (externally!) before chaging pixels
        '''

        print(' DB updating the id for:' + str(Blob2d.all[oldid]) + ' to new id' + str(newid))
        #TODO
        Blob2d.used_ids[oldid] = 0
        Blob2d.used_ids[newid] = 1
        buf = Blob2d.all[oldid]
        for pix in buf.pixels:
            pix.blob_id = newid
        buf.id = newid

        Blob2d.all[newid] = buf
        Blob2d.all[oldid] = None
        print(' DB updated to' + str(Blob2d.all[newid]))



        # self.id = newid
        # for pix in self.pixels:
        #     pix.blob_id = newid


    def setPossiblePartners(self, slide, **kwargs):
        '''
        Finds all blobs in the given slide that COULD overlap with the given blob.
        These blobs could be part of the same blob3D (partners)
        '''
        # A blob is a possible partner to another blob if they are in adjacent slides, and they overlap in area
        # Overlap cases (minx, maxx, miny, maxy at play)
        #  minx2 <= (minx1 | max1) <= maxx2
        #  miny2 <= (miny1 | maxy1) <= maxy2

        debugflag = kwargs.get('debugflag', -1)
        debug2ds = kwargs.get('debugforb2ds',[])

        debug_partners = [247,216,245,262,263]



        if debugflag == 1 and self.id in debug2ds:
            print('>Checking the pairings of a blob2d that is being debugged: ' + str(self))


        # print('DEBUG Checking blob for possible partners:' + str(self) + ' xrange: (' + str(self.minx) + ',' + str(self.maxx) + '), yrange: (' + str(self.miny) + ',' + str(self.maxy) + ')')
        for blob in slide.blob2dlist:
            # print('DEBUG  Comparing against blob:' + str(blob) + ' xrange: (' + str(blob.minx) + ',' + str(blob.maxx) + '), yrange: (' + str(blob.miny) + ',' + str(blob.maxy) + ')')
            if debugflag == 1 and blob.id in debug2ds:
                print('====Checking for a pairing with a blob2d that is being debugged: ' + str(blob))
            if debugflag == 1 and self.id in debug2ds:
                print(' checking against blob w/id:' + str(blob.id))

            debugging = debugflag == 1 and self.id in debug2ds and blob.id in debug2ds

            # Grabbing static entry:
            blob = Blob2d.get(blob)

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
            if debugging:
                print('Compared:')
                print(self)
                print(blob)
                print('>>InBounds=' + str(inBounds) + ', partnerSmaller=' + str(partnerSmaller))
                # from serodraw import plotBlob2ds
                # plotBlob2ds([self, blob], ids=True)
            if inBounds:
                self.possible_partners.append(blob.id)
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
                for p_num, pixel in enumerate(blob.edge_pixels):
                    pixel = Pixel.get(pixel)
                    if left_bound <= pixel.x <= right_bound and down_bound <= pixel.y <= up_bound:
                        partner_subpixel_indeces.append(p_num)
                for p_num, pixel in enumerate(self.edge_pixels):
                    pixel = Pixel.get(pixel)
                    if left_bound <= pixel.x <= right_bound and down_bound <= pixel.y <= up_bound:
                        my_subpixel_indeces.append(p_num)


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
            pixel = Pixel.get(pixel)
            for (pix_num2, pixel2) in enumerate(self.edge_pixels):
                pixel2 = Pixel.get(pixel2)
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
        pairingidsl = [pairing.lowerblob.id for pairing in self.pairings if pairing.lowerblob.id != self.id]
        pairingidsu = [pairing.upperblob.id for pairing in self.pairings if pairing.upperblob.id != self.id]
        pairingids = sorted(pairingidsl + pairingidsu)
        return str('B{id:' + str(self.id) + ', #P=' + str(len(self.pixels))) + ', #EP=' + str(len(self.edge_pixels)) + ', recur_depth=' + str(self.recursive_depth) + ', parentID=' + str(self.parentID) + ', pairedids=' + str(pairingids)  + ', height=' + str(self.height) + ', (xl,xh,yl,yh)range:(' + str(self.minx) + ',' + str(self.maxx) + ',' + str(self.miny) + ',' + str(self.maxy) +'), Avg(X,Y):(%.1f' % self.avgx + ',%.1f' % self.avgy + ', children=' + str(self.children) +')}'

    __repr__ = __str__

    def totalBlobs(self):
        '''
        This is currently being updated at the end of the slide_creation
        '''
        return Blob2d.total_blobs

    def updatePairings(self, stitches):
        # print('DB Updating pairings with pairings totally costing: ' + str(pairings.total_cost))
        self.pairings.append(stitches)

    def getconnectedblob2ds(self, debug=False):
        '''
        Recursively finds all blobs that are directly or indirectly connected to this blob via stitching
        :return: The list of all blobs that are connected to this blob, including the seed blob
            OR [] if this blob has already been formed into a chain, and cannot be used as a seed.
        '''
        # TODO update this documentation
        def followstitches(cursorblob, blob2dlist):
            '''
            Recursive support function for getconnectedblob2ds
            :param: cursorblob: The blob whose stitching is examined for connected blob2ds
            :param: blob2dlist: The accumulated list of a blob2ds which are connected directly or indirectly to the inital seed blob
            '''

            if hasattr(cursorblob, 'pairings') and len(cursorblob.pairings) != 0:
                if cursorblob not in blob2dlist:
                    if hasattr(cursorblob, 'assignedto3d') and cursorblob.assignedto3d:
                        print('====> DB Warning, adding a blob to list that has already been assigned: ' + str(cursorblob))
                    cursorblob.assignedto3d = True
                    blob2dlist.append(cursorblob)
                    for pairing in cursorblob.pairings:
                        for blob in (pairing.lowerblob, pairing.upperblob):
                            followstitches(blob, blob2dlist)
            else:
                 Blob2d.blobswithoutstitches += 1
        if hasattr(self, 'assignedto3d') and self.assignedto3d is True: # hasattr included to deal with outdate pickle files
            # Has already been assigned to a blob3d group, so no need to use as a seed
            return []
        blob2dlist = []
        followstitches(self, blob2dlist)
        return blob2dlist

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
                if blob2 == blob1:
                    if debug_set_merge:
                        print('   Found blobs to merge: ' + str(blob1) + ' & ' + str(blob2))
                    if Blob2d.get(blob1).recursive_depth != Blob2d.get(blob2).recursive_depth:
                        print('WARNING merging two blobs of different recursive depths:' + str(blob1) + ' & ' + str(blob2))
                    merged = True
                    newpixels = newpixels + Blob2d.get(blob2).pixels
            if merged == False:
                if debug_set_merge:
                    print('--Never merged on blob:' + str(blob1))
                newlist.append(blob1)
                del copylist[0]
            else:
                if debug_set_merge:
                    print(' Merging, newlist-pre:' + str(newlist))
                    print(' Merging, copylist-pre:' + str(copylist))
                index = 0
                while index < len(copylist):
                    if debug_set_merge:
                        print(' Checking to delete:' + str(copylist[index]))
                    if copylist[index] == blob1:
                        if debug_set_merge:
                            print('  Deleting:' + str(copylist[index]))
                        del copylist[index]
                        index -= 1
                    index += 1
                newlist.append(Blob2d(Blob2d.get(blob1).pixels + newpixels, Blob2d.get(blob1).master_array, Blob2d.get(blob1).slide, recursive_depth=Blob2d.get(blob1).recursive_depth, parentID=min(Blob2d.get(blob1).parentID, Blob2d.get(blob2).parentID), direct_children=Blob2d.get(blob1).children + Blob2d.get(blob2).children))
                if debug_set_merge:
                    print(' Merging, newlist-post:' + str(newlist))
                    print(' Merging, copylist-post:' + str(copylist))
        if debug_set_merge:
            print('Merge result' + str(newlist))
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
        # TODO FIXME PICKLE!!!! This below +1 has been fixed on 10/8, but the pickle files needs to be regen.
        arr = np.zeros((self.max_width + buffer + 1, self.max_height + buffer + 1))
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

        # TODO FIXME PICKLE!!!! This below +1 has been fixed on 10/8, but the pickle files needs to be regen.

        arr = np.zeros((self.max_width + buffer + 1, self.max_height + buffer + 1))
        for pixel in self.pixels:
            arr[pixel.x - offsetx][pixel.y - offsety] = pixel.val
        return arr

    def saveImage(self, filename, **kwargs):
        array_rep = self.edgeToArray(buffer=0)
        img = scipy_misc.toimage(array_rep, cmin=0.0, cmax=255.0)
        savename = FIGURES_DIR + filename
        print('Saving Image of Blob2d as: ' + str(savename))
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
            saturated[y][x] = hard_max_pixel_value
        # At this stage, the entire array is reversed, so will invert the value (not an array inv)
        saturated = abs(saturated - hard_max_pixel_value)
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
        # print('DB ORIGINALLY ALIVE IS:' + str(alive))
        while len(alive):
            # print('DB len of alive is;' + str(len(alive)))
            # print('    DB CALLING WITHIN pixels to blob2ds with args:' + str(alive))
            alivedict = Pixel.pixelidstodict(alive)
            pixel = next(iter(alive)) # Basically alive[0]
            neighbors = set(Pixel.get(pixel).neighborsfromdict(alivedict))
            index = 1
            done = False
            while (len(neighbors) == 0 or not done) and len(alive) > 0:
                # print('    Index:' + str(index) + ' len of neighbors:' + str(len(neighbors)) + ' len of alive:' + str(len(alive)))
                if index < len(alive):
                    try:
                        pixel = list(alive)[index] # Basically alive[0] # TODO fix this to get the index set to the next iteration
                        index += 1
                    except:
                        print('Error encountered')
                        print('Index:' + str(index))
                        print('Length of alive:' + str(len(alive)))
                        import pbt
                        pbt.set_trace()
                else:
                    done = True
                    # Note this needs testing!!!!
                    # Assuming that all the remaining pixels are their own blob2ds essentiall, and so are removed
                neighbors = set(Pixel.get(pixel).neighborsfromdict(alivedict))
                if len(neighbors) == 0:
                    # print('   Found a blob with no neighbors, removing')
                    alive = alive - set([pixel])
                    alonepixels.append(pixel)
                    index = index - 1 # Incase we damaged the index
                    if index < 0: # HACK
                        index = 0
            oldneighbors = set() # TODO can make this more efficient
            while len(oldneighbors) != len(neighbors):
                oldneighbors = set(neighbors)
                newneighbors = set(neighbors)
                # print(' Iterating through: ' + str(len(neighbors)) + ' neighbors to cursor pixel & its found neighbors')
                for pixel in neighbors:
                    newneighbors = newneighbors | set(pixel.neighborsfromdict(alivedict))
                neighbors = newneighbors
            # print(' DB found a group which make up a blob2d:' + str(neighbors) + ' \n  consisting of ' + str(len(neighbors)) + ' pixels')
            blob2dlists.append(list(neighbors))

            # print('DB NEIGHBORS:' + str(neighbors))
            # print('DB alive:' + str(alive))
            alive = alive - set(n.id for n in neighbors)
            # print('DB after update at end of loop, alive:' + str(alive))

        b2ds = [Blob2d(blob2dlist, blob2dlist[0].z, parentID=parentID, recursive_depth=recursive_depth) for blob2dlist in blob2dlists if len(blob2dlist) > 0]
        # TODO this update is very expensive, need to separate this lists of children from the blob2ds (into another dict), therefore no need for a deep copy of a blob2d
        buff = copy.deepcopy(Blob2d.get(parentID)) # Note confirmed this doesnt change Blob2d.all
        buff.children += [b2d.id for b2d in b2ds]
        Blob2d.all[parentID] = buff

        if Blob2d.get(parentID).recursive_depth > 0:
            # print('  BEFORE:' + str(Blob2d.get(parentID)))
            Blob2d.all[parentID].pixels += [pixel for b2d in b2ds for pixel in b2d.pixels]
            # print('  AFTER:' + str(Blob2d.get(parentID)))

        b2ds = [b2d.id for b2d in b2ds]

        return b2ds
