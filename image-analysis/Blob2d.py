from myconfig import Config
import numpy as np
import math
from Pixel import Pixel
from util import warn, printl, printd


class Blob2d:
    """
    This class contains a list of pixels, which comprise a 2d blob on a single image
    """
    # equivalency_set = set() # Used to keep track of touching blobs, which can then be merged.
    total_blobs = 0  # Note that this is set AFTER merging is done, and so is not modified by blobs
    blobswithoutstitches = 0
    used_ids = []
    min_free_id = 0

    all = dict()  # A dictionary containing ALL Blob2ds. A blob2d's key is it's id

    def __init__(self, list_of_pixels, height, recursive_depth=0,
                 parent_id=-1):  # CHANGED to height from slide, removed master_array
        assert (recursive_depth == 0 or parent_id != -1)
        Blob2d.total_blobs += 1
        for pixel in list_of_pixels:
            pixel.validate()
        self.minx = min(pixel.x for pixel in list_of_pixels)
        self.maxx = max(pixel.x for pixel in list_of_pixels)
        self.miny = min(pixel.y for pixel in list_of_pixels)
        self.maxy = max(pixel.y for pixel in list_of_pixels)
        self.avgx = sum(pixel.x for pixel in list_of_pixels) / len(list_of_pixels)
        self.avgy = sum(pixel.y for pixel in list_of_pixels) / len(list_of_pixels)
        self.b3did = -1

        self.pixels = [pixel.id for pixel in list_of_pixels]
        self.assignedto3d = False  # True once a blod2d has been added to a list that will be used to construct a blob3d
        self.recursive_depth = recursive_depth
        self.parent_id = parent_id
        self.children = []
        self.height = height
        self.possible_partners = []  # A list of blobs which MAY be in the same blob3d, deleted later
        self.pairings = []  # A list of pairings that this blob belongs to
        self.edge_pixels = list()
        self.set_edge_pixels()
        self.id = -1
        self.context_bins = None  # Set later when stitching
        self.validate_id()  # self is added to Blob2d.all dict here

    @staticmethod
    def get(blob2d_id):
        return Blob2d.all.get(blob2d_id)

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
            res = res + Blob2d.all[child].getdescendants(rdepth=rdepth + 1)
        return res

    def getdirectdescendants(self, include_self=False):
        if include_self:
            res = []
        else:
            res = []
        return res + [Blob2d.get(b2d) for b2d in self.children]

    def getrelated(self, include_self=False):
        desc = self.getdescendants(include_self=include_self)
        par = self.getparents()
        return desc + par  # TODO This does not operate through branching. Not critical currently, but needs fixing or an modified alternative

    def getpartnerschain(self):
        return list(self.getpartnerschain_recur(set()))

    def getpartnerschain_recur(self, partnerset):
        old_set = partnerset.copy()
        partnerset.add(self.id)
        for p in self.possible_partners:
            if p not in partnerset:
                partnerset.update(Blob2d.all[p].getpartnerschain_recur(partnerset))
        return partnerset.difference(old_set)

    def getparents(self):  # Excludes self
        buf = self.getparentsrecur([])
        return buf

    def getparentsrecur(self, buf):  # Todo nest
        if self.parent_id != -1:  # Unassigned
            buf.append(Blob2d.all[self.parent_id])
            return Blob2d.all[self.parent_id].getparentsrecur(buf)
        else:
            return buf

    def set_edge_pixels(self):
        pixeldict = Pixel.pixel_ids_to_dict(self.pixels)
        self.edge_pixels = [pixel for pixel in self.pixels if len(
            Pixel.get(pixel).get_neighbors_from_dict(pixeldict)) < 8]

    def printdescendants(self, rdepth=0):
        pad = ''
        for _ in range(rdepth):
            pad += '-'
        printl(pad + str(self))
        for child in self.children:
            b2d = Blob2d.all[child]
            b2d.printdescendants(rdepth=rdepth + 1)

    def validate_id(self, quiet=True):
        """
        Checks that a blob2d's id has not been used, and updates it's id if it has been used
        It then adds the blob to the Blob2d master dictionary 'all'
        :param quiet:
        :return:
        """

        def get_next_id():
            index = Blob2d.min_free_id
            while index < len(Blob2d.used_ids) and Blob2d.used_ids[index] == 1:
                index += 1
            if index == len(Blob2d.used_ids):
                Blob2d.used_ids.append(
                    0)  # NOTE can alter this value, for now expanding by 50, which will be filled with zeros
            Blob2d.min_free_id = len(Blob2d.used_ids)
            return index

        if self.id >= len(Blob2d.used_ids):
            Blob2d.used_ids.resize(
                [self.id + 50])  # NOTE can alter this value, for now expanding by 50, which will be filled with zeros
            Blob2d.used_ids[
                self.id] = 1  # 1 for used, no need to check if the value has been used as we are in a new range
            Blob2d.all[self.id] = self
        elif self.id < 0 or Blob2d.used_ids[self.id] == 1:  # This id has already been used
            oldid = self.id
            self.id = get_next_id()
            if not quiet:
                printl('Updated id from ' + str(oldid) + ' to ' + str(self.id) + '  ' + str(self))
            Blob2d.all[self.id] = self
            Blob2d.used_ids[self.id] = 1
        else:  # Fill this id entry for the first time
            if not quiet:
                printl('Updated entry for ' + str(self.id))
            Blob2d.used_ids[self.id] = 1
            Blob2d.all[self.id] = self

    def set_possible_partners(self, blob2dlist):
        """
        Finds all blobs in the given slide that COULD overlap with the given blob.
        These blobs could be part of the same blob3D (partners)
        :param blob2dlist:
        """
        # A blob is a possible partner to another blob if they are in adjacent slides, and they overlap in area
        # Overlap cases (minx, maxx, miny, maxy at play)
        #  minx2 <= (minx1 | max1) <= maxx2
        #  miny2 <= (miny1 | maxy1) <= maxy2

        printd('Setting possible partners for b2d: ' + str(self) + ' from ' + str(len(blob2dlist)) + ' other blob2ds',
               Config.debug_partners)
        my_pixel_coor = set(
            [(Pixel.get(pix).x, Pixel.get(pix).y) for b2d in self.getdescendants(include_self=True) for pix in
             b2d.pixels])
        for b_num, blob in enumerate(blob2dlist):
            blob = Blob2d.get(blob)
            inbounds = False
            partner_smaller = False
            if (blob.minx <= self.minx <= blob.maxx) or (
                            blob.minx <= self.maxx <= blob.maxx):  # Covers the case where the blob on the above slide is larger
                # Overlaps in the x axis; a requirement even if overlapping in the y axis
                if (blob.miny <= self.miny <= blob.maxy) or (blob.miny <= self.maxy <= blob.maxy):
                    inbounds = True
                    partner_smaller = False
            if not inbounds:
                if (self.minx <= blob.minx <= self.maxx) or (self.minx <= blob.maxx <= self.maxx):
                    if (self.miny <= blob.miny <= self.maxy) or (self.miny <= blob.maxy <= self.maxy):
                        inbounds = True
                        partner_smaller = True
            # If either of the above was true, then one blob is within the bounding box of the other
            if inbounds:
                printd(' Found b2d: ' + str(blob) + ' to be in-bounds, so checking other conditions',
                       Config.debug_partners)

                pair_coor = set(
                    (Pixel.get(pix).x, Pixel.get(pix).y) for b2d in blob.getdescendants(include_self=True) for pix in
                    b2d.pixels)
                overlap_amount = len(my_pixel_coor) - len(my_pixel_coor - pair_coor)

                if len(pair_coor) and len(my_pixel_coor) and ((overlap_amount / len(
                        my_pixel_coor) > Config.minimal_pixel_overlap_to_be_possible_partners and len(
                    my_pixel_coor) > 7)
                                                              or ((overlap_amount / len(
                        pair_coor) > Config.minimal_pixel_overlap_to_be_possible_partners) and len(
                        pair_coor) > 7)):  # HACK
                    self.possible_partners.append(blob.id)
                    printd('  Above b2d confirmed to be partner, updated pp: ' + str(self.possible_partners),
                           Config.debug_partners)

                    if partner_smaller:
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

    def set_shape_contexts(self, num_bins):
        """
        Uses the methods described here: https://www.cs.berkeley.edu/~malik/papers/BMP-shape.pdf
        to set a shape context histogram (with num_bins), for each edge pixel in the blob.
        Note that only the edge_pixels are used to determine the shape context,
        and that only edge points derive a context.

        num_bins is the number of bins in the histogram for each point
        :param num_bins:
        """
        # Note making the reference point for each pixel itself
        # Note that angles are NORMALLY measured COUNTER-clockwise from the +x axis,
        # Note  however the += 180, used to remove the negative values,
        # NOTE  makes it so that angles are counterclockwise from the NEGATIVE x-axis
        edgep = len(self.edge_pixels)
        self.context_bins = np.zeros((edgep, num_bins))  # Each edge pixel has rows of num_bins each
        # First bin is 0 - (360 / num_bins) degress
        edge_pixels = list(Pixel.get(pixel) for pixel in self.edge_pixels)
        for (pix_num, pixel) in enumerate(edge_pixels):
            for (pix_num2, pixel2) in enumerate(edge_pixels):
                if pix_num != pix_num2:  # Only check against other pixels.
                    distance = math.sqrt(math.pow(pixel.x - pixel2.x, 2) + math.pow(pixel.y - pixel2.y, 2))
                    angle = math.degrees(
                        math.atan2(pixel2.y - pixel.y, pixel2.x - pixel.x))  # Note using atan2 handles the dy = 0 case
                    angle += 180
                    if not 0 <= angle <= 360:
                        printl('\n\n\n--ERROR: Angle=' + str(angle))
                    # Now need bin # and magnitude for histogram
                    bin_num = math.floor((angle / 360.) * (num_bins - 1))  # HACK PSOE from -1
                    value = math.log(distance, 10)
                    self.context_bins[pix_num][bin_num] += value

    def __str__(self):
        pairingidsl = [pairing.lowerblob for pairing in self.pairings if pairing.lowerblob != self.id]
        pairingidsu = [pairing.upperblob for pairing in self.pairings if pairing.upperblob != self.id]
        pairingids = sorted(pairingidsl + pairingidsu)
        return str('B2D{id:' + str(self.id) + ', #P=' + str(len(self.pixels))) + ', #EP=' + str(
            len(self.edge_pixels)) + ', recur_depth=' + str(self.recursive_depth) + ', parent_id=' + str(
            self.parent_id) + ', b3did=' + str(self.b3did) + ', pairedids=' + str(pairingids) + ', height=' + str(
            self.height) + ', (xl,xh,yl,yh)range:(' + str(self.minx) + ',' + str(self.maxx) + ',' + str(
            self.miny) + ',' + str(
            self.maxy) + '), Avg(X,Y):(%.1f' % self.avgx + ',%.1f' % self.avgy + ', children=' + str(
            self.children) + ')}'

    __repr__ = __str__

    def gen_internal_blob2ds(self, depth=0):
        livepix = set(set(self.pixels) - set(self.edge_pixels))
        last_edge = set(self.edge_pixels)
        alldict = Pixel.pixel_ids_to_dict(livepix)
        edge_neighbors = set()
        for pixel in last_edge:
            edge_neighbors = edge_neighbors | set(
                Pixel.get(pixel).get_neighbors_from_dict(alldict))  # - set(blob2d.edge_pixels)
        edge_neighbors = edge_neighbors - last_edge
        bloomstage = livepix
        livepix = livepix - edge_neighbors

        b2ds = Blob2d.pixels_to_blob2ds(bloomstage, parent_id=self.id,
                                        recursive_depth=self.recursive_depth + 1)  # NOTE making new pixels, rather than messing with existing

        for num, b2d in enumerate(b2ds):
            b2d = Blob2d.get(b2d)
            Blob2d.all[self.id].pixels = list(set(Blob2d.all[self.id].pixels) - set(b2d.pixels))

        if len(self.pixels) < len(Blob2d.get(self.id).pixels):
            warn('Gained pixels!!!! (THIS SHOULD NEVER HAPPEN!)')

        if depth < Config.max_depth:
            if len(livepix) > 1:
                for b2d in b2ds:
                    Blob2d.get(b2d).gen_internal_blob2ds(depth=depth + 1)

    def get_stitched_partners(self):
        """
        Recursively finds all blobs that are directly or indirectly connected to this blob via stitching
        :return: The list of all blobs that are connected to this blob, including the seed blob
            OR [] if this blob has already been formed into a chain, and cannot be used as a seed.
        """

        # TODO update this documentation
        def followstitches(cursorblob, blob2dlist):
            """
            Recursive support function for get_stitched_partners
            :param cursorblob:
            :param blob2dlist:
            :param: cursorblob: The blob whose stitching is examined for connected blob2ds
            :param: blob2dlist: The accumulated list of a blob2ds which are connected directly or indirectly to the inital seed blob
            """
            if type(cursorblob) is int:
                cursorblob = Blob2d.get(cursorblob)
            if hasattr(cursorblob, 'pairings') and len(cursorblob.pairings) != 0:
                if cursorblob not in blob2dlist:
                    if hasattr(cursorblob, 'assignedto3d') and cursorblob.assignedto3d:
                        printl('====> DB Warning, adding a blob to list that has already been assigned: ' + str(
                            cursorblob))
                    cursorblob.assignedto3d = True
                    blob2dlist.append(cursorblob)
                    for pairing in cursorblob.pairings:
                        for blob in (pairing.lowerblob, pairing.upperblob):
                            followstitches(blob, blob2dlist)
            else:
                Blob2d.blobswithoutstitches += 1

        if hasattr(self,
                   'assignedto3d') and self.assignedto3d is True:  # hasattr included to deal with outdate pickle files
            # Has already been assigned to a blob3d group, so no need to use as a seed
            return []
        b2ds = []
        followstitches(self, b2ds)
        return b2ds

    @staticmethod
    def mergeblobs(bloblist):
        """
        Returns a NEW list of blobs, which have been merged after having their ids updated (externally, beforehand)
        Use the global variable 'debug_set_merge' to control output
        :param bloblist:
        """
        newlist = []
        copylist = list(bloblist)  # http://stackoverflow.com/questions/2612802/how-to-clone-or-copy-a-list-in-python
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
                        printl('WARNING merging two blobs of different recursive depths:' + str(blob1) + ' & ' + str(
                            blob2))
                    merged = True
                    newpixels = newpixels + Blob2d.get(blob2).pixels
            if not merged:
                printd('--Never merged on blob:' + str(blob1), Config.debug_set_merge)
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
                newlist.append(Blob2d(Blob2d.get(blob1).pixels + newpixels,
                                      Blob2d.get(blob1).height,
                                      recursive_depth=Blob2d.get(blob1).recursive_depth,
                                      parent_id=min(Blob2d.get(blob1).parentID, Blob2d.get(blob2).parentID)))
                printd(' Merging, newlist-post:' + str(newlist), Config.debug_set_merge)
                printd(' Merging, copylist-post:' + str(copylist), Config.debug_set_merge)
        printd('Merge result' + str(newlist), Config.debug_set_merge)
        return newlist

    def edge_to_array(self, offset=True, buffer=0):
        """
        Creates an array representing the edge of a Blob2d. Each index is the value of the represented pixel.
        :param offset: Will mostly want this as True, to avoid a huge array
        :param buffer: Number of pixels to leave around the outside, good when operating on image (for border)
        :return:
        """
        if offset:
            offsetx = self.minx - buffer
            offsety = self.miny - buffer
        else:
            offsetx = 0
            offsety = 0
        arr = np.zeros((self.maxx - self.minx + 1 + buffer + 1, (self.maxy - self.miny + 1) + buffer + 1))
        for pixel in self.edge_pixels:
            arr[pixel.x - offsetx][pixel.y - offsety] = pixel.val
        return arr

    def body_to_array(self, offset=True, buffer=0):
        """
        Creates an array representing the body (including the edge) of a Blob2d.
        Each index is the value of the represented pixel.
        :param offset: Will mostly want this as True, to avoid a huge array
        :param buffer: Number of pixels to leave around the outside, good when operating on image (for border)
        :return:
        """
        if offset:
            offsetx = self.minx - buffer
            offsety = self.miny - buffer
        else:
            offsetx = 0
            offsety = 0
        arr = np.zeros((self.maxx - self.minx + 1 + buffer + 1, self.maxy - self.miny + 1 + buffer + 1))
        for pixel in self.pixels:
            arr[pixel.x - offsetx][pixel.y - offsety] = pixel.val
        return arr

    def save_image(self, filename):
        from scipy import misc as scipy_misc
        array_rep = self.edge_to_array(buffer=0)
        img = scipy_misc.toimage(array_rep, cmin=0.0, cmax=255.0)
        savename = Config.FIGURES_DIR + filename
        printl('Saving Image of Blob2d as: ' + str(savename))
        img.save(savename)

    def gen_saturated_array(self):
        """
        :return: Tuple(Array with all pixels outside of this blob2d's edge_pixels saturated, xoffset, yoffset)
        """
        body_arr = self.body_to_array()
        height, width = body_arr.shape

        xy_sat = [(x, y) for x in range(width) for y in range(height)
                  if body_arr[y][x] == 0]
        saturated = self.edge_to_array()
        for x, y in xy_sat:
            saturated[y][x] = Config.hard_max_pixel_value
        # At this stage, the entire array is reversed, so will invert the value (not an array inv)
        saturated = abs(saturated - Config.hard_max_pixel_value)
        return saturated

    @staticmethod
    def pixels_to_blob2ds(pixellist, parent_id=-1, recursive_depth=0):
        alonepixels = []
        alive = set(pixellist)
        blob2dlists = []
        while len(alive):
            alivedict = Pixel.pixel_ids_to_dict(alive)
            pixel = next(iter(alive))  # Basically alive[0]
            neighbors = set(Pixel.get(pixel).get_neighbors_from_dict(alivedict))
            index = 1
            done = False
            while (len(neighbors) == 0 or not done) and len(alive) > 0:
                if index < len(alive):
                    try:
                        pixel = list(alive)[
                            index]  # Basically alive[0] # TODO fix this to get the index set to the next iteration
                        index += 1
                    except Exception as exc:
                        printl('Error encountered: ' + str(exc))
                        printl('Index:' + str(index))
                        printl('Length of alive:' + str(len(alive)))
                        import pbt
                        pbt.set_trace()
                else:
                    done = True
                    # Assuming that all the remaining pixels are their own blob2ds essentially, and so are removed
                neighbors = set(Pixel.get(pixel).get_neighbors_from_dict(alivedict))
                if len(neighbors) == 0:
                    alive = alive - {pixel}
                    alonepixels.append(pixel)
                    index -= 1  # Incase we damaged the index
                    if index < 0:  # HACK
                        index = 0
            oldneighbors = set()  # TODO can make this more efficient
            while len(oldneighbors) != len(neighbors):
                oldneighbors = set(neighbors)
                newneighbors = set(neighbors)
                for pixel in neighbors:
                    newneighbors = newneighbors | set(pixel.get_neighbors_from_dict(alivedict))
                neighbors = newneighbors
            blob2dlists.append(list(neighbors))
            alive = alive - set(n.id for n in neighbors)

        b2ds = [Blob2d(blob2dlist, blob2dlist[0].z, parent_id=parent_id, recursive_depth=recursive_depth) for blob2dlist
                in blob2dlists if len(blob2dlist) > 0]

        # TODO this update is very expensive, need to separate this lists of children from the blob2ds (into another dict), therefore no need for a deep copy of a blob2d

        Blob2d.all[parent_id].children = Blob2d.all[parent_id].children + [b2d.id for b2d in b2ds]

        if Blob2d.get(parent_id).recursive_depth > 0:
            Blob2d.all[parent_id].pixels += [pixel for b2d in b2ds for pixel in b2d.pixels]
        b2ds = [b2d.id for b2d in b2ds]
        return b2ds
