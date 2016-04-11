from Blob2d import Blob2d
from Pixel import Pixel
from util import warn
from util import printl, printd
from myconfig import Config
import numpy as np
import math

def get_blob2ds_b3ds(blob2dlist, ids=False):
    """
    Gives the list of b3ds that together contain all b2ds in the supplied list
    :param blob2dlist: A list of blob2ds (not blob2d ids)
    :param ids: If true, returns B3d ids, else returns B3ds
    :return: A list of B3d-ids or B3ds
    """
    if ids:
        return list(set(b2d.b3did for b2d in blob2dlist if b2d.b3did != -1))
    else:
        return list(set(
            Blob3d.get(b2d.b3did) for b2d in blob2dlist if b2d.b3did != -1))  # Excluding b2ds that dont belong to a b3d


class Blob3d:
    """
    A group of blob2ds that chain together with pairings into a 3d shape
    Setting subblob=True indicates that this is a blob created from a pre-existing blob3d.
    """
    next_id = 0

    possible_merges = []  # Format: b3did1, b3did2, b2did (the one that links them!
    all = dict()

    def __init__(self, blob2dlist, r_depth=0):

        self.id = Blob3d.next_id
        Blob3d.next_id += 1
        self.blob2ds = blob2dlist  # List of the blob 2ds used to create this blob3d
        # Now find my pairings
        self.pairings = []
        self.lowslideheight = min(Blob2d.get(blob).height for blob in self.blob2ds)
        self.highslideheight = max(Blob2d.get(blob).height for blob in self.blob2ds)
        self.recursive_depth = r_depth
        self.children = []
        self.parent_id = None
        self.isBead = None

        ids_that_are_removed_due_to_reusal = set()
        for blobid in self.blob2ds:
            blob = Blob2d.get(blobid)
            if Blob2d.all[blob.id].b3did != -1:  # DEBUG #FIXME THE ISSUES COME BACK TO THIS, find the source
                # warn('NOT assigning a new b3did (' + str(self.id) + ') to blob2d: ' + str(Blob2d.all[blob.id]))
                printl('---NOT assigning a new b3did (' + str(self.id) + ') to blob2d: ' + str(Blob2d.all[blob.id]))
                Blob3d.possible_merges.append((Blob2d.all[blob.id].b3did, self.id, blob.id))
                ids_that_are_removed_due_to_reusal.add(blobid)
            else:  # Note not adding to the new b3d
                Blob2d.all[blob.id].b3did = self.id
                for stitch in blob.pairings:
                    if stitch not in self.pairings:  # TODO set will be faster
                        self.pairings.append(stitch)
        self.blob2d = list(set(self.blob2ds) - ids_that_are_removed_due_to_reusal)
        self.maxx = max(Blob2d.get(blob).maxx for blob in self.blob2ds)
        self.maxy = max(Blob2d.get(blob).maxy for blob in self.blob2ds)
        self.miny = min(Blob2d.get(blob).miny for blob in self.blob2ds)
        self.minx = min(Blob2d.get(blob).minx for blob in self.blob2ds)
        self.avgx = sum(Blob2d.get(blob).avgx for blob in self.blob2ds) / len(self.blob2ds)
        self.avgy = sum(Blob2d.get(blob).avgy for blob in self.blob2ds) / len(self.blob2ds)
        self.avgz = (self.lowslideheight + self.highslideheight) / 2
        self.isSingular = False
        self.subblobs = []
        self.note = ''  # This is a note that can be manually added for identifying certain characteristics..
        if r_depth != 0:
            all_b2d_parents = [Blob2d.get(Blob2d.get(b2d).parent_id) for b2d in blob2dlist]
            # printl('All b2d_parents of our b2ds that are going into a new b3d: ' + str(all_b2d_parents))
            parent_b3dids = set([b2d.b3did for b2d in all_b2d_parents if b2d.b3did != -1])
            # printl('Their b3dids: ' + str(parent_b3dids))
            if len(parent_b3dids) > 0:
                printd('Attempting to create a new b3d with id: ' + str(self.id)
                       + '\nAll b2d_parents of our b2ds that are going into a new b3d: ' + str(all_b2d_parents)
                       + '\nAll of the b2ds\'_parents\' b3dids: ' + str(parent_b3dids), Config.debug_b3d_merge)

                if len(parent_b3dids) > 1:
                    printd('*Found more than one b3d parent for b3d: ' + str(self) + ', attempting to merge parents: '
                           + str(list(Blob3d.get(b3d) for b3d in parent_b3dids)), Config.debug_b3d_merge)
                    Blob3d.merge(list(parent_b3dids))
                    new_parent_b3dids = list(set([b2d.b3did for b2d in all_b2d_parents if
                                                  b2d.b3did != -1]))  # TODO can remove this, just for safety for now
                    printd('  Post merging b3d parents, updated available-parent b3dids: ' + str(new_parent_b3dids),
                           Config.debug_b3d_merge)
                else:
                    new_parent_b3dids = list(parent_b3dids)
                self.parent_id = new_parent_b3dids[0]  # HACK HACK HACK
                if len(new_parent_b3dids) != 0 or self.parent_id == -1:
                    printd(" Updating b3d " + str(self.id) + '\'s parent_id to: ' + str(self.parent_id)
                           + ' from new_parent_ids(after regen after merge): ' + str(
                        list(Blob3d.getb3d(b3d)) for b3d in new_parent_b3dids), Config.debug_b3d_merge)
                Blob3d.all[self.parent_id].children.append(self.id)
                printd(' Added b3d ' + str(self.id) + ' to parent\'s list of children, updated parent: ' + str(
                    Blob3d.all[self.parent_id]), Config.debug_b3d_merge)
                if len(new_parent_b3dids) != 1:
                    warn('New b3d (' + str(self.id) + ') should have ended up with more than one parent!')
            else:
                warn('Creating a b3d at depth ' + str(r_depth) + ' with id ' + str(
                    self.id) + ' which could not find a b3d parent')
        self.validate()

    @staticmethod
    def merge(b1, b2):
        """
        Merges two blob3ds, and updates the entires of all data structures that link to these b3ds
        The chosen id to merge to is the smaller of the two available
        Returns the new merged blob3d in addition to updating its entry in Blob3d.all
        :param b1: The first b3d to merge
        :param b2: The second b3d to merge
        :return:
        """
        b1 = Blob3d.get(b1)
        b2 = Blob3d.get(b2)
        if b1.id < b2.id:
            smaller = b1
            larger = b2
        else:
            smaller = b2
            larger = b1
        for blob2d in larger.blob2ds:
            Blob2d.all[blob2d].b3did = smaller.id
            smaller.blob2ds.append(blob2d)
        smaller.blob2ds = list(set(smaller.blob2ds))
        del Blob3d.all[larger.id]
        return smaller

    @staticmethod
    def merge(b3dlist):
        printd('Called merge on b3dlist: ' + str(b3dlist), Config.debug_b3d_merge)
        res = b3dlist.pop()
        while len(b3dlist):
            cur = b3dlist.pop()
            res = Blob3d.merge2(res, cur)
        printd(' Final result of calling merge on b3dlist is b3d: ' + str(res), Config.debug_b3d_merge)
        return res

    @staticmethod
    def merge2(b1, b2):
        """
        Merges two blob3ds, and updates the entires of all data structures that link to these b3ds
        The chosen id to merge2 to is the smaller of the two available
        Returns the new merged blob3d in addition to updating its entry in Blob3d.all
        :param b1: The first b3d to merge2
        :param b2: The second b3d to merge2
        :return:
        """

        if b1 == -1 or b2 == -1:
            warn('***Skipping merging b3ds' + str(b1) + ' and ' + str(
                b2) + ' because at least one of them is -1, this should be fixed soon..')  # TODO
        else:
            b1 = Blob3d.get(b1)
            b2 = Blob3d.get(b2)
            printd(' Merging two b3ds: ' + str(b1) + '   ' + str(b2), Config.debug_b3d_merge)

            if b1.id < b2.id:  # HACK TODO revert this once issue is solved. This just makes things simpler to DEBUG
                smaller = b1
                larger = b2
            else:
                smaller = b2
                larger = b1

            for blob2d in larger.blob2ds:
                Blob2d.all[blob2d].b3did = smaller.id
                Blob3d.all[smaller.id].blob2ds.append(blob2d)

            # smaller.children += larger.children # CHANGED DEBUG
            Blob3d.all[smaller.id].children += larger.children

            if larger.parent_id is not None:
                Blob3d.all[larger.parent_id].children.remove(larger.id)
                if smaller.id not in Blob3d.all[larger.parent_id].children:  # Would occur if they have the same parent
                    Blob3d.all[larger.parent_id].children.append(smaller.id)

            for child in larger.children:
                Blob3d.all[child].parent_id = smaller.id

            if smaller.parent_id is not None:
                printd('  After Merging, the parent of the original smaller is: ' + str(Blob3d.get(smaller.parent_id)),
                       Config.debug_b3d_merge)
            if larger.parent_id is not None:
                printd('  After Merging, the parent of the original larger is: ' + str(Blob3d.get(larger.parent_id)),
                       Config.debug_b3d_merge)
            del Blob3d.all[larger.id]

            return smaller.id

    def validate(self):
        Blob3d.all[self.id] = self

    @staticmethod
    def get(blob3d_id):
        return Blob3d.all[blob3d_id]

    @staticmethod
    def at_depth(depth, ids=True):
        if ids:
            return list(b3d.id for b3d in Blob3d.all.values() if b3d.recursive_depth == depth)
        else:
            return list(b3d for b3d in Blob3d.all.values() if b3d.recursive_depth == depth)

    def __str__(self):
        parent_str = ' , Parent B3d: ' + str(self.parent_id)
        child_str = ' , Children: ' + str(self.children)
        return str(
            'B3D(' + str(self.id) + '): #b2ds:' + str(len(self.blob2ds)) + ', r_depth:' + str(self.recursive_depth) +
            ', bead=' + str(self.isBead) + parent_str + child_str + ')')
        # ' lowslideheight=' + str(self.lowslideheight) + ' highslideheight=' + str(self.highslideheight) +
        # ' #edgepixels=' + str(len(self.edge_pixels)) + ' #pixels=' + str(len(self.pixels)) +
        # ' (xl,xh,yl,yh)range:(' + str(self.minx) + ',' + str(self.maxx) + ',' + str(self.miny) + ',' + str(self.maxy) + parent_str + child_str + ')')

    __repr__ = __str__

    def get_edge_pixel_count(self):
        edge = 0
        for b2d in self.blob2ds:
            edge += len(Blob2d.get(b2d).edge_pixels)
        return edge

    def get_edge_pixels(self):
        edge = []
        for b2d in self.blob2ds:
            b2d = Blob2d.get(b2d)
            edge = edge + [Pixel.get(pix) for pix in b2d.edge_pixels]
        return edge

    def get_pixels(self, ids=False):
        pixel_ids = []
        for b2d in self.blob2ds:
            b2d = Blob2d.get(b2d)
            b2d_descend = b2d.getdescendants(include_self=True)
            # printl("B2d: " + str(b2d) + ' ----- had descendants (incl self(' + str(len(b2d_descend)) + ') ------' + str(b2d_descend))
            for blob2d in b2d_descend:
                pixel_ids += blob2d.pixels
        if ids:
            return pixel_ids
        else:
            return [Pixel.get(pixel_id) for pixel_id in pixel_ids]

    @staticmethod
    def tag_blobs_singular(blob3dlist, quiet=False):
        singular_count = 0
        non_singular_count = 0
        for blob3d in blob3dlist:
            singular = True
            for blob2d_num, blob2d in enumerate(blob3d.blob2ds):
                if blob2d_num != 0 or blob2d_num != len(blob3d.blob2ds):  # Endcap exceptions due to texture
                    if len(blob3d.pairings) > 3:  # Note ideally if > 2 # FIXME strange..
                        singular = False
                        break
            blob3d.isSingular = singular
            # Temp:
            if singular:
                singular_count += 1
            else:
                non_singular_count += 1
        if not quiet:
            printl('There are ' + str(singular_count) + ' singular 3d-blobs and ' + str(
                non_singular_count) + ' non-singular 3d-blobs')

    @staticmethod
    def tag_all_beads():
        printd('Tagging bead blob3ds', Config.debug_bead_tagging)
        base_b3ds = Blob3d.at_depth(0, ids=False)
        printl(str(len(base_b3ds)) + ' / ' + str(len(Blob3d.all)) + ' blob3ds are at recursive_depth=0')

        # DEBUG
        num_base_with_children = len(list(b3d for b3d in base_b3ds if len(b3d.children)))
        printl(str(num_base_with_children) + ' / ' + str(len(base_b3ds)) + ' base b3ds have children!')

        for b3d in base_b3ds:
            b3d.check_bead()
        printd(' ' + str(len(base_b3ds)) + ' of the ' + str(len(base_b3ds)) + ' base b3ds were tagged as beads',
               Config.debug_bead_tagging)

        # clean up
        unset = sorted(list(b3d for b3d in Blob3d.all.values() if b3d.isBead is None),
                       key=lambda b3d: b3d.recursive_depth)  # Do by recursive depth
        if len(unset):
            printd('When tagging all beads, there were ' + str(
                len(unset)) + ' b3ds which could not be reached from base b3ds', Config.debug_bead_tagging)
            printd(' They are: ' + str(unset),
                   Config.debug_bead_tagging)  # Want this to always be zero, otherwise theres a tree problem
        for b3d in unset:
            b3d.check_bead()
        printl("Total number of beads = " + str(sum(b3d.isBead for b3d in Blob3d.all.values())) + ' / ' + str(
            len(Blob3d.all)))

    def check_bead(self, indent=1):
        prefix = ' ' * indent
        printd(prefix + 'Called check_bead on b3d: ' + str(self), Config.debug_bead_tagging)
        child_bead_count = 0
        for child in self.children:
            printd(prefix + 'Checking if child of ' + str(self) + ' is bead:', Config.debug_bead_tagging)
            child_is_bead = Blob3d.get(child).check_bead(indent=indent + 1)
            if child_is_bead:
                child_bead_count += 1
        printd(prefix + 'Number of direct children which are beads = ' + str(child_bead_count) + ' / ' + str(
            len(self.children)), Config.debug_bead_tagging)
        # printl('Calling check_bead, max_subbeads_to_be_a_bead = ' + str(max_subbeads_to_be_a_bead), end='')
        # printl(', max_pixels_to_be_a_bead = ' + str(max_pixels_to_be_a_bead) + ', child_bead_difference = ' + str(child_bead_difference))
        # if self.recursive_depth > 0:
        # DEBUG
        self.isBead = \
            ((child_bead_count < Config.max_subbeads_to_be_a_bead)
             and (self.get_edge_pixel_count() <= Config.max_pixels_to_be_a_bead)) \
            or (self.recursive_depth == 0 and len(self.children) == 0)
        # and (self.recursive_depth > 0) # <== This makes bead tagging greedy and merges otherwise correctly disconnected beads
        printd(prefix + ' set isBead = ' + str(self.isBead), Config.debug_bead_tagging)

        # DEBUG
        printd(prefix + ' ^ was decided as: (' + str(child_bead_count) + ' < ' + str(Config.max_subbeads_to_be_a_bead)
               + ' and ' + str(self.get_edge_pixel_count()) + ' <= ' + str(Config.max_pixels_to_be_a_bead) + ') OR ('
               + str(self.recursive_depth) + ' == 0 and ' + str(len(self.children)) + ' == 0)',
               Config.debug_bead_tagging)

        #  and  (child_bead_count > (len(self.children) - config.child_bead_difference))
        return self.isBead

    @staticmethod
    def clean_b3ds():
        """
        This is a dev method, used to clean up errors in b3ds. Use sparingly!
        :return:
        """
        printl('<< CLEANING B3DS >>')
        # printl("These are the b3ds that will need fixing!")
        set_isBead_after = False
        adjusted_b3d_minmax = 0
        for b3d in Blob3d.all.values():
            if not hasattr(b3d, 'isBead'):
                b3d.isBead = None
                set_isBead_after = True
            remove_children = []
            for child in b3d.children:
                if child not in Blob3d.all:
                    remove_children.append(child)
            if len(remove_children):
                for child in remove_children:
                    b3d.children.remove(child)
                printl(' While cleaning b3d:' + str(b3d) + ' had to remove children that no longer existed ' + str(
                    remove_children))
            if b3d.parent_id is None and b3d.recursive_depth != 0:
                printd(' Found b3d with None parent_id: ' + str(b3d), Config.debug_b3d_merge)
            elif b3d.parent_id is not None and b3d.parent_id not in Blob3d.all:
                printl(' While cleaning b3d:' + str(b3d) + ' had to set parent_id to None, because parent_id: ' + str(
                    b3d.parent_id) + ' is not a valid blob3d-id')
                b3d.parent_id = None

            # for b2d in b3d.blob2ds:
            #     b2d = Blob2d.get(b2d)
            #     if b2d.maxx > b3d.maxx or b2d.maxy > b3d.maxy or b2d.minx < b3d.minx or b2d.miny < b3d.miny:
            #         adjusted_b3d_minmax += 1
            #         Blob3d.all[b3d.id].maxx = max(b2d.maxx, b3d.maxx)
            #         Blob3d.all[b3d.id].maxy = max(b2d.maxy, b3d.maxy)
            #         Blob3d.all[b3d.id].minx = min(b2d.minx, b3d.minx)
            #         Blob3d.all[b3d.id].miny = min(b2d.miny, b3d.miny)


        if set_isBead_after:
            printl(' While cleaning b3ds, found b3ds without isBead attr, so setting isBead for all b3ds')
            Blob3d.tag_all_beads()
        if adjusted_b3d_minmax:
            warn("Had to adjust the ranges for a total of " + str(adjusted_b3d_minmax) + ' blob3ds because their b2ds were out of range') # FIXME

    def save2d(self, filename):
        """
        This saves the 2d area around a blob3d for all slides, so that it can be used for testing later
        :param filename: The base filename to save, will have numerical suffix
        :return:
        """
        from scipy import misc as scipy_misc
        slice_arrays = []
        for i in range(self.highslideheight - self.lowslideheight + 1):
            slice_arrays.append(np.zeros((self.maxx - self.minx + 1, self.maxy - self.miny + 1)))
        savename = Config.FIGURES_DIR + filename
        for b2d in self.blob2ds:
            for pixel in b2d.pixels:
                slice_arrays[pixel.z - self.lowslideheight][pixel.x - self.minx][pixel.y - self.miny] = pixel.val
        for slice_num, slice_arr in enumerate(slice_arrays):
            img = scipy_misc.toimage(slice_arr, cmin=0.0, cmax=255.0)
            printl('Saving Image of Blob2d as: ' + str(savename) + str(slice_num) + '.png')
            img.save(savename + str(slice_num) + '.png')

    def get_first_child_beads(self):
        # This is meant to be called for a non_bead
        # Will descent it's tree, finding all beads that aren't in anyway children of another bead
        beads = []
        # print("Called get_first_child_beads on: " + str(self))
        # print("Iterating though children: " + str(self.children))
        for child in self.children:
            child = Blob3d.get(child)
            # print(" Examining child: " + str(child))
            if child.isBead:
                beads.append(child)
            else:
                beads = beads + child.get_first_child_beads()
        return beads

    def has_parent_nonbead(self):
        res = False

        if self.parent_id is not None:
            res = res or Blob3d.get(self.parent_id).has_parent_nonbead()
        return res

    def gen_skeleton(self):
        # Begin by creating a 3d array, with each element either None or the id of the pixel
        # Then create a second 3d array, with the distances from each internal point to the closest edge point
            # Find internals by doing all pixels - edge_pixels

        print("CALLED GEN_SKELETON!!!!")

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')




        xdim = self.maxx - self.minx + 1
        ydim = self.maxy - self.miny + 1
        zdim = self.highslideheight - self.lowslideheight + 1
        minx = self.minx
        miny = self.miny
        minz = self.lowslideheight



        def pixel_to_pos(pixel):
            # Returns the (x,y,z) location of the pixel in any one of the 3d arrays
            return (pixel.x - minx, pixel.y - miny, pixel.z - minz)

        def distance_from_offset(x_offset, y_offset, z_offset):
            return math.sqrt(math.pow(x_offset, 2) + math.pow(y_offset, 2) + math.pow(z_offset, 2))

        def inBounds(x,y,z):
            if x >= 0 and y >= 0 and z >= 0 and x < xdim and y < ydim and z < zdim:
                return True
            return False


        def find_nearest_neighbor(x, y, z, arr, recur=1):
            """

            :param x: X coordinate within given arr
            :param y: Y coordinate within given arr
            :param z: Z coordinate within given arr
            :param arr: An array populated by the ids of pixels
            :param recur: How far outwards to extend the 'search cube'
            :return: (x_coor, y_coor, z_coor, distance, pixel_id)
            """
            print("Finding nearest neighbor of: " + str((x, y, z, recur)))

            possible_coordinates = [] # Contains coordinate tuples (x,y,z)
            # Because am using cube instead of spheres, need to remember all matches and then find the best

            # arr should

            # TODO restrict the ranges so that they don't overlap

            # X restricts Y and Z
            # Y restricts Z
            # Z restricts nothing

            for x_offset in [-recur, recur]:
                curx = x + x_offset
                for y_offset in range(-recur, recur + 1): # +1 to include endcap
                    cury = y + y_offset
                    for z_offset in range(-recur, recur + 1): # +1 to include endcap
                        curz = z + z_offset

                        if inBounds(curx, cury, curz) and not np.isnan(arr[curx][cury][curz]):
                            possible_coordinates.append((curx, cury, curz, distance_from_offset(x_offset, y_offset, z_offset), int(arr[curx][cury][curz])))
                                # x, y, z, distance, pixel_id

            print("Coordinates found: " + str(possible_coordinates))

            if len(possible_coordinates) == 0:
                print("----Making a recursive call!")
                return find_nearest_neighbor(x, y, z, arr, recur + 1)

            #TODO Y and Z


            else:
                # Find the closest coordinate
                possible_coordinates.sort(key=lambda x_y_z_dist_id: x_y_z_dist_id[3]) # Sort by distance
                print("SORTED POSSIBLE COORDINATES: " + str(possible_coordinates))
                return possible_coordinates[0]

        #TODO have an option to create a shell around the blob3d, by taking the highest and lowest levels, and making them all count temporarily as
        # edge pixels. This way, the effects of segmentation will be less dependent on the scan direction



        edge_pixels = self.get_edge_pixels() # Actual pixels not ids
        all_pixels = self.get_pixels()
        inner_pixels = [Pixel.get(cur_pixel) for cur_pixel in (set(pixel.id for pixel in all_pixels) - set(pixel.id for pixel in edge_pixels))]

        edge_array = np.empty((xdim, ydim, zdim))#, dtype=np.int)
        inner_array = np.empty((xdim, ydim, zdim))#, dtype=np.int)
        distances = np.empty((xdim, ydim, zdim), dtype=np.float)
        edge_array[:] = np.nan
        inner_array[:] = np.nan
        distances[:] = np.nan





        for pixel in edge_pixels:
            x, y, z = pixel_to_pos(pixel)
            edge_array[x][y][z] = pixel.id

        for pixel in inner_pixels:
            x, y, z = pixel_to_pos(pixel)
            inner_array[x][y][z] = pixel.id



        inner_pos = np.zeros([len(inner_pixels), 3])
        near_pos = np.zeros([len(inner_pixels), 3])
        line_endpoints = np.zeros([2 * len(inner_pixels), 3])


        for index, pixel in enumerate(inner_pixels):
            print("Pixel: " + str(pixel))
            x, y, z = pixel_to_pos(pixel)
            inner_pos[index] = x / xdim, y / ydim, z / zdim
            nn = find_nearest_neighbor(x, y, z, edge_array)
            near_pos[index] = (nn[0] / xdim, nn[1] / ydim, nn[2] / zdim)
            line_endpoints[2 * index] = (nn[0] / xdim, nn[1] / ydim, nn[2] / zdim)
            line_endpoints[2 * index + 1] = (x / xdim, y / ydim, z / zdim)


        import vispy.io
        import vispy.scene
        from vispy.scene import visuals
        from vispy.util import keys


        canvas = vispy.scene.SceneCanvas(size=(1200,1200), keys='interactive', show=True)
        view = canvas.central_widget.add_view()
        view.camera = vispy.scene.cameras.FlyCamera(parent=view.scene, fov=30, name='Fly')
        # view.camera = vispy.scene.cameras.TurntableCamera(fov=0, azimuth=80, parent=view.scene, distance=1,
        #                                                   elevation=-55, name='Turntable')
        # view.camera = vispy.scene.cameras.ArcballCamera(parent=view.scene, fov=50, distance=1,
        #                                                    name='Arcball')
        inner_markers = visuals.Markers()
        near_markers = visuals.Markers()
        lines = visuals.Line(method=Config.linemethod)
        lines.set_data(pos=line_endpoints, connect='segments', color='y')

        inner_markers.set_data(inner_pos, face_color='r', size=10)
        near_markers.set_data(near_pos, face_color='g', size=10)
        print("Inner pos: " + str(inner_pos))
        print("Near pos: " + str(near_pos))


        view.add(inner_markers)
        view.add(near_markers)
        view.add(lines)
        vispy.app.run()


        print('------------')

        # print("\n\nEP: " + str(len(edge_pixels)))
        # print("Pix: " + str(len(all_pixels)))
        # print("Inner pix: " + str(len(inner_pixels)))
        # print("Children: " + str(self.children))
        # for color_index, pix_list in enumerate([edge_pixels, inner_pixels]):
        #     cur_color = ['r', 'g', 'b'][color_index]
        #     xs = [0] * len(pix_list)
        #     ys = [0] * len(pix_list)
        #     zs = [0] * len(pix_list)
        #
        #     for index, pixel in enumerate(pix_list):
        #         # print(pixel)
        #         # print('  ' + str(pixel_to_pos(pixel)))
        #         x, y, z =  pixel_to_pos(pixel)
        #         edge_array[x][y][z] = pixel.id
        #         xs[index] = x
        #         ys[index] = y
        #         zs[index] = z
        #     ax.scatter(xs, ys, zs, c=cur_color)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()









