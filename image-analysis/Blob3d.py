from Blob2d import Blob2d
from Pixel import Pixel
from util import warn
from util import debug
from myconfig import config

def printGeneralInfo(prefix='', indent=0, suffix=''):
    prefix = (' ' * indent) + prefix
    print(prefix + '<Blob3d>: Count:' + str(len(Blob3d.all)) + suffix)
    print(prefix + '<Blob2d>: Count:' + str(len(Blob2d.all)) + suffix)

def getBlob2dOwners(blob2dlist, ids=False):
    '''
    Gives the list of b3ds that together contain all b2ds in the supplied list
    :param blob2dlist: A list of blob2ds (not blob2d ids)
    :param ids: If true, returns B3d ids, else returns B3ds
    :return: A list of B3d-ids or B3ds
    '''
    if ids:
        return list(set(b2d.b3did for b2d in blob2dlist if b2d.b3did != -1))
    else:
       return list(set(Blob3d.get(b2d.b3did) for b2d in blob2dlist if b2d.b3did != -1)) # Excluding b2ds that dont belong to a b3d



class Blob3d:
    '''
    A group of blob2ds that chain together with pairings into a 3d shape
    Setting subblob=True indicates that this is a blob created from a pre-existing blob3d.
    '''
    next_id = 0

    possible_merges = [] # Format: b3did1, b3did2, b2did (the one that links them!
    all = dict()

    def __init__(self, blob2dlist, r_depth=0):

        self.id = Blob3d.next_id
        Blob3d.next_id += 1
        self.blob2ds = blob2dlist          # List of the blob 2ds used to create this blob3d
        # Now find my pairings
        self.pairings = []
        self.lowslideheight = min(Blob2d.get(blob).height for blob in self.blob2ds)
        self.highslideheight = max(Blob2d.get(blob).height for blob in self.blob2ds)
        self.recursive_depth = r_depth
        self.children = []
        self.parentID = None
        self.isBead = None


        ids_that_are_removed_due_to_reusal = set()
        for blobid in self.blob2ds:
            blob = Blob2d.get(blobid)
            if Blob2d.all[blob.id].b3did != -1: # DEBUG #FIXME THE ISSUES COME BACK TO THIS, find the source
                # warn('NOT assigning a new b3did (' + str(self.id) + ') to blob2d: ' + str(Blob2d.all[blob.id]))
                print('---NOT assigning a new b3did (' + str(self.id) + ') to blob2d: ' + str(Blob2d.all[blob.id]))
                Blob3d.possible_merges.append((Blob2d.all[blob.id].b3did, self.id, blob.id))
                ids_that_are_removed_due_to_reusal.add(blobid)
            else: # Note not adding to the new b3d
                Blob2d.all[blob.id].b3did = self.id
                for stitch in blob.pairings:
                    if stitch not in self.pairings: # TODO set will be faster
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
        self.note = '' # This is a note that can be manually added for identifying certain characteristics..
        if r_depth != 0:
            all_b2d_parents = [Blob2d.get(Blob2d.get(b2d).parentID) for b2d in blob2dlist]
            # print('All b2d_parents of our b2ds that are going into a new b3d: ' + str(all_b2d_parents))
            parent_b3dids = set([b2d.b3did for b2d in all_b2d_parents])
            # print('Their b3dids: ' + str(parent_b3dids))
            if len(parent_b3dids) > 0:
                if len(parent_b3dids) > 1:
                    print(' Found more than one b3d parent for b3d: ' + str(self.id) + ' attempting to merge parents')
                    Blob3d.merge(list(parent_b3dids))
                    new_parent_b3dids = list(set([b2d.b3did for b2d in all_b2d_parents])) # TODO can remove this, just for safety for now
                    print('  Post merging b3d parents, updated parent b3dids: ' + str(new_parent_b3dids))
                else:
                    new_parent_b3dids = list(parent_b3dids)
                self.parentID = new_parent_b3dids[0] # HACK HACK HACK
                Blob3d.all[new_parent_b3dids[0]].children.append(self.id)
                # print('--> set parentID to: ' + str(self.parentID) + ' from the available parent_b3dids (after merging): ' + str(new_parent_b3dids))
                # print('Which has been updated to: ' + str(Blob3d.get(self.parentID)))
                if len(new_parent_b3dids) != 1:
                    warn('New b3d (' + str(self.id) + ') ended up with more than one parent!')
            else:
                warn('Creating a b3d at depth ' + str(r_depth) + ' with id ' + str(self.id) + ' which could not find a b3d parent')
        self.validate()


    @staticmethod
    def merge(b1, b2):
        '''
        Merges two blob3ds, and updates the entires of all data structures that link to these b3ds
        The chosen id to merge to is the smaller of the two available
        Returns the new merged blob3d in addition to updating its entry in Blob3d.all
        :param b1: The first b3d to merge
        :param b2: The second b3d to merge
        :return:
        '''
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
    def mergeall():
        '''
        This cleans up any blob3ds that attempted to obtain a blob2d that was already part of another blob3d
        Blob3d.possible_merges is generated as Blob3ds are
        Ideally this wouldn't be necessary but code happens
        :return:
        '''
        # Experimenting with merging blob3ds.
        if len(Blob3d.possible_merges):
            print('Before merging:--------------')
            printGeneralInfo()
            all_ids_to_merge = set(id for triple in Blob3d.possible_merges for id in [triple[0], triple[1]])
            merged_set_no = [-1] * (max(all_ids_to_merge) + 1)
            merges = []
            for b3d1, b3d2, b2d in Blob3d.possible_merges:
                if merged_set_no[b3d1] == -1: # not yet in a set
                    if merged_set_no[b3d2] == -1:
                        merges.append(set([b3d1, b3d2]))
                        merged_set_no[b3d1] = len(merges) - 1
                        merged_set_no[b3d2] = len(merges) - 1
                    else:
                        # b3d1 not yet in a set, b3d2 is
                        merges[merged_set_no[b3d2]].add(b3d1)
                        merged_set_no[b3d1] = merged_set_no[b3d2]
                else:
                    # b3d1 is in a set
                    if merged_set_no[b3d2] == -1:
                        # b3d2 not yet in a set, b3d1 is
                        merges[merged_set_no[b3d1]].add(b3d2)
                        merged_set_no[b3d2] = merged_set_no[b3d1]
                    else:
                        # Both are already in sets, THEY BETTER BE THE SAME!!!!
                        if merged_set_no[b3d1] != merged_set_no[b3d2]:
                            warn('FOUND TWO THAT SHOULD HAVE BEEN MATCHED IN DIFFERENT SETS!!!!!')
            print('After merge:----------------')
            printGeneralInfo()
            for merge_set in merges:
                Blob3d.merge(list(merge_set))
        else:
            print('Didnt find any blob3ds to merge')


    @staticmethod
    def merge(b3dlist):
        print('Called merge on b3dlist: ' + str(b3dlist))
        b3d = b3dlist.pop()
        while len(b3dlist):
            next = b3dlist.pop()
            Blob3d.merge2(b3d, next)
        return b3d


    @staticmethod
    def merge2(b1, b2):
        '''
        Merges two blob3ds, and updates the entires of all data structures that link to these b3ds
        The chosen id to merge2 to is the smaller of the two available
        Returns the new merged blob3d in addition to updating its entry in Blob3d.all
        :param b1: The first b3d to merge2
        :param b2: The second b3d to merge2
        :return:
        '''
        if b1 == -1 or b2 == -1:
            print('Skipping merging b3ds' + str(b1) + ' and ' + str(b2) + ' because at least one of them is -1, this should be fixed soon..') # TODO
        else:
            b1 = Blob3d.get(b1)
            b2 = Blob3d.get(b2)
            print('-MERGING two b3ds: ' + str(b1) + '   ' + str(b2))

            # if b1.id < b2.id: #HACK
            smaller = b1
            larger = b2
            # else:
            #     smaller = b2
            #     larger = b1
            for blob2d in larger.blob2ds:
                Blob2d.all[blob2d].b3did = smaller.id
                smaller.blob2ds.append(blob2d)

            smaller.children += larger.children

            if larger.parentID is not None:
                Blob3d.get(larger.parentID).children.remove(larger.id)
                Blob3d.get(larger.parentID).children.append(smaller.id)


            for child in larger.children:
                Blob3d.all[child].parentID = smaller.id

            smaller.blob2ds = list(set(smaller.blob2ds))
            del Blob3d.all[larger.id]


    def validate(self):
        Blob3d.all[self.id] = self


    @staticmethod
    def get(id):
        return Blob3d.all[id]

    @staticmethod
    def getDepth(depth, ids=True):
        if ids:
            return list(b3d.id for b3d in Blob3d.all.values() if b3d.recursive_depth == depth)
        else:
            return list(b3d for b3d in Blob3d.all.values() if b3d.recursive_depth == depth)

    def __str__(self):
        parent_str = ' , Parent B3d: ' + str(self.parentID)
        child_str = ' , Children: ' + str(self.children)
        return str('B3D(' + str(self.id) + '): #b2ds:' + str(len(self.blob2ds)) + ', r_depth:' + str(self.recursive_depth) +
                   ', bead=' + str(self.isBead) +
                   ' lowslideheight=' + str(self.lowslideheight) + ' highslideheight=' + str(self.highslideheight) +
                   #' #edgepixels=' + str(len(self.edge_pixels)) + ' #pixels=' + str(len(self.pixels)) +
                   ' (xl,xh,yl,yh)range:(' + str(self.minx) + ',' + str(self.maxx) + ',' + str(self.miny) + ',' + str(self.maxy) + parent_str + child_str + ')')

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

    def add_note(self, str):
        if hasattr(self, 'note'):
            self.note += str
        else:
            self.set_note(str)

    @staticmethod
    def tagBlobsSingular(blob3dlist, quiet=False):
        singular_count = 0
        non_singular_count = 0
        for blob3d in blob3dlist:
            singular = True
            for blob2d_num, blob2d in enumerate(blob3d.blob2ds):
                if blob2d_num != 0 or blob2d_num != len(blob3d.blob2ds): # Endcap exceptions due to texture
                    if len(blob3d.pairings) > 3: # Note ideally if > 2 # FIXME strange..
                        singular = False
                        break
            blob3d.isSingular = singular
            # Temp:
            if singular:
                singular_count += 1
            else:
                non_singular_count += 1
        if not quiet:
            print('There are ' + str(singular_count) + ' singular 3d-blobs and ' + str(non_singular_count) + ' non-singular 3d-blobs')

    @staticmethod
    def tag_all_beads():
        base_b3ds = Blob3d.getDepth(0, ids=False)
        for b3d in base_b3ds:
            b3d.check_bead()
        # clean up
        unset = sorted( list(b3d for b3d in Blob3d.all.values() if b3d.isBead is None),
                        key=lambda b3d: b3d.recursive_depth) # Do by recursive depth
        print('When tagging all beads, there were ' + str(len(unset)) + ' b3ds which could not be reached from base b3ds')
        print(' They are: ' + str(unset)) # Want this to always be zero, otherwise theres a tree problem
        for b3d in unset:
            b3d.check_bead()
        print("Total number of beads = " + str(sum(b3d.isBead for b3d in Blob3d.all.values())) + ' / ' + str(len(Blob3d.all)))
        print('DB printing all b3ds:')
        for b3d in Blob3d.all.values():
            print('  ' + str(b3d))


    def check_bead(self):
        child_bead_count = 0
        for child in self.children:
            child_is_bead = Blob3d.get(child).check_bead()
            if child_is_bead:
                child_bead_count += 1
        # print('Calling check_bead, max_subbeads_to_be_a_bead = ' + str(max_subbeads_to_be_a_bead), end='')
        # print(', max_pixels_to_be_a_bead = ' + str(max_pixels_to_be_a_bead) + ', child_bead_difference = ' + str(child_bead_difference))
        # if self.recursive_depth > 0:
            # DEBUG
        print('Checking if b3d: ' + str(self) + ' is a bead, child_bead_count=' + str(child_bead_count) )

        self.isBead = (child_bead_count < config.max_subbeads_to_be_a_bead) and (self.get_edge_pixel_count() <= config.max_pixels_to_be_a_bead) and (self.recursive_depth > 0)# and  (child_bead_count > (len(self.children) - config.child_bead_difference))
        return self.isBead

    @staticmethod
    def cleanB3ds():
        '''
        This is a dev method, used to clean up errors in b3ds. Use sparingly!
        :return:
        '''
        print('<< CLEANING B3DS >>')
        for b3d in Blob3d.all.values():
            remove_children = []
            for child in b3d.children:
                if child not in Blob3d.all:
                    remove_children.append(child)
            if len(remove_children):
                for child in remove_children:
                    b3d.children.remove(child)
                print('While cleaning b3d:' + str(b3d) + ' had to remove children that no longer existed ' + str(remove_children))


    def save2d(self, filename):
        '''
        This saves the 2d area around a blob3d for all slides, so that it can be used for testing later
        :param filename: The base filename to save, will have numerical suffix
        :return:
        '''
        from scipy import misc as scipy_misc
        import numpy as np
        slice_arrays = []
        for i in range(self.highslideheight - self.lowslideheight + 1):
            slice_arrays.append(np.zeros((self.maxx - self.minx + 1, self.maxy - self.miny + 1)))
        savename = config.FIGURES_DIR + filename
        for b2d in self.blob2ds:
            for pixel in b2d.pixels:
                slice_arrays[pixel.z - self.lowslideheight][pixel.x - self.minx][pixel.y - self.miny] = pixel.val
        for slice_num, slice in enumerate(slice_arrays):
            img = scipy_misc.toimage(slice, cmin=0.0, cmax=255.0)
            print('Saving Image of Blob2d as: ' + str(savename) + str(slice_num) + '.png')
            img.save(savename+ str(slice_num) + '.png')
