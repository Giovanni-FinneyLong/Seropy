from Slide import Slide, SubSlide, printElapsedTime
from Stitches import Pairing
from sero import doPickle
from serodraw import plotBlob3d, showSlide, showBlob2d, plotBlob2ds, debug
import time
from Blob2d import Blob2d
class Blob3d:
    '''
    A group of blob2ds that chain together with pairings into a 3d shape
    Setting subblob=True indicates that this is a blob created from a pre-existing blob3d.
    '''
    total_blobs = 0

    def __init__(self, blob2dlist, subblob=False):
        self.id = Blob3d.total_blobs
        Blob3d.total_blobs += 1
        self.blob2ds = blob2dlist          # List of the blob 2ds used to create this blob3d
        # Now find my pairings
        self.isSubblob = subblob # T/F
        self.pairings = []
        self.lowslideheight = min(Blob2d.get(blob).height for blob in self.blob2ds)
        self.highslideheight = max(Blob2d.get(blob).height for blob in self.blob2ds)
        self.pixels = []
        self.edge_pixels = []
        self.recursive_depth = 0
        for blobid in self.blob2ds:
            blob = Blob2d.get(blobid)
            if blob is None:
                print('WARNING got None when looking for a blob2d with id:' + str(blobid))
                print('All:' + str(Blob2d.getall()))
                print('Used ids:' + str(Blob2d.used_ids))
                print('All raw: ' + str(Blob2d.all))
                buf = Blob2d.all
                debug()

            self.pixels += blob.pixels
            self.edge_pixels += blob.edge_pixels
            for stitch in blob.pairings:
                if stitch not in self.pairings: # TODO set will be faster
                    self.pairings.append(stitch)
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

    def __str__(self):
        if hasattr(self, 'recursive_depth'):
            sb = str(self.recursive_depth)
        else:
            sb = '0'
        return str('B3D(' + str(sb) + '): #b2ds:' + str(len(self.blob2ds)) + ', r_depth:' + str(self.recursive_depth) + ' lowslideheight=' + str(self.lowslideheight) + ' highslideheight=' + str(self.highslideheight) + ' #edgepixels=' + str(len(self.edge_pixels)) + ' #pixels=' + str(len(self.pixels)) + ' (xl,xh,yl,yh)range:(' + str(self.minx) + ',' + str(self.maxx) + ',' + str(self.miny) + ',' + str(self.maxy) + ')')

    def add_note(self, str):
        if hasattr(self, 'note'):
            self.note += str
        else:
            self.set_note(str)
    def set_note(self, str):
        self.note = str

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

    def save2d(self, filename):
        '''
        This saves the 2d area around a blob3d for all slides, so that it can be used for testing later
        :param filename: The base filename to save, will have numerical suffix
        :return:
        '''
        # slice_arrays = [np.zeros((self.maxx - self.minx + 1, self.maxy - self.miny + 1))] * (self.highslideheight - self.lowslideheight + 1) # HACK on +1
        slice_arrays = []
        for i in range(self.highslideheight - self.lowslideheight + 1):
            slice_arrays.append(np.zeros((self.maxx - self.minx + 1, self.maxy - self.miny + 1)))
        savename = FIGURES_DIR + filename
        for b2d in self.blob2ds:
            for pixel in b2d.pixels:
                slice_arrays[pixel.z - self.lowslideheight][pixel.x - self.minx][pixel.y - self.miny] = pixel.val
        for slice_num, slice in enumerate(slice_arrays):
            img = scipy_misc.toimage(slice, cmin=0.0, cmax=255.0)
            print('Saving Image of Blob2d as: ' + str(savename) + str(slice_num) + '.png')
            img.save(savename+ str(slice_num) + '.png')

class SubBlob3d(Blob3d):
    '''

    '''
    def __init__(self, blob2dlist, parentB3d):
        super().__init__(blob2dlist, True)
        if type(parentB3d) is Blob3d: # Not subblob3d
            self.recursive_depth = 1
        else:
            self.recursive_depth = parentB3d.recursive_depth + 1
        self.parent = parentB3d
        # These subblobs need to have offsets, so that they can be correctly placed within their corresponding b3ds when plotting
        # NOTE minx,miny,maxx,maxy are all set in Blob3d.__init__
        self.offsetx = self.minx
        self.offsety = self.miny
        self.width = self.maxx - self.minx
        self.height = self.maxy - self.miny
