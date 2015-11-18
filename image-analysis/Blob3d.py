from Slide import Slide, SubSlide
from Stitches import Pairing


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
        self.lowslideheight = min(blob.slide.height for blob in self.blob2ds)
        self.highslideheight = max(blob.slide.height for blob in self.blob2ds)
        self.pixels = []
        self.edge_pixels = []
        for blob in self.blob2ds:
            self.pixels += blob.pixels
            self.edge_pixels += blob.edge_pixels
            for stitch in blob.pairings:
                if stitch not in self.pairings: # TODO set will be faster
                    self.pairings.append(stitch)
        self.maxx = max(blob.maxx for blob in self.blob2ds)
        self.maxy = max(blob.maxy for blob in self.blob2ds)
        self.miny = min(blob.miny for blob in self.blob2ds)
        self.minx = min(blob.minx for blob in self.blob2ds)
        # self.avgx = sum(blob.avgx * len(blob.pixels) for blob in self.blob2ds) / len(self.pixels) # FIXME this is faster is more accurate, but not working correctly..
        # self.avgy = sum(blob.avgx * len(blob.pixels) for blob in self.blob2ds) / len(self.pixels)
        self.avgx = sum(blob.avgx for blob in self.blob2ds) / len(self.blob2ds)
        self.avgy = sum(blob.avgy for blob in self.blob2ds) / len(self.blob2ds)


        # self.avgx = sum(pixel.x for blob in self.blob2ds for pixel in blob.edge_pixels) / len(self.pixels)
        # self.avgy = sum(pixel.y for blob in self.blob2ds for pixel in blob.edge_pixels) / len(self.pixels)

        # print('DB avg x,y,x2,y2=' + str([self.avgx, self.avgy, self.avgx2, self.avgy2]))


        self.avgz = (self.lowslideheight + self.highslideheight) / 2



        self.isSingular = False
        self.subblobs = []
        self.note = '' # This is a note that can be manually added for identifying certain characteristics..

    def __str__(self):
        if hasattr(self, 'subblob') and self.subblob is True:
            sb = 'sub'
        else:
            sb = ''
        return str('B3D(' + str(sb) + '): lowslideheight=' + str(self.lowslideheight) + ' highslideheight=' + str(self.highslideheight) + ' #edgepixels=' + str(len(self.edge_pixels)) + ' #pixels=' + str(len(self.pixels)) + ' (xl,xh,yl,yh)range:(' + str(self.minx) + ',' + str(self.maxx) + ',' + str(self.miny) + ',' + str(self.maxy) + ')')

    def add_note(self, str):
        if hasattr(self, 'note'):
            self.note += str
        else:
            self.set_note(str)
    def set_note(self, str):
        self.note = str

    @staticmethod
    def tagBlobsSingular(blob3dlist):
        singular_count = 0
        non_singular_count = 0
        for blob3d in blob3dlist:
            singular = True
            for blob2d_num, blob2d in enumerate(blob3d.blob2ds):
                if blob2d_num != 0 or blob2d_num != len(blob3d.blob2ds): # Endcap exceptions due to texture
                    if len(blob3d.pairings) > 3: # Note ideally if > 2
                        singular = False
                        break
            blob3d.isSingular = singular
            # Temp:
            if singular:
                singular_count += 1
            else:
                non_singular_count += 1
        print('There are ' + str(singular_count) + ' singular 3d-blobs and ' + str(non_singular_count) + ' non-singular 3d-blobs')



    def gen_subblob3ds(self, save=False, filename=''):
        test_slides = []
        # for blob3d in blob3dlist:
        for blob2d in self.blob2ds:
            test_slides.append(SubSlide(blob2d, self))
        Slide.setAllPossiblePartners(test_slides)
        Slide.setAllShapeContexts(test_slides)
        test_stitches = Pairing.stitchAllBlobs(test_slides)
        list3ds = []
        for slide_num, slide in enumerate(test_slides):
            for blob in slide.blob2dlist:
                buf = blob.getconnectedblob2ds()
                if len(buf) != 0:
                    list3ds.append((buf, slide))
        b3ds = []
        for (blob2dlist, sourceSubSlide) in list3ds:
            b3ds.append(SubBlob3d(blob2dlist, sourceSubSlide))
        if save:
            doPickle(b3ds, filename)
        # print('Derived a total of ' + str(len(test_b3ds)) + ' 3d blobs')
        Blob3d.tagBlobsSingular(b3ds)
        if not hasattr(self, 'subblobs'): # HACK FIXME once regen pickle
            self.subblobs = []
        self.subblobs = self.subblobs + b3ds
        return b3ds, test_stitches

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
    def __init(self, blob2dlist, parentB3d):
        super().__init__(self, blob2dlist, True)
        self.parent = parentB3d
        # These subblobs need to have offsets, so that they can be correctly placed within their corresponding b3ds when plotting
        # NOTE minx,miny,maxx,maxy are all set in Blob3d.__init__
        self.offsetx = self.minx
        self.offsety = self.miny
        self.width = self.maxx - self.minx
        self.height = self.maxy - self.miny
        # print('DB creating a new subblob3d')
        # for blob_num, b2d in enumerate(blob2dlist):
        #     print('  B2d: ' + str(blob_num) + ' / ' + str(len(blob2dlist)) + ' = ' + str(b2d))
        #     print("  Minx:" + str(b2d.minx) + ' Maxx:' + str(b2d.maxx) + ' Miny:' + str(b2d.miny) + ' Maxy:' + str(b2d.maxy))
