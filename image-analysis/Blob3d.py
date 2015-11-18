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
        self.stitches = []
        self.lowslide = min(blob.slide.id_num for blob in self.blob2ds)
        self.highslide = max(blob.slide.id_num for blob in self.blob2ds)
        # self.edge_pixels = [pixel for pixels in self.blob2ds for pixel in pixels]
        # blob.edge_pixels for blob in self.blob2ds
        # self.pixels = [pixel for pixels in blob.pixels for pixel in pixels]
        # blob.pixels for blob in self.blob2ds
        self.pixels = []
        self.edge_pixels = []
        for blob in self.blob2ds:
            self.pixels += blob.pixels
            self.edge_pixels += blob.edge_pixels
            for stitch in blob.stitches:
                if stitch not in self.stitches: # TODO set will be faster
                    self.stitches.append(stitch)
        self.maxx = max(blob.maxx for blob in self.blob2ds)
        self.maxy = max(blob.maxy for blob in self.blob2ds)
        self.miny = min(blob.miny for blob in self.blob2ds)
        self.minx = min(blob.minx for blob in self.blob2ds)
        self.isSingular = False
        self.subblobs = []
        self.note = '' # This is a note that can be manually added for identifying certain characteristics..

    def __str__(self):
        if hasattr(self, 'subblob') and self.subblob is True:
            sb = 'sub'
        else:
            sb = ''
        return str('B3D(' + str(sb) + '): lowslide=' + str(self.lowslide) + ' highslide=' + str(self.highslide) + ' #edgepixels=' + str(len(self.edge_pixels)) + ' #pixels=' + str(len(self.pixels)) + ' (xl,xh,yl,yh)range:(' + str(self.minx) + ',' + str(self.maxx) + ',' + str(self.miny) + ',' + str(self.maxy) +')')

    def add_note(self, str):
        if hasattr(self, 'note'):
            self.note += str
        else:
            self.set_note(str)
    def set_note(self, str):
        self.note = str

    def gen_subblob3ds(self, save=False, filename=''):
        test_slides = []
        # for blob3d in blob3dlist:
        for blob2d in self.blob2ds:
            test_slides.append(SubSlide(blob2d, self))
        setAllPossiblePartners(test_slides)
        setAllShapeContexts(test_slides)
        test_stitches = stitchAllBlobs(test_slides)
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
        tagBlobsSingular(b3ds)
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
        # slice_arrays = [np.zeros((self.maxx - self.minx + 1, self.maxy - self.miny + 1))] * (self.highslide - self.lowslide + 1) # HACK on +1
        slice_arrays = []
        for i in range(self.highslide - self.lowslide + 1):
            slice_arrays.append(np.zeros((self.maxx - self.minx + 1, self.maxy - self.miny + 1)))
        savename = FIGURES_DIR + filename

        for b2d in self.blob2ds:
            for pixel in b2d.pixels:
                slice_arrays[pixel.z - self.lowslide][pixel.x - self.minx][pixel.y - self.miny] = pixel.val
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
