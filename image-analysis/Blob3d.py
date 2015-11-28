from Slide import Slide, SubSlide
from Stitches import Pairing
from sero import doPickle
from serodraw import plotBlob3d, showSlide, showBlob2d, plotBlob2ds

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
        self.recursive_depth = 0
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
        if hasattr(self, 'recursive_depth'):
            sb = str(self.recursive_depth)
        else:
            sb = '0'
        return str('B3D(' + str(sb) + '): lowslideheight=' + str(self.lowslideheight) + ' highslideheight=' + str(self.highslideheight) + ' #edgepixels=' + str(len(self.edge_pixels)) + ' #pixels=' + str(len(self.pixels)) + ' (xl,xh,yl,yh)range:(' + str(self.minx) + ',' + str(self.maxx) + ',' + str(self.miny) + ',' + str(self.maxy) + ')')

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
                    if len(blob3d.pairings) > 3: # Note ideally if > 2
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



    def gen_subblob3ds(self, save=False, filename='', **kwargs):

        debugflag = kwargs.get('debugflag', -1)
        debug2ds = kwargs.get('debugforb2ds',[])
        # DEBUG
        debugging = (debugflag == 1)
        if debugging:
            print('Debugging for blob3d:' + str(self))
            print('Targeting blob2ds#:' + str(debug2ds))
            # plotBlob3d(self,coloring='blob2d', b2dids=True)
            # print('DB showing plotting with plotblob2ds')
            # plotBlob2ds(self.blob2ds, ids=True)

        display = False
        # /DEBUG

        test_slides = []
        # # HACK
        # self.blob2ds = sorted(self.blob2ds, key=lambda b2d: len(b2d.pixels), reverse=True)
        # #HACK

        # DEBUG
        # debug2ds = [5,9,11]
        slides_from_debug_blob2ds = []
        debug_blob2ds_sublobs = []

        # /DEBUG


        for b2d_num,blob2d in enumerate(self.blob2ds):
            if debugging and b2d_num in debug2ds and display:
                print('From blob2d #' + str(b2d_num) + ':' + str(blob2d))
                showBlob2d(blob2d)
            test_slides.append(SubSlide(blob2d, self))
            if debugging and b2d_num in debug2ds:
                # print('>>Adding another ' + str(len(test_slides[-1].blob2dlist)) + ' blob2ds from generated subslide')
                slides_from_debug_blob2ds.append(test_slides[-1])
                debug_blob2ds_sublobs += test_slides[-1].blob2dlist

                if display:
                    print('Created subslide:' + str(test_slides[-1]))
                    showSlide(test_slides[-1])
                    print('---Now showing the ' + str(len(test_slides[-1].blob2dlist)) + ' blob2ds which have been generated')
                    for subb2d in test_slides[-1].blob2dlist:
                        showBlob2d(subb2d)
        # DEBUG
        # NOTE finding the correspond slide_nums and blob_nums so that they can be identified and observed within the below static Slide methods
        # TODO TODO TODO Continue tracking the source of the issue

        if debugging: #DEBUG
            debug_blob2ds = [blob2d for b2d_num, blob2d in enumerate(self.blob2ds) if b2d_num in debug2ds]
            print('> Found the following ' + str(len(debug_blob2ds)) + ' blob2ds which are being debugged: ' + str(debug_blob2ds))
            # Now need to find slides that derived from debug_blob2ds
            print('> Found the following ' + str(len(slides_from_debug_blob2ds)) + ' slides which are being debugged: ' + str(slides_from_debug_blob2ds))
            print('> Found the following ' + str(len(debug_blob2ds_sublobs)) + ' blob2ds which are within the slides from the blob2ds being debugged: ' + str(debug_blob2ds_sublobs))
            # NOTE at this point, have lists of debug_blob2ds, and their generated slides and blob2ds:
            # All generated subslides and subblob2ds have had a debugFlag set
            for slide in slides_from_debug_blob2ds:
                slide.debugFlag = True
            for b2d in debug_blob2ds_sublobs:
                b2d.debugFlag = True
        Slide.setAllPossiblePartners(test_slides, **kwargs)

        if debugging:
            for b2d_num,b2d in enumerate(debug_blob2ds_sublobs):
                print('Possible partners of b2d #' + str(b2d_num) + ':' + str(b2d.possible_partners))
        # DEBUG to here, everything ok Seems to be finding possible_partners ok


        Slide.setAllShapeContexts(test_slides)
        test_pairings = Pairing.stitchAllBlobs(test_slides, quiet=True, debug=False)
        if debugging:
            print('Done stitching all blob2ds')
            for b2d in debug_blob2ds_sublobs:
                print(' B2D:' + str(b2d) + ' has ' + str(len(b2d.possible_partners)) + ' possible partners')
                # print('  Connectedb2ds:' + str(b2d.getconnectedblob2ds()))
                print('  # Pairings:' + str(len(b2d.pairings)))

        list3ds = []
        for slide_num, slide in enumerate(test_slides):
            for blob in slide.blob2dlist:
                buf = blob.getconnectedblob2ds(debug=True) # DEBUG
                if len(buf) != 0:
                    list3ds.append((buf, slide))

        #DEBUG
        if debugging:
            print('>>>>The debug_blob2ds_sublobs are:' + str(debug_blob2ds_sublobs))

            print("There are a total of " + str(len(list3ds)) + ' lists of blob2ds, each of which will make a blob2d')
            for list2dblobs, sourcesubslide in list3ds:
                inter_buf = list(set(list2dblobs).intersection(debug_blob2ds_sublobs))
                if len(inter_buf) != 0:
                    print(' Found ' + str(len(inter_buf)) + ' common blob2ds which are:' + str(inter_buf))
            print('There are a total of ' + str(len(test_pairings)) + ' pairings between the created blobs')
            for pairing in test_pairings:
                if pairing.lowerblob in debug_blob2ds_sublobs or pairing.upperblob in debug_blob2ds_sublobs:
                    print(' Found a stitch containing a debug_subblob:' + str(pairing))


        original_b2ds = [blob2d for blob2d in self.blob2ds]
        sub_b2ds = [blob2d for slide in test_slides for blob2d in slide.blob2dlist]
        # DEBUG DEBUG DEBUG
        # plotBlob2ds(original_b2ds + sub_b2ds)
        #DEBUG


        # NOTE currently found the generated subblobs (which arent showing up), within test_pairings and list3ds
        # Now need to check if they are making it into the 3d blobs
        # DEBUG DEBUG The debug b2ds are not showing up in the above lists. SO they are not getting added correctly in .getconnectedblob2ds()
        b3ds = []
        for (blob2dlist, sourceSubSlide) in list3ds:
            b3ds.append(SubBlob3d(blob2dlist, self))

         # DEBUG
        if debugging:
            print("There are a total of " + str(len(b3ds)) + ' blob3ds')
            for b3d in b3ds:
                inter_buf =  list(set(b3d.blob2ds).intersection(debug_blob2ds_sublobs))
                if len(inter_buf) != 0:
                    print(' Found ' + str(len(inter_buf)) + ' blob2ds within a blob3d. The blob2ds are:' + str(inter_buf))
            print('>>>>The debug_blob2ds_sublobs are:' + str(debug_blob2ds_sublobs))







        if save:
            doPickle(b3ds, filename)
        # print('Derived a total of ' + str(len(test_b3ds)) + ' 3d blobs')
        Blob3d.tagBlobsSingular(b3ds, quiet=True)
        if not hasattr(self, 'subblobs'): # HACK FIXME once regen pickle
            self.subblobs = []
        self.subblobs = self.subblobs + b3ds
        return b3ds, test_pairings, test_slides

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
        # print('DB creating a new subblob3d')
        # for blob_num, b2d in enumerate(blob2dlist):
        #     print('  B2d: ' + str(blob_num) + ' / ' + str(len(blob2dlist)) + ' = ' + str(b2d))
        #     print("  Minx:" + str(b2d.minx) + ' Maxx:' + str(b2d.maxx) + ' Miny:' + str(b2d.miny) + ' Maxy:' + str(b2d.maxy))
