__author__ = 'gio'


import munkres as Munkres
from Slide import *
from Blob3d import *
import pickle # Note uses cPickle automatically ONLY IF python 3
from Stitches import Pairing
from serodraw import *
import glob




def munkresCompare(blob1, blob2):
    '''
    Uses the Hungarian Algorithm implementation from the munkres package to find an optimal combination of points
    between blob1 and blob2 as well as deriving the point->point relationships and storing them in indeces
    '''
    # TODO try this with and without manual padding; change min/max for ndim and change the end padding portion of makeCostArray

    def costBetweenPoints(bins1, bins2):
        assert len(bins1) == len(bins2)
        cost = 0
        for i in range(len(bins1)):
            if (bins1[i] + bins2[i]) != 0:
                cost += math.pow(bins1[i] - bins2[i], 2) / (bins1[i] + bins2[i])
        return cost / 2

    def makeCostArray(blob1, blob2):
        # Setting up cost array with the costs between respective points
        ndim = max(len(blob1.edge_pixels), len(blob2.edge_pixels)) # HACK HACK min for now, in the hopes that munkres can handle non-square matrices.
        cost_array = np.zeros([len(blob1.edge_pixels), len(blob2.edge_pixels)])

        for i in range(len(blob1.edge_pixels)):
            for j in range(len(blob2.edge_pixels)):
                cost_array[i][j] = costBetweenPoints(blob1.context_bins[i], blob2.context_bins[j])
        return cost_array

    # TODO run this on some simple examples provided online to confirm it is accurate.
    cost_array = makeCostArray(blob1, blob2)
    munk = Munkres()
    indeces = munk.compute(np.copy(cost_array).tolist())
    total_cost = 0
    for row, col in indeces:
        value = cost_array[row][col]
        total_cost += value
        # print ('(%d, %d) -> %d' % (row, col, value))
    # print('Total Cost = ' + str(total_cost))
    return total_cost, indeces


def doPickle(blob3dlist, filename, directory=PICKLEDIR, note=''):
    pickledict = dict()
    pickledict['blob3ds'] = blob3dlist
    pickledict['xdim'] = xdim
    pickledict['ydim'] = ydim
    pickledict['zdim'] = zdim
    pickledict['allb2ds'] = Blob2d.all
    pickledict['allpixels'] = Pixel.all
    pickledict['usedb2ds'] = Blob2d.used_ids
    pickledict['note'] = note # TODO use info to rememeber info about different pickles
    if directory != '':
        if directory[-1] not in ['/', '\\']:
            slash = '/'
        else:
            slash = ''
        filename = directory + slash + filename
    try:
        print('Saving to pickle:'+ str(filename))
        if note != '':
            print('Including note:' + str(note))
        pickle.dump(pickledict, open(filename, "wb"))
    except RuntimeError:
        print('\nIf recursion depth has been exceeded, you may increase the maximal depth with: sys.setrecursionlimit(<newdepth>)')
        print('The current max recursion depth is: ' + str(sys.getrecursionlimit()))
        print('Opening up an interactive console, press \'n\' then \'enter\' to load variables before interacting, and enter \'exit\' to resume execution')
        debug()
        pass

# @profile
def doPickle2(blob3dlist, filename, directory=PICKLEDIR):
    if directory != '':
        if directory[-1] not in ['/', '\\']:
            slash = '/'
        else:
            slash = ''
    filename = directory + slash + filename
    print('Saving to pickle:'+ str(filename) +  'current recursion limit:' + str(sys.getrecursionlimit()))
    done = False
    while not done:
        try:
            print('Pickling ' + str(len(blob3dlist)) + ' b3ds')
            t = time.time()
            pickle.dump({'b3ds' : blob3dlist}, open(filename + '_b3ds', "wb"), protocol=0)
            printElapsedTime(t,time.time())

            print('Pickling ' + str(len(Blob2d.all)) + ' b2ds')
            t = time.time()
            pickle.dump({'b2ds' : Blob2d.all, 'used_ids': Blob2d.used_ids}, open(filename + '_b2ds', "wb"), protocol=0)
            printElapsedTime(t,time.time())

            print('Pickling ' + str(len(Pixel.all)) + ' pixels from the total possible ' + str(Pixel.total_pixels))
            t = time.time()
            pickle.dump({'pixels' : Pixel.all, 'total_pixels' : Pixel.total_pixels}, open(filename + '_pixels', "wb"), protocol=0)
            printElapsedTime(t,time.time())
            done = True
        except RuntimeError:
            print('\nIf recursion depth has been exceeded, you may increase the maximal depth with: sys.setrecursionlimit(<newdepth>)')
            print('The current max recursion depth is: ' + str(sys.getrecursionlimit()))
            print('Opening up an interactive console, press \'n\' then \'enter\' to load variables before interacting, and enter \'exit\' to resume execution')
            debug()
            pass

# @profile
def unPickle2(filename, directory=PICKLEDIR):
        if directory[-1] not in ['/', '\\']:
            slash = '/'
        else:
            slash = ''
        filename = directory + slash + filename
        t_start = time.time()
        print('Loading from pickle:' + str(filename))
        print('Loading b3ds ', end='',flush=True)
        t = time.time()
        b3ds = pickle.load(open(filename + '_b3ds', "rb"))['b3ds']
        printElapsedTime(t, time.time())
        print('Loading b2ds ', end='',flush=True)
        t = time.time()

        buff = pickle.load(open(filename + '_b2ds', "rb"))
        Blob2d.all = buff['b2ds']
        Blob2d.used_ids = buff['used_ids']
        Blob2d.total_blobs = len(Blob2d.all)
        printElapsedTime(t, time.time())
        print('Loading pixels ', end='',flush=True)
        t = time.time()
        buff = pickle.load(open(filename + '_pixels', "rb"))
        Pixel.all = buff['pixels']
        Pixel.total_pixels = len(Pixel.all)
        printElapsedTime(t, time.time())

        print('There are a total of:' + str(len(b3ds)) + ' b3ds')
        print('There are a total of:' + str(len(Blob2d.all)) + ' b2ds')
        print('There are a total of:' + str(len(Pixel.all)) + ' pixels')
        print('Total to unpickle: ', end='')
        printElapsedTime(t_start, time.time())
        return b3ds


def unPickle(filename, directory=PICKLEDIR):
        if directory[-1] not in ['/', '\\']:
            slash = '/'
        else:
            slash = ''
        filename = directory + slash + filename
        print('Loading from pickle:' + str(filename))
        pickledict = pickle.load(open(filename, "rb"))
        blob3dlist = pickledict['blob3ds']
        xdim = pickledict['xdim']
        ydim = pickledict['ydim']
        zdim = pickledict['zdim']

        Blob2d.all = pickledict['allb2ds']
        Blob2d.used_ids = pickledict['usedb2ds']
        # TODO look for info
        Pixel.all = pickledict['allpixels']
        Pixel.total_pixels = len(Pixel.all)

        if 'note' in pickledict and pickledict['note'] != '':
            print('Included note:' + pickledict['note'])
        setglobaldims(xdim, ydim, zdim)
        for blob3d in blob3dlist:
            for blob2d in blob3d.blob2ds:
                Blob2d.all[blob2d].validateID(quiet=True) # NOTE by validating the id here, we are adding the blob2d to the master array
                Blob2d.total_blobs += 1
        return blob3dlist


def bloomInwards(blob2d, depth=0):
    # TODO this will require a method to determine if a point is inside a polygon
    # See: https://en.wikipedia.org/wiki/Point_in_polygon
    livepix = set(set(blob2d.pixels) - set(blob2d.edge_pixels))
    # else:
    #     livepix = set(blob2d.edge_pixels)
    last_edge = set(blob2d.edge_pixels)

    # print('     DB CALLING WITHIN BLOOM INWARDS with args:' + str(livepix))
    alldict = Pixel.pixelidstodict(livepix)
    edge_neighbors = set()
    for pixel in last_edge:
        edge_neighbors = edge_neighbors | set(Pixel.get(pixel).neighborsfromdict(alldict)) # - set(blob2d.edge_pixels)
    edge_neighbors = edge_neighbors - last_edge
    bloomstage = livepix
    livepix = livepix - edge_neighbors

    # print('DB the result of bloomstages (should be ids): ' + str(bloomstage))
    b2ds = Blob2d.pixels_to_blob2ds(bloomstage, parentID=blob2d.id, recursive_depth=blob2d.recursive_depth+1, modify=False) # NOTE making new pixels, rather than messing with existing
    # print("DB done converting pixels to b2ds")

    depth_offset = ''
    for d in range(depth):
        depth_offset += '-'
    # print(depth_offset + ' DB: Done blooming b2d:' + str(blob2d) + ' at depth ' + str(depth))
    # print(depth_offset + ' Total children: ' + str(len(blob2d.getdescendants())) + ' = ' + str(blob2d.getdescendants()))

    # NEW trying to 'steal' / inheret non-edgepixels from parents..
    # print(' Parent was originally:' + str(blob2d))




    # FIXME FIXME FIXME FIXME
    # print(' DB stealing from parent: ' + str(blob2d) + ' # children: ' + str(len(b2ds)))
    # print('  Currently entry for parent is:' + str(Blob2d.get(blob2d.id)))
    for num,b2d in enumerate(b2ds):
        b2d = Blob2d.get(b2d)
        # print(' Child ' + str(num) + ' has ' + str(len(b2d.pixels)) + ' pixels and ' + str(len(b2d.edge_pixels)) + ' edge pixels')
        old_size = len(blob2d.pixels)
        # DEBUG
        # blob2d.pixels = list(set(blob2d.pixels) - set(b2d.pixels))
        # Blob2d.all[blob2d.id] = blob2d
        # DEBUG
        #         blob2d.pixels = list(set(blob2d.pixels) - set(b2d.pixels))
        pixels_before = len(Blob2d.all[blob2d.id].pixels)
        Blob2d.all[blob2d.id].pixels = list(set(Blob2d.all[blob2d.id].pixels) - set(b2d.pixels))
        # Blob2d.all[blob2d.id].children.append(b2d.id)

        # print(depth_offset + ' The number of pixels has changed from ' + str(pixels_before) + ' to ' + str(len(Blob2d.all[blob2d.id].pixels)))

    # FIXME FIXME FIXME FIXME



        # print("  Updated parent from " + str(old_size) + ' pixels to ' + str(len(blob2d.pixels)))
    print(depth_offset + ' After being bloomed the parent is:' + str(Blob2d.get(blob2d.id)))
    if (len(blob2d.pixels) < len(Blob2d.get(blob2d.id).pixels)):
        warn('Gained pixels!!!! (THIS SHOULD NEVER HAPPEN!)')

    if depth < max_depth:# HACK
        if len(livepix) > 1:
            for b2d in b2ds:
                bloomInwards(Blob2d.get(b2d), depth=depth+1)



# @profile
def experiment(blob3dlist):

    # plotBlob3ds(blob3dlist)
    allb2ds = sorted([Blob2d.get(blob2d) for blob3d in blob3dlist for blob2d in blob3d.blob2ds], key=lambda b2d: len(b2d.edge_pixels), reverse=True)


    start_offset = 0

    all_desc = []
    t_start_bloom = time.time()
    num_unbloomed = len(allb2ds)
    prev_count = len(Blob2d.all)
    for bnum, blob2d in enumerate(allb2ds[start_offset:]): # HACK need to put the colon on the right of start_offset
        # showBlob2d(b2d)
        print('Blooming b2d: ' + str(bnum + start_offset) + '/' + str(len(allb2ds)) + ' = ' + str(blob2d) )
        # print(' The current number of B2ds = ' + str(len(Blob2d.all)) + ' the previous count = ' + str(prev_count))
        prev_count = len(Blob2d.all)
        bloomInwards(blob2d) # NOTE will have len 0 if no blooming can be done
        # print(' After blooming the above blob, len(Blob2d.all) = ' + str(len(Blob2d.all)))
        # base = Blob2d.get(blob2d.id)
        # desc = base.getdescendants()
        # all_desc += desc

    print('To complete all blooming:')
    printElapsedTime(t_start_bloom, time.time())
    print('Before blooming there were: ' + str(num_unbloomed) + ' b2ds, there are now ' + str(len(Blob2d.all)))


    plotBlob2ds([b2d for b2d in Blob2d.all.values()], edge=True, ids=False, parentlines=True,explode=True)



def explorememoryusage(blob3dlist):
    # Note: Sys.getsizeof() is in bytes
    # slide = slides[0]

    dir = 'pickle_sizes/'
    for b3d_num, b3d in enumerate(blob3dlist):

        print('B3d:' + str(b3d_num) + '/' + str(len(blob3dlist)))
        b3d_dict = {'b3d' : b3d}
        pickle.dump(b3d_dict, open(dir + 'b3d/b3d_size' + str(b3d_num) + '.pickle', "wb"))
        for b2d_num, b2d in enumerate(b3d.blob2ds):
            b2d = Blob2d.get(b2d)
            b2d_dict = {'b2d' : b2d}
            pickle.dump(b2d_dict, open(dir + 'b2d/b2d_size' + str(b3d_num) + '_' + str(b2d_num) + '.pickle', "wb"))
            for p_num, pixel in enumerate(b2d.pixels):
                pixel = Pixel.get(pixel)
                pixel_dict = {'pixel' : pixel}
                pickle.dump(pixel_dict, open(dir + 'pixel/pixel_size' + str(b3d_num) + '_' + str(b2d_num) + '_' + str(p_num) + '.pickle', "wb"))

        # pixel = Pixel.get(b2d.edge_pixels[0])
        # # pickle.dump(pickledict, open(filename, "wb"))
        #
        # b2d_dict = {'b2d' : b2d}
        # pickle.dump(b2d_dict, open(dir + 'b2d_size.pickle', "wb"))
        # pixel_dict = {'pixel' : pixel}
        # pickle.dump(pixel_dict, open(dir + 'pixel_size.pickle', "wb"))



    # from Pixel import Pixel

    # to_check = [b3d]
    # for check in to_check:
    #     print('Checking the memory of :' + str(check) + '=' + str(sys.getsizeof(check)))
    #     for attr in check.__dict__.items():
    #         print(str(attr) + ' => ' + str(sys.getsizeof(attr[1])))
    #         if attr[0] == 'pixels':
    #             pixel =
    #             print(' size of a pixel:' + str(sys.getsizeof(Pixel.get(attr[1][0]))))



# @profile
def main():
    print('Current recusion limit:' + str(sys.getrecursionlimit()) + ' updating to:' + str(recursion_limit))
    sys.setrecursionlimit(recursion_limit) # HACK

    note = 'Was created by setting distance cost log to base 2 instead of 10, and multiplying by contour_cost'
    if test_instead_of_data:
         # picklefile = 'pickletest_refactor4.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
         # picklefile = 'pickletest_converting_blob2ds_to_static.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
         picklefile = 'All_test_redone_with_maximal_blooming.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
    else:
        picklefile = 'All_data_redone_1-5_with_maximal_blooming.pickle'
    if not dePickle:
        setMasterStartTime()
        if test_instead_of_data:
            dir = TEST_DIR
            extension = '*.png'
        else:
            dir = DATA_DIR
            extension = 'Swell*.tif'
        all_images = glob.glob(dir + extension)
        #
        # # HACK
        # if not test_instead_of_data:
        #     all_images = all_images[:3]
        # # HACK
        #
		
        #print(all_images)
        all_slides = []

        for imagefile in all_images:
            all_slides.append(Slide(imagefile)) # Pixel computations are done here, as the slide is created.
        # Note now that all slides are generated, and blobs are merged, time to start mapping blobs upward, to their possible partners

        print('DB - Total # of pixels: ' + str(Pixel.total_pixels))

        print("Pairing all blob2ds with their potential partners in adjacent slides", flush=True)
        Slide.setAllPossiblePartners(all_slides)
        print("Setting shape contexts for all blob2ds",flush=True)
        Slide.setAllShapeContexts(all_slides)
        t_start_munkres = time.time()
        stitchlist = Pairing.stitchAllBlobs(all_slides, debug=False)
        t_finish_munkres = time.time()
        print('Done stitching together blobs, total time for all: ', end='')
        printElapsedTime(t_start_munkres, t_finish_munkres)
        print('About to merge 2d blobs into 3d')
        list3ds = []
        for slide_num, slide in enumerate(all_slides):
            for blob in slide.blob2dlist:
                buf = Blob2d.get(blob).getconnectedblob2ds()
                if len(buf) != 0:
                    list3ds.append([b2d.id for b2d in buf])
        blob3dlist = []
        for blob2dlist in list3ds:
            blob3dlist.append(Blob3d(blob2dlist))
        print('There are a total of ' + str(len(blob3dlist)) + ' blob3ds')
        Blob3d.tagBlobsSingular(blob3dlist) # TODO improve the simple classification
        for blob3d in blob3dlist:
            blob3d.set_note(note)
        print('Pickling the results of stitching:')
        doPickle2(blob3dlist, picklefile)

    else:
        blob3dlist = unPickle2(picklefile) # DEBUG DEBUG DEBUG

        # NOTE: Total # of pixels: 38,953,178
        # 708062 actual
        # # Actually alive = ~40K per slide => 800k



    #
    all_b2ds = [b2d for b2d in Blob2d.all.values()]
    # test = all_b2ds[1]
    # print("Printing test b2d's dict: (len = " + str(len(test.__dict__)) + ')' )
    # for d in test.__dict__.items():
    #     print(d)
    # print("Performing blooming on test b2d")
    # bloomInwards(test)
    # testu = Blob2d.get(test.id)
    # desc = testu.getdescendants()
    # print("Total # of descendants:" + str(len(desc)))
    # print("First desc:" + str(desc[1]))
    # print("Printing test descendant b2d's dict: (len = " + str(len(desc[1].__dict__)) + ')')
    # for d in desc[1].__dict__.items():
    #     print(d)
    #
    # print('---------')
    # print("Original b2d:" + str(test))
    # print("Updated b2d:" + str(testu))

    # print('WARNING after this plotting window is closed, will bloom all b2ds that are available..')
    # plotBlob2ds(all_b2ds, stitches=False, ids=False, parentlines=True,explode=True, canvas_size=(300,300))

    #NOTE at this point, after unpickling the entire swellshark dataset, memory usage is 2.2GB (for Python.exe)



    if True:
        # used_b2ds = [b2d for b3d in blob3dlist for b2d in b3d.blob2ds ]
        # print(len(Blob2d.all.values()))
        # plotBlob2ds(Blob2d.all.values(), stitches=False, explode=False, parentlines=False)
        # print('DONE PICKLING THE NORMAL BLOBS, NOW BLOOMING')
        print('DB blob3dlist:' + str(blob3dlist))
        experiment(blob3dlist)
        doPickle2(blob3dlist, picklefile + '_BLOOMED')

    else:
        blob3dlist = unPickle2(picklefile + '_BLOOMED') # DEBUG DEBUG DEBUG
    all_b2ds = [b2d for b2d in Blob2d.all.values()]
    plotBlob2ds(all_b2ds, stitches=False, ids=False, parentlines=True,explode=True, edge=False)


    # explorememoryusage(blob3dlist)

    # print('Blob2d.all=' + str(Blob2d.all))
    # print(len(all_b2ds))
    # # print('DB printing all b2ds:')
    # for b2d in all_b2ds:
    #     print(b2d)



if __name__ == '__main__':
    main()  # Run the main function

    # NOTE: Idea to test: keep looking through bloom stages until all blob2ds less than a certain size. Then stitch with the original

    # I expect this will involve creating groups of touching pixels
    # This can be done most efficicently by using just the layers of bloom that have been returned; as there is no need
    # Need to find cases where 2 groups are separated exclusively by the previous layer's pixels
    # NOTE that the advantage to this is
    # a) Can setup bloom to work on blob2ds
    # b) Can plot internals with blob2d methods
    # c) Can loop recursively only on larger blob2ds instead of whole layer

    #todo STRATEGY
    # Will make blob2ds out of layers of bloom, by casting each layer to an array
    # Then find the possible partners across bloom layers
    # Then, instead of doing munkres, look between levels for blob2ds which have a min and max x,y to be within the min and max x,y of the earlier bloom



    # NOTE temp: in the original 20 swellshark scans, there are ~ 11K blobs, ~9K pairings
    # Note: Without endcap mod (3 instead of 2), had 549 singular blobs, 900 non-singular
    # Note: Ignoring endcaps, and setting general threshold to 3 instead of 2, get 768 songular, and 681 non-singular