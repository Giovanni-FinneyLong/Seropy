__author__ = 'gio'


from myconfig import *
from Stitches import *
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
        if 'note' in pickledict and pickledict['note'] != '':
            print('Included note:' + pickledict['note'])
        setglobaldims(xdim, ydim, zdim)
        for blob3d in blob3dlist:
            for blob2d in blob3d.blob2ds:
                Blob2d.all[blob2d].validateID(quiet=True) # NOTE by validating the id here, we are adding the blob2d to the master array
                Blob2d.total_blobs += 1
        return blob3dlist


def segment_horizontal(blob3d):
    splitblobs2dpairs = [] # Store as tuples
    for blob2d_num,blob2d in enumerate(blob3d.blob2ds):
        upward_stitch = 0
        downward_stitch = 0
        display = False
        for stitch in blob2d.pairings:
            if blob2d == stitch.lowerblob:
                upward_stitch += 1
            if blob2d == stitch.upperblob:
                downward_stitch += 1
        if upward_stitch > 1 or downward_stitch > 1:
            print('Found instance of multiple stitching on b2d:' + str(blob2d_num) + ' which has ' + str(downward_stitch)
                  + ' downward and ' + str(upward_stitch) + ' upward pairings')
            display = True
        if display:
            plotBlob3d(blob3d)


def bloomInwards(blob2d, depth=0):
    # TODO this will require a method to determine if a point is inside a polygon
    # See: https://en.wikipedia.org/wiki/Point_in_polygon
    livepix = set(set(blob2d.pixels) - set(blob2d.edge_pixels))
    # else:
    #     livepix = set(blob2d.edge_pixels)
    bloomstages = []
    last_edge = set(blob2d.edge_pixels)

    alldict = Pixel.pixelstodict(livepix)
    edge_neighbors = set()
    for pixel in last_edge:
        edge_neighbors = edge_neighbors | set(pixel.neighborsfromdict(alldict)) # - set(blob2d.edge_pixels)
    edge_neighbors = edge_neighbors - last_edge
    bloomstages.append(livepix)
    livepix = livepix - edge_neighbors
    b2ds = Blob2d.pixels_to_blob2ds(bloomstages[-1], parentID=blob2d.id, recursive_depth=blob2d.recursive_depth+1, modify=False) # NOTE making new pixels, rather than messing with existing
    if len(livepix) > 1:
        for b2d in b2ds:
            bloomInwards(Blob2d.get(b2d), depth=depth+1)



def main():

    sys.setrecursionlimit(7000) # HACK

    note = 'Was created by setting distance cost log to base 2 instead of 10, and multiplying by contour_cost'
    if test_instead_of_data:
         # picklefile = 'pickletest_refactor4.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
         # picklefile = 'pickletest_converting_blob2ds_to_static.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
         picklefile = 'pickletest_envy.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
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
        print(all_images)
        all_slides = []

        for imagefile in all_images:
            all_slides.append(Slide(imagefile)) # Pixel computations are done here, as the slide is created.
        # Note now that all slides are generated, and blobs are merged, time to start mapping blobs upward, to their possible partners

        Slide.setAllPossiblePartners(all_slides)
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
            # print('DB Creating b3d from blob2dlist:' + str(blob2dlist))
            blob3dlist.append(Blob3d(blob2dlist))
        print('There are a total of ' + str(len(blob3dlist)) + ' blob3ds')
        Blob3d.tagBlobsSingular(blob3dlist) # TODO improve the simple classification
        for blob3d in blob3dlist:
            blob3d.set_note(note)
        doPickle(blob3dlist, picklefile)

    else:
        # blob3dlist = unPickle(directory='H:/Dropbox/Serotonin/pickles/recursive/', filename='depth1_subset_of_b3ds.pickle'))
        blob3dlist = unPickle(picklefile)
    # plotBlob3ds(blob3dlist)
    allb2ds = sorted([Blob2d.get(blob2d) for blob3d in blob3dlist for blob2d in blob3d.blob2ds], key=lambda b2d: len(b2d.edge_pixels), reverse=True)


    start_offset = 0

    all_desc = []
    t_start_bloom = time.time()
    for bnum, blob2d in enumerate(allb2ds[start_offset:]): # HACK need to put the colon on the right of start_offset
        # showBlob2d(b2d)
        print('Blooming b2d: ' + str(bnum + start_offset) + '/' + str(len(allb2ds)) + ' = ' + str(blob2d) )
        bloomstages = bloomInwards(blob2d) # NOTE will have len 0 if no blooming can be done
        base = Blob2d.get(blob2d.id)
        desc = base.getdescendants()
        all_desc += desc
        # print('Printing descendants ')
        # base.printdescendants()
        # plotBlob2ds(desc, edge=False, ids=True, parentlines=True)

        # for index, b2d in enumerate(desc):
        #     b2d.height = b2d.recursive_depth
        # print('Desc:' + str(desc))
        # plotBlob2ds(desc, edge=False, ids=True, parentlines=True,explode=True, titleNote=str('The blob serving as the base is #' + str(bnum) + ' / ' + str(len(allb2ds))))

    print('To complete all blooming:')
    printElapsedTime(t_start_bloom, time.time())

    #refresh
    # allb2ds = sorted([Blob2d.get(blob2d) for blob3d in blob3dlist for blob2d in blob3d.blob2ds], key=lambda b2d: len(b2d.edge_pixels), reverse=True)

    # rec_is_3 = [b2d for b2d in Blob2d.all.values() if b2d.recursive_depth == 3]
    # print(len(rec_is_3))
    # test = rec_is_3[0]
    # print('Test=' + str(test))
    # test_desc = test.getrelated()
    # print('All related:')
    #
    # for d in test_desc:
    #     print('  ' + str(d))
    # debug()


    print('Plotting all blob2ds exploded')
    plotBlob2ds(all_desc, edge=True, ids=False, parentlines=True,explode=True)

    debug()




    #refresh
    allb2ds = sorted([Blob2d.get(blob2d) for blob3d in blob3dlist for blob2d in blob3d.blob2ds], key=lambda b2d: len(b2d.edge_pixels), reverse=True)

    count = 0
    children = 0
    for b2d in Blob2d.all.values():
        if len(b2d.children):
            count += 1
            children += len(b2d.children)
    base_without_children = [b2d for b2d in Blob2d.all.values() if len(b2d.children) == 0 and b2d.recursive_depth == 0]
    print('There are ' + str(count) + ' b2ds with children')
    print('There have a total of ' + str(children) + ' children')
    print('There are ' + str(len(base_without_children)) + ' base b2ds who didnt have children')
    print('There are a total of ' + str(len(Blob2d.all)) + ' b2ds')
    # NOTE: The sum of base blobs with & without children and the count of their children = the total number of b2ds, so at this point we are successfully



    # NOTE: Idea to test: keep looking through bloom stages until all blob2ds less than a certain size. Then stitch with the original

    # I expect this will involve creating groups of touching pixels
    # This can be done most efficicently by using just the layers of bloom that have been returned; as there is no need
    # Need to find cases where 2 groups are separated exclusively by the previous layer's pixels
    # NOTE that the avdvantage to this is
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

if __name__ == '__main__':
    main()  # Run the main function

# TODO time to switch to sparse matrices, it seems that there are indeed computational optimizations
# TODO other sklearn clustering techniques: http://scikit-learn.org/stable/modules/clustering.html
