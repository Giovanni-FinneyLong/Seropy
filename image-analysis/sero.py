__author__ = 'gio'


from myconfig import *
from Stitches import *
import munkres as Munkres
from Slide import *
from Blob3d import *
import pickle # Note uses cPickle automatically ONLY IF python 3
from Stitches import Pairing
from serodraw import *





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


def doPickle(blob3dlist, filename, directory='pickles', note=''):
    pickledict = dict()
    pickledict['blob3ds'] = blob3dlist
    pickledict['xdim'] = xdim
    pickledict['ydim'] = ydim
    pickledict['zdim'] = zdim
    pickledict['note'] = note # TODO use info to rememeber info about different pickles
    if directory != '':
        filename = directory + '/' + filename
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
        filename = directory + '/' + filename
        print('Loading from pickle:' + str(filename))
        pickledict = pickle.load(open(filename, "rb"))
        blob3dlist = pickledict['blob3ds']
        xdim = pickledict['xdim']
        ydim = pickledict['ydim']
        zdim = pickledict['zdim']
        # TODO look for info
        if 'note' in pickledict and pickledict['note'] != '':
            print('Included note:' + pickledict['note'])
        setglobaldims(xdim, ydim, zdim)
        # print('Before unpickle, Blob2d.total_blobs=' + str(Blob2d.total_blobs))
        for blob3d in blob3dlist:
            for blob2d in blob3d.blob2ds:
                blob2d.validateID(quiet=True)
                Blob2d.total_blobs += 1
        # for index, val in enumerate(Blob2d.used_ids):
        #     print('(' + str(index) + ',' + str(val) + ')', end=',')
        # new_b2ds = [blob2d for blob3d in blob3dlist for blob2d in blob3d.blob2ds]
        # print('\nThere is a total of ' + str(len(new_b2ds)) + ' new b2ds')
        # for b2d in new_b2ds:
        #     print(b2d.id, end = ',')

        # print('After unpickle, Blob2d.total_blobs=' + str(Blob2d.total_blobs))


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


def bloomInwards(blob2d):
    # TODO this will require a method to determine if a point is inside a polygon
    # See: https://en.wikipedia.org/wiki/Point_in_polygon

    usedpix = set(blob2d.edge_pixels)
    livepix = set(set(blob2d.pixels) - set(blob2d.edge_pixels))

    bloomstages = []
    last_edge = set(blob2d.edge_pixels)

    while(len(livepix) > 1):
        alldict = Pixel.pixelstodict(livepix)
        edge_neighbors = set()
        for pixel in last_edge:
            edge_neighbors = edge_neighbors | set(pixel.neighborsfromdict(alldict)) # - set(blob2d.edge_pixels)
        edge_neighbors = edge_neighbors - last_edge
        bloomstages.append(list(edge_neighbors))
        last_edge = edge_neighbors
        livepix = livepix - edge_neighbors
        # plotPixels(blob2d.edge_pixels)
        # plotPixels(edge_neighbors)
        # plotPixels(livepix)
    # print(bloomstages)
    # print('Iterations:' + str(len(bloomstages)))
    return bloomstages


def main():

    sys.setrecursionlimit(7000) # HACK

    note = 'Was created by setting distance cost log to base 2 instead of 10, and multiplying by contour_cost'
    if test_instead_of_data:
         picklefile = 'pickletest_refactor4.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
    else:
        picklefile = 'all_data_regen_after_stitches_refactored_to_pairing_log2_times.pickle'
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
        if not test_instead_of_data:
            all_images = all_images[:3]
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
                buf = blob.getconnectedblob2ds()
                if len(buf) != 0:
                    list3ds.append(buf)
        blob3dlist = []
        for blob2dlist in list3ds:
            blob3dlist.append(Blob3d(blob2dlist))
        print('There are a total of ' + str(len(blob3dlist)) + ' blob3ds')
        Blob3d.tagBlobsSingular(blob3dlist) # TODO improve the simple classification
        for blob3d in blob3dlist:
            blob3d.set_note(note)
        doPickle(blob3dlist, picklefile)

    else:

        # blob3dlist = unPickle(directory='H:/Dropbox/Serotonin/pickles/recursive/', filename='depth1_subset_of_b3ds.pickle'))
        blob3dlist = unPickle(picklefile)

    # b2ds = [b2d for b3d in blob3dlist for b2d in b3d.blob2ds]
    # plotBlob2ds(b2ds) #TODO need to remove stitches from blob2ds?
    # plotBlob3ds(blob3dlist)


    # for blob3d in blob3dlist: # HACK
    #     blob3d.recursive_depth = 0
    # if False:
    #     print('Before:' + str(len(blob3dlist)))
    #     Blob3d.generateSublobs(blob3dlist)
    #     print('After:' + str(len(blob3dlist)))
    #     if test_instead_of_data:
    #         doPickle(blob3dlist, 'all_test_blobs_and_subblobs.pickle')
    #     else:
    #         doPickle(blob3dlist, 'all_data_blobs_and_subblobs.pickle')
    #
    # else:
    #     if test_instead_of_dat0a:
    #         blob3dlist = unPickle('all_test_blobs_and_subblobs.pickle')
    #     else:
    #         blob3dlist = unPickle('all_data_blobs_and_subblobs.pickle')

    # plotBlob2ds(blob3dlist[0].blob2ds) # DEBUG no issue plotting b2ds here
    # blob3dlist = sorted(blob3dlist, key=lambda b3d: b3d.pixels, reverse=True) # Sorting for biggest first
    allb2ds = sorted([blob2d for blob3d in blob3dlist for blob2d in blob3d.blob2ds], key=lambda b2d: len(b2d.edge_pixels), reverse=True)

    #picklefix
    for b2d in allb2ds:
        b2d.recursive_depth = 0
        b2d.parentID = -1
        b2d.children = []



    # print('Number of b2ds=' + str(len(allb2ds)))
    # print('Original b2ds:')
    # prepixellists = [b2d.pixels for b2d in allb2ds]
    # plotPixelLists(prepixellists)

    no_bloom_b2ds = []
    all_gen_b2ds = []
    for bnum, blob2d in enumerate(allb2ds):
        # showBlob2d(b2d)
        print('Blooming b2d: ' + str(bnum) + '/' + str(len(allb2ds)) + ' = ' + str(blob2d) )
        bloomstages = bloomInwards(blob2d) # NOTE will have len 0 if no blooming can be done

        # print(' Showing blooming, stages=' + str(len(bloomstages)))
        # plotPixelLists(bloomstages)

        if len(bloomstages) == 0:
            print('Didnt derive any bloomstages for b2d:' + str(blob2d))
            # plotBlob2ds([blob2d], stitches=False)
            no_bloom_b2ds.append(blob2d)
        #TODO now need to analyze the stages of blooming
        blob2dlists_by_stage = []


        for snum, stage in enumerate(bloomstages):
            b2ds = Blob2d.pixels_to_blob2ds(stage, parentID=blob2d.id, recursive_depth=blob2d.recursive_depth+1+snum, modify=False) # NOTE making new pixels, rather than messing with existing
            # TODO UPDATE CHILDREN OF BLOB2d


            blob2dlists_by_stage.append(b2ds)
        b2ds_from_b2d = [b2d for blob2dlist in blob2dlists_by_stage for b2d in blob2dlist]
        all_gen_b2ds = all_gen_b2ds + b2ds_from_b2d
        print(' Generated ' + str(len(b2ds_from_b2d)) + ' blob2ds from b2d:' + str(blob2d))
        if False: # Visualizing results of b2d => stages => blob2ds
            print('Blob2d:' + str(blob2d) + ' generated ' + str(len(bloomstages)) + ' bloom stages')
            print('Bloomstages:')
            plotPixelLists(bloomstages)
            print('Plotting all b2ds generated from b2d:' + str(blob2d) + ' a total of: ' + str(len(b2ds_from_b2d)))
            allpixels = [b2d.pixels for b2d in b2ds_from_b2d]
            print('Plotting pixel lists from generated blob2ds')
            # for pixel in blob2d.pixels:
            #     pixel.z -= 2
            # allpixels.append(blob2d.pixels)
            plotPixelLists(allpixels)
    print('Generated a total of ' + str(len(all_gen_b2ds)) + ' blob2ds, from the original' + str(len(allb2ds)))
    # plotBlob2ds(all_gen_b2ds, stitches=True, coloring='depth')



    plotBlob2ds(all_gen_b2ds + allb2ds, stitches=True, coloring='depth')
    allpixellists = [b2d.pixels for b2d in all_gen_b2ds]
    # plotBlob2ds(all_gen_b2ds)
    # print('RETESTING plotblob2ds')
    # plotBlob2ds(blob3dlist[0].blob2ds) # DEBUG
    # plotBlob2ds([all_gen_b2ds[0]]) # DEBUG
    # plotBlob3d(blob3dlist[0])

    # plotPixelLists(allpixellists)





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
    # blob3dlist[2].blob2ds[0].saveImage('test2.jpg')
    # Note, current plan is to find all blob3d's that exists with singular pairings between each 2d blob
    # These blobs should be 'known' to be singular blobs.
    # An additional heuristic may be necessary, potentially using the cost from Munkres()
    # Or computing a new cost based on the displacements between stitched pixelsS
    # Note: Without endcap mod (3 instead of 2), had 549 singular blobs, 900 non-singular
    # Note: Ignoring endcaps, and setting general threshold to 3 instead of 2, get 768 songular, and 681 non-singular

if __name__ == '__main__':
    main()  # Run the main function

# TODO time to switch to sparse matrices, it seems that there are indeed computational optimizations
# TODO other sklearn clustering techniques: http://scikit-learn.org/stable/modules/clustering.html
