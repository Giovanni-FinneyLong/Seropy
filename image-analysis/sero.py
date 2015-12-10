__author__ = 'gio'


from myconfig import *

from Stitches import *
import munkres as Munkres
from Slide import *
from Blob3d import *
import pickle # Note uses cPickle automatically ONLY IF python 3
from Stitches import Pairing
from serodraw import *
# from skimage import filters
# from skimage import segmentation
# from PIL import ImageFilter
# from collections import OrderedDict
# import readline
# import code
# import rlcompleter
# from pympler import asizeof
# from scipy import misc as scipy_misc
# import threading






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


def experiment():
    from skimage import measure


    # Construct some test data
    x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
    # print('x:' + str(x))
    # print('y:' + str(y))
    rr = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2))) # Has shape (100,100)
    print('r:' + str(rr))
    debug()
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(rr, 0.8)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def expDistance():

    save2ds = False
    repickle = False
    offsetx = 000
    offsety = 400
    if repickle:
        b3ds = unPickle('pickledata.pickle')
        b3dds = unPickle('pickledata_distance.pickle')

        interests = [b3ds[3], b3dds[3]]
        for pixel in interests[0].pixels:
            pixel.x += offsetx
            pixel.y += offsety
        for b2d in interests[0].blob2ds:
            b2d.minx += offsetx
            b2d.maxx += offsetx
            b2d.miny += offsety
            b2d.maxy += offsety

        doPickle(interests, 'experiment.pickle')
    else:
        interests = unPickle('experiment.pickle')

    regen3list = unPickle('pickletest.pickle')
    regen3listlog2 = unPickle('pickletest_logbase2.pickle')

    for regen3 in regen3list:
        for pixel in regen3.pixels:
            pixel.x += 2 * offsetx
            pixel.y += 2 *offsety
        for b2d in regen3.blob2ds:
            b2d.minx += 2 *offsetx
            b2d.maxx += 2 *offsetx
            b2d.miny += 2 *offsety
            b2d.maxy += 2 *offsety
    for regen3 in regen3listlog2:
        for pixel in regen3.pixels:
            pixel.x += 3 * offsetx
            pixel.y += 3 *offsety
        for b2d in regen3.blob2ds:
            b2d.minx += 3 *offsetx
            b2d.maxx += 3 *offsetx
            b2d.miny += 3 *offsety
            b2d.maxy += 3 *offsety



    if save2ds:
        image_based_name = 'bloblist(3)b2ds_'
        interests[1].save2d(image_based_name)


    interests[0].isSingular = True # GREEN FOR SINGULAR
    interests[1].isSingular = False # RED FOR NOT SINGULAR

    # for stitch in interests[0].pairings:
    #     for lowerpnum, upperpnum in stitch.indeces:
    #         stitch.lowerpixels[lowerpnum]
    #         stitch.upperpixels[upperpnum]

    plotBlob3ds(interests + regen3list + regen3listlog2, color='singular')


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
        # blob3dlist = unPickle(picklefile)
        pass

    # plotBlob3ds(blob3dlist)

    # for blob3d in blob3dlist: # HACK
    #     blob3d.recursive_depth = 0
    if False:
        print('Before:' + str(len(blob3dlist)))
        Blob3d.generateSublobs(blob3dlist)
        print('After:' + str(len(blob3dlist)))
        if test_instead_of_data:
            doPickle(blob3dlist, 'all_test_blobs_and_subblobs.pickle')
        else:
            doPickle(blob3dlist, 'all_data_blobs_and_subblobs.pickle')

    else:
        if test_instead_of_data:
            blob3dlist = unPickle('all_test_blobs_and_subblobs.pickle')
        else:
            blob3dlist = unPickle('all_data_blobs_and_subblobs.pickle')

    b2d = blob3dlist[0].blob2ds[0]
    plotBloomInwards(b2d)




    # plotBlob3ds(blob3dlist, coloring='blob', showStitches=True)
    # showColors()
    # TODO TODO USE BLOOMING INWARD TECHNIQUE (ONLY ON EXISTING PIXELS),
    # Look for the first collision, keep track of generator pixel, so that can assume that they came from other directions




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
