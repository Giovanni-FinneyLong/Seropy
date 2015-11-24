__author__ = 'gio'


from myconfig import *

from Stitches import *
import munkres as Munkres
from Slide import *
from Blob3d import *
import pickle # Note uses cPickle automatically ONLY IF python 3
from Stitches import Pairing
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


def doPickle(blob3dlist, filename, directory='', note=''):
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


def unPickle(filename, directory=''):
        if directory != '':
            filename = directory + '/' + filename
        print('Loading from pickle:' + str(filename))
        pickledict = pickle.load(open(filename, "rb"))
        blob3dlist = pickledict['blob3ds']
        xdim = pickledict['xdim']
        ydim = pickledict['ydim']
        zdim = pickledict['zdim']
        # TODO look for info
        if 'note' in pickledict:
            print('Included note:' + pickledict['note'])
        setglobaldims(xdim, ydim, zdim)
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

    sys.setrecursionlimit(3000) # HACK

    # expDistance()
    # debug()
    note = 'Was created by setting distance cost log to base 2 instead of 10, and multiplying by contour_cost'
    if test_instead_of_data:
         ## picklefile = 'pickletest_testsnip3.pickle'
         ## picklefile = 'pickletest_testsnip2.pickle' # Done logbase 2 for distance, +

         # picklefile = 'pickletest_refactor.pickle' # THIS IS DONE +, and log distance base 10
         #picklefile = 'pickletest_refactor2.pickle' # THIS IS DONE *, and log distance base 2
         picklefile = 'pickletest_refactor3.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 50
        # doPickle(test_b3ds + primary_blobs, directory='H:/Dropbox/Serotonin/pickles/recursive/', filename='depth1_subset_of_b3ds.pickle', note=picklenote)



         # picklefile = 'pickletest_refactor4.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
        # pickletest1 holds the results of recomputing over gen slides from bloblist[3]
         picklefile = 'H:/Dropbox/Serotonin/pickles/recursive/' + 'depth1_subset_of_b3ds.pickle'
    else:
        picklefile = 'pickledata.pickle'

    # NOTE: pickledata.pickle holds the pickle data from all primary b3ds that were created without the additional consideration
    # For the distance between points (not counting the log scaling applied to bins
    # Experiment9.pickle holds the results of pickling on the 'special b3ds' designated below, and does include consideration for distance
        #picklefile = 'pickledata_distance.pickle'

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

        print(all_images)
        all_slides = []
        for imagefile in all_images:
            all_slides.append(Slide(imagefile)) # Pixel computations are done here, as the slide is created.
        # Note now that all slides are generated, and blobs are merged, time to start mapping blobs upward, to their possible partners

        Slide.setAllPossiblePartners(all_slides)
        Slide.setAllShapeContexts(all_slides)
        t_start_munkres = time.time()
        stitchlist = Pairing.stitchAllBlobs(all_slides)
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



    # plotBlob3ds(blob3dlist, coloring='singular', costs=0, b3dmidpoints=True)
    # debug()


# '''    # DEBUG
#     mult_b3ds= unPickle('pickletest_snip.pickle')
#     offsetx = 500
#     offsety = 00
#     for b3d in mult_b3ds:
#         for pixel in b3d.pixels:
#             pixel.x += offsetx
#             pixel.y += offsety
#         for b2d in b3d.blob2ds:
#             b2d.minx += offsetx
#             b2d.maxx += offsetx
#             b2d.miny += offsety
#             b2d.maxy += offsety
#         b3d.isSingular = True
#     # plotBlob3ds(blob3dlist + mult_b3ds, coloring='singular', costs=False)
# '''


    experimenting = False
    if experimenting:
        # NOTE Blob3dlist[3].blob2ds[6] should be divided into subblobs
        # NOTE [3][1] is also good
        unpickle_exp = True
        pickle_exp = True # Only done if not unpickling
        exp_pickle = 'pickletest_subblobs.pickle' # 2,3 working #6 working except height, 7 working except stitch height, 8 works!!!

        primary_blobs = blob3dlist #[blob3dlist[3], blob3dlist[8], blob3dlist[40]]
        if unpickle_exp:
            test_b3ds = unPickle(exp_pickle)
        else:
            # primary_blobs = blob3dlist
            test_b3ds = []
            for b3d_num, b3d in enumerate(primary_blobs):
                print('DB GENERATING SUBBLOBS For B3d #' + str(b3d_num) + ' / ' + str(len(primary_blobs)))
                buf = b3d.gen_subblob3ds()
                test_b3ds = test_b3ds + buf[0] # buf[0] b/c currently returning b3ds,stitchlist
                # print('Derived a total of ' + str(len(buf[0])) + ' subblob3ds from primary b3d:' + str(b3d))
            if pickle_exp:
                doPickle(test_b3ds, exp_pickle)
                #doPickle(primary_blobs, picklefile) # Note this will cause an error if run more than once

        # picklenote = "These were generated from a few interesting blob3ds from the full list of blob3ds\nThere has been one level of recursion performed"
        # doPickle(test_b3ds + primary_blobs, directory='H:/Dropbox/Serotonin/pickles/recursive/', filename='depth1_subset_of_b3ds.pickle', note=picklenote)




        for b3d in primary_blobs:
            b3d.isSingular = True
        for blob in test_b3ds:
            blob.isSingular = False
        # print(test_b3ds)
        # print(primary_blobs)
        # print(test_b3ds + primary_blobs)
        # plotBlob3ds(test_b3ds, color='blobs')
        # plotBlob3ds(primary_blobs, color='blobs')
        # plotBlob3ds(test_b3ds + primary_blobs, color='blob')
        # for b_num, b3d in enumerate(primary_blobs):
        #     print(b_num, end = ': ')
        #     print(b3d)
        #     print(b_num, end = ': ')
        #     print(b3d.subblobs)
        # for sub_bnum, b3d in enumerate(test_b3ds):
        #     print('SB:' + str(sub_bnum) + '/' + str(len(test_b3ds)) + ':'  + str(b3d))
        #     if hasattr(b3d, 'parent'):
        #         print('HAS PARENT:' + str(b3d.parent))
        #     else:
        #         print('No Parent:')
            # print('  ' + str(b3d.parent))



        # plotBlob3ds(test_b3ds + primary_blobs, coloring='depth', b2dmidpoints=False, canvas_size=(1000,1000), b2d_midpoint_values=10)
        # plotBlod3ds(blob3dlist)
    plotBlob3ds(blob3dlist, coloring='depth', b2dmidpoints=False, canvas_size=(1000,1000), b2d_midpoint_values=10)
    debug()

    #TODO


    ## sub_b3ds, sub_stitchs =  blob3dlist[40].gen_subblob3ds(save=True, filename='subblobs1.pickle')
    # print('Derived a total of ' + str(len(list3ds)) + ' 3d blob components')
    # print('Derived a total of ' + str(len(test_b3ds)) + ' 3d blobs')
    # # GOOD BLOBS FOR RECURSIVE TESTING:
    # # 3,8,40!
    # print('DB plotting sublob3ds:')
    #
    # tagBlobsSingular(sub_b3ds)
    # # plotBlod3ds(sub_b3ds, color='singular')
    # print('DB PICKLING ALL WITH 1 LAYER SUBBLOBS')
    # doPickle(sub_b3ds, 'all_subblobs.pickle')

    # sub_b3ds = unPickle('all_subblobs.pickle')






    # plotSlidesVC(all_slides, stitchlist, pairings=True, polygons=False, edges=True, color='slides', subpixels=False, midpoints=False, context=False, animate=False, orders=anim_orders, canvas_size=(1000, 1000), gif_size=(400,400))#, color=None)
    # NOTE: Interesting blob3ds:
    # 3: Very complex, mix of blobs, irregular stitching example, even including a seperate group (blob3d)
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
