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
    print('Saving to pickle:'+ str(filename) +  '\n current recursion limit:' + str(sys.getrecursionlimit()))
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
    livepix = set(set(blob2d.pixels) - set(blob2d.edge_pixels))
    last_edge = set(blob2d.edge_pixels)

    alldict = Pixel.pixelidstodict(livepix)
    edge_neighbors = set()
    for pixel in last_edge:
        edge_neighbors = edge_neighbors | set(Pixel.get(pixel).neighborsfromdict(alldict)) # - set(blob2d.edge_pixels)
    edge_neighbors = edge_neighbors - last_edge
    bloomstage = livepix
    livepix = livepix - edge_neighbors

    b2ds = Blob2d.pixels_to_blob2ds(bloomstage, parentID=blob2d.id, recursive_depth=blob2d.recursive_depth+1, modify=False) # NOTE making new pixels, rather than messing with existing

    # depth_offset = ''
    # for d in range(depth):
    #     depth_offset += '-'

    for num,b2d in enumerate(b2ds):
        b2d = Blob2d.get(b2d)
        Blob2d.all[blob2d.id].pixels = list(set(Blob2d.all[blob2d.id].pixels) - set(b2d.pixels))

    # print(depth_offset + ' After being bloomed the parent is:' + str(Blob2d.get(blob2d.id)))
    if (len(blob2d.pixels) < len(Blob2d.get(blob2d.id).pixels)):
        warn('Gained pixels!!!! (THIS SHOULD NEVER HAPPEN!)')

    if depth < max_depth:
        if len(livepix) > 1:
            for b2d in b2ds:
                bloomInwards(Blob2d.get(b2d), depth=depth+1)


# @profile
def bloom_b3ds(blob3dlist):

    allb2ds = [Blob2d.get(b2d) for b3d in blob3dlist for b2d in b3d.blob2ds]

    t_start_bloom = time.time()
    num_unbloomed = len(allb2ds)
    prev_count = len(Blob2d.all)
    for bnum, blob2d in enumerate(allb2ds): # HACK need to put the colon on the right of start_offset
        print('Blooming b2d: ' + str(bnum) + '/' + str(len(allb2ds)) + ' = ' + str(blob2d) )
        bloomInwards(blob2d) # NOTE will have len 0 if no blooming can be done
        # rel = blob2d.getrelated(include_self=True)
        # plotBlob2ds(rel)
    print('To complete all blooming:')
    printElapsedTime(t_start_bloom, time.time())
    print('Before blooming there were: ' + str(num_unbloomed) + ' b2ds, there are now ' + str(len(Blob2d.all)))
    #Note that at this point memory usage is 3.4gb, with 12.2K b2ds



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


# @profile
def main():

    # showColors()
    print('Current recusion limit:' + str(sys.getrecursionlimit()) + ' updating to:' + str(recursion_limit))
    sys.setrecursionlimit(recursion_limit) # HACK

    if test_instead_of_data:
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

        all_slides = []

        for imagefile in all_images:
            all_slides.append(Slide(imagefile)) # Pixel computations are done here, as the slide is created.
        # Note now that all slides are generated, and blobs are merged, time to start mapping blobs upward, to their possible partners

        print('Total # of pixels: ' + str(Pixel.total_pixels))
        print('Total # of blob2ds: ' + str(len(Blob2d.all)))
        print("Pairing all blob2ds with their potential partners in adjacent slides", flush=True)
        Slide.setAllPossiblePartners(all_slides)
        # print('DB plotting all b2ds')
        # plotBlob2ds(Blob2d.all.values())

        print("Setting shape contexts for all blob2ds",flush=True)
        Slide.setAllShapeContexts(all_slides)
        t_start_munkres = time.time()
        stitchlist = Pairing.stitchAllBlobs(all_slides, debug=False) # TODO change this to work with a list of ids or blob2ds
        t_finish_munkres = time.time()
        print('Done stitching together blobs, total time for all: ', end='')
        printElapsedTime(t_start_munkres, t_finish_munkres)
        print('About to combine 2d blobs into 3d', flush=True)
        list3ds = []
        for slide_num, slide in enumerate(all_slides):
            for blob in slide.blob2dlist:
                # print(' Working on b2d: ' + str(blob) + ' = ' + str(Blob2d.get(blob)))
                # print('  pairings:' + str(Blob2d.get(blob).pairings), flush=True)
                buf = Blob2d.get(blob).getconnectedblob2ds()
                # print('  buf = ' + str(buf))
                if len(buf) != 0:
                    list3ds.append([b2d.id for b2d in buf])
        blob3dlist = []
        for blob2dlist in list3ds:
            blob3dlist.append(Blob3d(blob2dlist))
        print('There are a total of ' + str(len(blob3dlist)) + ' blob3ds')
        Blob3d.tagBlobsSingular(blob3dlist)

        print('Pickling the results of stitching:')
        doPickle2(blob3dlist, picklefile)
        print('Plotting all b3ds that were generated:')
        plotBlob3ds(blob3dlist, color='blob')
        print('Plotting all b2ds that were generated:')
        plotBlob2ds(Blob2d.all.values(), ids=(len(Blob2d.all) < 500)) # Only show ids if less that 500 blob2ds
        print('All b3ds generated:')
        for b3d in blob3dlist:
            print(' ' + str( b3d))
        print('All b2ds generated:')
        for b2d in Blob2d.all.values():
            print(' ' + str(b2d))

    else:


        if False:
            blob3dlist = unPickle2(picklefile) # DEBUG DEBUG DEBUG
            bloom_b3ds(blob3dlist)
            doPickle2(blob3dlist, picklefile + '_BLOOMED')

        else:
            if False:
                blob3dlist = unPickle2(picklefile + '_BLOOMED') # DEBUG DEBUG DEBUG
                # Fall through to do computations below
            else:
                blob3dlist = unPickle2(picklefile + '_BLOOMED_stitched') # DEBUG DEBUG DEBUG
                chosen_depths = [1,2,3,4]
                chosen_b3ds = [b3d for b3d in blob3dlist if b3d.recursive_depth in chosen_depths]
                # for b3d in chosen_b3ds:
                #     print('Plotting b3d: ' + str(b3d))
                #     for b2d in b3d.blob2ds:
                #         print(' ' + str(Blob2d.get(b2d)))
                #     plotBlob2ds([Blob2d.get(b2d) for b2d in b3d.blob2ds],ids=True)
                #
                # for b3d in chosen_b3ds:
                #     print(b3d)


                print('Plotting entire blob3dslist 1')
                plotBlob3ds(blob3dlist, color='depth')
                print('Plotting entire blob3dslist 2')
                plotBlob3ds(blob3dlist, color='blob')
                print('Plotting chosen b2ds 1')
                plotBlob3ds(chosen_b3ds, color='blob')
                plotBlob3ds(chosen_b3ds, color='depth')
                # print('Plotting chosen b2ds 2')
                # plotBlob3ds(chosen_b3ds, color='depth')

                plotBlob2ds(Blob2d.all.values())
                exit()

    # Time to try to pair together inner b2ds

    # depth_0 = [b2d.id for b2d in Blob2d.all.values() if b2d.recursive_depth == 0]
    # print(len(depth_0))

    if dePickle:

        max_avail_depth = max(b2d.recursive_depth for b2d in Blob2d.all.values())

        for cur_depth in range(max_avail_depth)[1:]: # Skip those at depth 0
            print('CUR DEPTH = ' + str(cur_depth))
            depth = [b2d.id for b2d in Blob2d.all.values() if b2d.recursive_depth == cur_depth]
            max_h_d0 = max(Blob2d.all[b2d].height for b2d in depth)
            min_h_d0 = min(Blob2d.all[b2d].height for b2d in depth)
            print(' Number at depth ' + str(cur_depth) + ' : ' + str(len(depth)))
            print(' Min max heights at depth ' + str(cur_depth) + ' : (' + str(min_h_d0) + ', ' + str(max_h_d0) + ')')
            ids_by_height = [[] for i in range(max_h_d0 - min_h_d0 + 1)]
            for b2d in depth:
                ids_by_height[Blob2d.get(b2d).height - min_h_d0].append(b2d)
            print(' Ids by height:')
            for height_val,h in enumerate(ids_by_height[:-1]): # All but the last one
                print('  Height:' + str(height_val))
                for b2d in h:
                    b2d = Blob2d.all[b2d]
                    # print('   Setting partners for:' + str(b2d))
                    b2d.setPossiblePartners(ids_by_height[height_val + 1])
                    # print('   Set possible partners = :' + str(b2d.possible_partners))
                # print('  DB set possible partners for b2ds at height')
                # plotBlob2ds([Blob2d.get(b2d) for b2d in h])

            for h in ids_by_height:
                for b2d in h:
                    b2d = Blob2d.all[b2d]
                    b2d.setShapeContexts(36)

        print('DB the max depth available is:' + str(max_avail_depth))
        b3ds_by_depth_offset = []
        for depth_offset in range(max_avail_depth+1)[1:]: # Skip offset of zero, which refers to the b3ds which have already been stitched
            print('Depth_offset: ' + str(depth_offset))

            new_b3ds = []
            for b3d in blob3dlist:
                all_d1_with_pp_in_this_b3d = []
                for b2d in b3d.blob2ds:
                    #Note this is the alternative to storing b3dID with b2ds
                    b2d = Blob2d.get(b2d)
                    # print(' DB b2d: ' + str(b2d))
                    # print(' DB r_depth of all descendants: ' + str(set([blob2d.recursive_depth for blob2d in b2d.getdescendants()])))
                    d_1 = [blob for blob in b2d.getdescendants() if blob.recursive_depth == b2d.recursive_depth + depth_offset]
                    if len(d_1):
                        for desc in d_1:
                            if len(desc.possible_partners):
                                all_d1_with_pp_in_this_b3d.append(desc.id)
                print(' For b3d: ' + str(b3d) + ' found ' + str(all_d1_with_pp_in_this_b3d))
                all_d1_with_pp_in_this_b3d = set(all_d1_with_pp_in_this_b3d)
                for b2d in all_d1_with_pp_in_this_b3d:
                    b2d = Blob2d.get(b2d)
                    cur_matches = [b2d] # NOTE THIS WAS CHANGED BY REMOVED .getdescendants() #HACK
                    for pp in b2d.possible_partners:
                        if pp in all_d1_with_pp_in_this_b3d:
                            print('--> Found a partner to b2d: ' + str(b2d) + ' which is: ' + str(Blob2d.get(pp)))
                            print('     Adding related:' + str(Blob2d.get(pp).getpartnerschain()))
                            print('     -DB FROM related, the different recursive depths are:' + str(set([Blob2d.get(blob2d).recursive_depth for blob2d in Blob2d.get(pp).getpartnerschain()])))
                            print('     len of cur_matches before: ' + str(len(cur_matches)))
                            cur_matches += [Blob2d.get(b) for b in Blob2d.get(pp).getpartnerschain()]
                            print('     len of cur_matches after: ' + str(len(cur_matches)))

                    if len(cur_matches) > 1:
                        print('  All cur_matches: (' + str(len(cur_matches)) + ') ' + str(cur_matches) + '    from b2d: ' + str(b2d))
                        if len(cur_matches) != len(set(cur_matches)):
                            warn(' There are duplicates in cur_matches!')
                            print('CUR_MATCHES=' + str(cur_matches))
                        # plotBlob2ds(cur_matches)
                        # matches_with_parents = cur_matches + list(set([Blob2d.get(b2d.parentID) for b2d in cur_matches]))
                        # print(' Matches with parents: (' + str(len(matches_with_parents)) + '), ' + str(matches_with_parents))
                        # plotBlob2ds(matches_with_parents, stitches=False)
                        # print('  DB appending a new b3d from: ' + str([blob for blob in cur_matches])
                        new_b3ds.append(Blob3d([blob.id for blob in cur_matches if blob.recursive_depth == b2d.recursive_depth], subblob=True, r_depth = b2d.recursive_depth))
            print('All new_b3ds: (' + str(len(new_b3ds)) + ') : ' + str(new_b3ds))
            b3ds_by_depth_offset.append(new_b3ds)
            # plotBlob3ds(new_b3ds, coloring='blob')
            # plotBlob3ds(new_b3ds + blob3dlist, coloring='depth')

            # for b3d in new_b3ds:
            #     plotBlob3d(b3d)
        all_gen_b3ds = []
        for offset_num, depth_offset_b3ds in enumerate(b3ds_by_depth_offset):
            print('Working on offset ' + str(offset_num) + ' / ' + str(len(b3ds_by_depth_offset)))
            # for b3d_num,b3d in enumerate(depth_offset_b3ds):
            #     print(' Working on b3d: ' + str(b3d_num) + ' / ' + str(len(depth_offset_b3ds)))
            #     Pairing.stitchBlob2ds(b3d.blob2ds, debug=False)
            # plotBlob3ds(depth_offset_b3ds, coloring='blob')
            all_gen_b3ds += depth_offset_b3ds
        print('Plotting all b3ds that were just generated')
        plotBlob3ds(all_gen_b3ds, color='blob')
        doPickle2(all_gen_b3ds + blob3dlist, picklefile + '_BLOOMED_stitched')





    # plotBlob2ds(depth, stitches=True, ids=False, parentlines=False,explode=True, edge=False)



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