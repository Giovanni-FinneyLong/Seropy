__author__ = 'gio'

from Slide import Slide, printElapsedTime
from Blob3d import Blob3d
from Blob2d import Blob2d
import pickle # Note uses cPickle automatically ONLY IF python 3
from Stitches import Pairing
from serodraw import *
import glob
import sys


# @profile
def save(blob3dlist, filename, directory=PICKLEDIR):
    if directory != '':
        if directory[-1] not in ['/', '\\']:
            slash = '/'
        else:
            slash = ''
    filename = directory + slash + filename
    print('Saving to pickle:'+ str(filename))
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
def load(filename, directory=PICKLEDIR):
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


# @profile
def bloom_b3ds(blob3dlist):

    allb2ds = [Blob2d.get(b2d) for b3d in blob3dlist for b2d in b3d.blob2ds]
    t_start_bloom = time.time()
    num_unbloomed = len(allb2ds)
    for bnum, blob2d in enumerate(allb2ds): # HACK need to put the colon on the right of start_offset
        print('Blooming b2d: ' + str(bnum) + '/' + str(len(allb2ds)) + ' = ' + str(blob2d) )
        blob2d.bloomInwards() # NOTE will have len 0 if no blooming can be done
    print('To complete all blooming:')
    printElapsedTime(t_start_bloom, time.time())
    print('Before blooming there were: ' + str(num_unbloomed) + ' b2ds contained within b3ds, there are now ' + str(len(Blob2d.all)))

# @profile
def main():

    print('Current recusion limit: ' + str(sys.getrecursionlimit()) + ' updating to: ' + str(recursion_limit))
    sys.setrecursionlimit(recursion_limit) # HACK
    if test_instead_of_data:
         picklefile = 'All_test_pre_b3d_tree.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
    else:
        picklefile = 'All_data_pre_b3d_tree.pickle'
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
        t_gen_slides_0 = time.time()
        for imagefile in all_images:
            all_slides.append(Slide(imagefile)) # Pixel computations are done here, as the slide is created.
        # Note now that all slides are generated, and blobs are merged, time to start mapping blobs upward, to their possible partners

        print('Total # of non-zero pixels: ' + str(Pixel.total_pixels) + ', total number of pixels after filtering: ' + str(len(Pixel.all)))
        print('Total # of blob2ds: ' + str(len(Blob2d.all)))
        print('To generate all slides, ', end='')
        printElapsedTime(t_gen_slides_0, time.time())
        print("Pairing all blob2ds with their potential partners in adjacent slides", flush=True)
        Slide.setAllPossiblePartners(all_slides)
        print("Setting shape contexts for all blob2ds",flush=True)
        Slide.setAllShapeContexts(all_slides)
        t_start_munkres = time.time()
        stitchlist = Pairing.stitchAllBlobs(all_slides, debug=False) # TODO change this to work with a list of ids or blob2ds
        t_finish_munkres = time.time()
        print('Done stitching together blobs, ', end='')
        printElapsedTime(t_start_munkres, t_finish_munkres)
        print('About to combine 2d blobs into 3d', flush=True)
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
        Blob3d.tagBlobsSingular(blob3dlist)
        print('Pickling the results of stitching:')
        save(blob3dlist, picklefile)

    else:


        if False:
            blob3dlist = load(picklefile) # DEBUG DEBUG DEBUG
            bloom_b3ds(blob3dlist)
            save(blob3dlist, picklefile + '_BLOOMED')
            #Fall through and do computations
        else:
            if False:
                blob3dlist = load(picklefile + '_BLOOMED') # DEBUG DEBUG DEBUG
                # Fall through to do computations below
            else:
                blob3dlist = load(picklefile + '_BLOOMED_stitched') # DEBUG DEBUG DEBUG
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


                print('Plotting entire blob3dslist by depth')
                plotBlob3ds(blob3dlist, color='depth')
                print('Plotting entire blob3dslist blob')
                plotBlob3ds(blob3dlist, color='blob')
                print('Plotting chosen b3ds by blob')
                plotBlob3ds(chosen_b3ds, color='blob')
                print('Plotting chosen b3ds by depth')

                plotBlob3ds(chosen_b3ds, color='depth')
                # print('Plotting chosen b2ds 2')
                # plotBlob3ds(chosen_b3ds, color='depth')

                plotBlob2ds(Blob2d.all.values()) # TODO this is much faster, write a wrapper already!
                print('Listing all b3ds:')
                for b3d in blob3dlist:
                    print(b3d)

                exit()

    # Time to try to pair together inner b2ds
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
                    b2d.setPossiblePartners(ids_by_height[height_val + 1])

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
        save(all_gen_b3ds + blob3dlist, picklefile + '_BLOOMED_stitched')





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