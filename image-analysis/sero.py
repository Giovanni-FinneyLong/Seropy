__author__ = 'gio'

from util import printElapsedTime
from Slide import Slide
from Blob3d import Blob3d
from Blob2d import Blob2d
from Pixel import Pixel
import pickle # Note uses cPickle automatically ONLY IF python 3
from Stitches import Pairing
import glob
import sys
from util import warn, getImages, progressBar
from myconfig import *
import time

if mayPlot: #TODO try test running this as false
    from serodraw import *
    filterAvailableColors()


# @profile
def save(blob3dlist, filename, directory=PICKLEDIR):
    if directory != '':
        if directory[-1] not in ['/', '\\']:
            slash = '/'
        else:
            slash = ''
    filename = directory + slash + filename
    print('\nSaving to file:'+ str(filename))
    done = False
    while not done:
        try:
            print('Pickling ' + str(len(blob3dlist)) + ' b3ds ', end='')
            t0 = t = time.time()
            pickle.dump({'b3ds' : blob3dlist}, open(filename + '_b3ds', "wb"), protocol=0)
            printElapsedTime(t, time.time(), prefix='took')

            print('Pickling ' + str(len(Blob2d.all)) + ' b2ds ', end='')
            t = time.time()
            pickle.dump({'b2ds' : Blob2d.all, 'used_ids': Blob2d.used_ids}, open(filename + '_b2ds', "wb"), protocol=0)
            printElapsedTime(t, time.time(), prefix='took')

            print('Pickling ' + str(len(Pixel.all)) + ' pixels ', end='')
            t = time.time()
            pickle.dump({'pixels' : Pixel.all, 'total_pixels' : Pixel.total_pixels}, open(filename + '_pixels', "wb"), protocol=0)
            printElapsedTime(t, time.time(), prefix='took')
            done = True

            print('Saving took:', end='')
            printElapsedTime(t0, time.time(), prefix='')
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
        print('Loading from file :' + str(filename))
        print('Loading b3ds ', end='',flush=True)
        t = time.time()
        b3ds = pickle.load(open(filename + '_b3ds', "rb"))['b3ds']
        printElapsedTime(t,time.time(), prefix='took')
        print('Loading b2ds ', end='',flush=True)
        t = time.time()

        buff = pickle.load(open(filename + '_b2ds', "rb"))
        Blob2d.all = buff['b2ds']
        Blob2d.used_ids = buff['used_ids']
        Blob2d.total_blobs = len(Blob2d.all)
        printElapsedTime(t,time.time(), prefix='took')
        print('Loading pixels ', end='',flush=True)
        t = time.time()
        buff = pickle.load(open(filename + '_pixels', "rb"))
        Pixel.all = buff['pixels']
        Pixel.total_pixels = len(Pixel.all)
        printElapsedTime(t,time.time(), prefix='took')

        print('There are a total of: ' + str(len(b3ds)) + ' 3d blobs')
        print('There are a total of: ' + str(len(Blob2d.all)) + ' 2d blobs')
        print('There are a total of: ' + str(len(Pixel.all)) + ' pixels')
        print('Total time to load: ', end='')
        printElapsedTime(t_start, time.time(), prefix='')
        return b3ds


# @profile
def bloom_b3ds(blob3dlist, stitch=False, create_progress_bar=True):
    print('\nProcessing internals of 2d blobs via \'blooming\' ', end='')
    allb2ds = [Blob2d.get(b2d) for b3d in blob3dlist for b2d in b3d.blob2ds]
    t_start_bloom = time.time()
    num_unbloomed = len(allb2ds)
    pb = progressBar(max_val=sum(len(b2d.edge_pixels) for b2d in allb2ds))
    for bnum, blob2d in enumerate(allb2ds): # HACK need to put the colon on the right of start_offset
        # print(' Blooming b2d: ' + str(bnum) + '/' + str(len(allb2ds)) + ' = ' + str(blob2d), flush=True)
        blob2d.bloomInwards() # NOTE will have len 0 if no blooming can be done
        pb.update(len(blob2d.edge_pixels), set=False) # set is false so that we add to an internal counter
        # print('DB pb symbols = ' + str(pb.symbols_printed), flush=True)

    pb.finish()

    printElapsedTime(t_start_bloom, time.time(), prefix='took')
    print('Before blooming there were: ' + str(num_unbloomed) + ' b2ds contained within b3ds, there are now ' + str(len(Blob2d.all)))

    # Setting possible_partners
    print('Pairing all new blob2ds with their potential partners in adjacent slides')
    max_avail_depth = max(b2d.recursive_depth for b2d in Blob2d.all.values()) # Note may want to adjust this later to do just some b2ds
    for cur_depth in range(max_avail_depth)[1:]: # Skip those at depth 0
        depth = [b2d.id for b2d in Blob2d.all.values() if b2d.recursive_depth == cur_depth]
        max_h_d = max(Blob2d.all[b2d].height for b2d in depth)
        min_h_d = min(Blob2d.all[b2d].height for b2d in depth)
        # print(' Number at depth ' + str(cur_depth) + ' : ' + str(len(depth)))
        # print(' Min max heights at depth ' + str(cur_depth) + ' : (' + str(min_h_d) + ', ' + str(max_h_d) + ')')
        ids_by_height = [[] for i in range(max_h_d - min_h_d + 1)]
        for b2d in depth:
            ids_by_height[Blob2d.get(b2d).height - min_h_d].append(b2d)
        for height_val,h in enumerate(ids_by_height[:-1]): # All but the last one
            for b2d in h:
                b2d = Blob2d.all[b2d]
                b2d.setPossiblePartners(ids_by_height[height_val + 1])

    # Creating b3ds
    print('Creating 3d blobs from the generated 2d blobs')
    all_new_b3ds = []
    for depth_offset in range(max_avail_depth+1)[1:]: # Skip offset of zero, which refers to the b3ds which have already been stitched
        # print('Depth_offset: ' + str(depth_offset))
        new_b3ds = []
        for b3d in blob3dlist:
            all_d1_with_pp_in_this_b3d = []
            for b2d in b3d.blob2ds:
                #Note this is the alternative to storing b3dID with b2ds
                b2d = Blob2d.get(b2d)
                d_1 = [blob for blob in b2d.getdescendants() if blob.recursive_depth == b2d.recursive_depth + depth_offset]
                if len(d_1):
                    for desc in d_1:
                        if len(desc.possible_partners):
                            all_d1_with_pp_in_this_b3d.append(desc.id)

            all_d1_with_pp_in_this_b3d = set(all_d1_with_pp_in_this_b3d)
            for b2d in all_d1_with_pp_in_this_b3d:
                b2d = Blob2d.get(b2d)
                cur_matches = [b2d] # NOTE THIS WAS CHANGED BY REMOVED .getdescendants() #HACK
                for pp in b2d.possible_partners:
                    if pp in all_d1_with_pp_in_this_b3d:
                        # print('--> Found a partner to b2d: ' + str(b2d) + ' which is: ' + str(Blob2d.get(pp)))
                        # print('     Adding related:' + str(Blob2d.get(pp).getpartnerschain()))
                        # print('     -DB FROM related, the different recursive depths are:' + str(set([Blob2d.get(blob2d).recursive_depth for blob2d in Blob2d.get(pp).getpartnerschain()])))
                        # print('     len of cur_matches before: ' + str(len(cur_matches)))
                        cur_matches += [Blob2d.get(b) for b in Blob2d.get(pp).getpartnerschain()]
                        # print('     len of cur_matches after: ' + str(len(cur_matches)))

                if len(cur_matches) > 1:
                    # print('  All cur_matches: (' + str(len(cur_matches)) + ') ' + str(cur_matches) + '    from b2d: ' + str(b2d))
                    if len(cur_matches) != len(set(cur_matches)):
                        warn(' There are duplicates in cur_matches!')
                        if not disable_warnings:
                            print('CUR_MATCHES=' + str(cur_matches))
                    new_b3ds.append(Blob3d([blob.id for blob in cur_matches if blob.recursive_depth == b2d.recursive_depth], subblob=b3d.id, r_depth = b2d.recursive_depth))
        # print('All new_b3ds: (' + str(len(new_b3ds)) + ') : ' + str(new_b3ds))
        all_new_b3ds += new_b3ds

    if stitch:
        # Set up shape contexts
        print('Setting shape contexts for stitching')
        for b2d in [Blob2d.all[b2d] for b3d in all_new_b3ds for b2d in b3d.blob2ds]:
            b2d.setShapeContexts(36)

        # Stitching
        print('Stitching the newly generated 2d blobs')
        for b3d_num,b3d in enumerate(all_new_b3ds):
            print(' Working on b3d: ' + str(b3d_num) + ' / ' + str(len(all_new_b3ds)))
            Pairing.stitchBlob2ds(b3d.blob2ds, debug=False)
    return all_new_b3ds

#HACK

process_internals = True # Do blooming, set possible partners for the generated b2ds, then create b3ds from them
stitch_base_b2ds = False # FIXME why does making this False cause the progressBar to go crazy?
stitch_bloomed_b2ds = False # Default False

# HACK Move these to both configs!
picklefile =''


# @profile
def main():
    print('Current recusion limit: ' + str(sys.getrecursionlimit()) + ' updating to: ' + str(recursion_limit))
    sys.setrecursionlimit(recursion_limit) # HACK
    if test_instead_of_data:
         picklefile = 'All_test_pre_b3d_tree.pickle' # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
    else:
        if swell_instead_of_c57bl6:
            picklefile = 'Swellshark_Adult_012615.pickle'
        else:
            picklefile = 'C57BL6_Adult_CerebralCortex.pickle'
    if not dePickle:
        all_slides, blob3dlist = Slide.dataToSlides(stitch=stitch_base_b2ds) # Reads in images and converts them to slides.
                                                     # This process involves generating Pixels & Blob2ds, but NOT Blob3ds
        if process_internals:
            bloom_b3ds(blob3dlist, stitch=stitch_bloomed_b2ds) # Includes setting partners, and optionally stitching
        Blob3d.tagBlobsSingular(blob3dlist)
        save(blob3dlist, picklefile)
        print('Plotting all generated blobs:')
        plotBlob2ds(Blob2d.all.values(), stitches=True)
    else:
        # HACK
        load_base = True # Note that each toggle dominates those below it due to elif
        load_bloomed = True
        dosave = True
        # HACK

        if load_base:
            blob3dlist = load(picklefile)
            newb3ds = bloom_b3ds(blob3dlist, stitch=False) # This will set pairings, and stitch if so desired
            if dosave:
                save(blob3dlist + newb3ds, picklefile + '_bloomed')
        elif load_bloomed:
            blob3dlist = load(picklefile + '_bloomed')
            newb3ds = bloom_b3ds(blob3dlist, stitch=True) # This will set pairings, and stitch if so desired
            if dosave:
                save(blob3dlist + newb3ds, picklefile + '_bloomed_stitched')
        else:
            blob3dlist = load(picklefile + '_bloomed_stitched')
        plotBlob2ds([blob2d for blob3d in blob3dlist for blob2d in blob3d.blob2ds],ids=False, parentlines=False,explode=True, coloring='blob3d',edge=False)
        for b3d in blob3dlist:
            print(b3d)
        exit()
    # plotBlob2ds(depth, stitches=True, ids=False, parentlines=False,explode=True, edge=False)



if __name__ == '__main__':
    main()  # Run the main function

