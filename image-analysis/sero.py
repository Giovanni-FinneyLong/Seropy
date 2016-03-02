__author__ = 'gio'
import pickle  # Note uses cPickle automatically ONLY IF python 3
import traceback
import sys
import time

from Slide import Slide
from Stitches import Pairing
from myconfig import Config
from util import print_elapsed_time
from util import ProgressBar, log, printl, printd  # Log is the actual object that will be shared between files


def save(blob3dlist, filename, directory=Config.PICKLEDIR):
    slash = ''
    if directory != '':
        if directory[-1] not in ['/', '\\']:
            slash = '/'
    filename = directory + slash + filename
    printl('\nSaving to file \'' + str(filename) + str('\''))
    done = False
    while not done:
        try:
            printl('Pickling ' + str(len(blob3dlist)) + ' b3ds ', end='')
            t0 = t = time.time()
            pickle.dump({'b3ds': Blob3d.all, 'possible_merges': Blob3d.possible_merges}, open(filename + '_b3ds', "wb"),
                        protocol=0)
            print_elapsed_time(t, time.time(), prefix='took')

            printl('Pickling ' + str(len(Blob2d.all)) + ' b2ds ', end='')
            t = time.time()
            pickle.dump({'b2ds': Blob2d.all, 'used_ids': Blob2d.used_ids}, open(filename + '_b2ds', "wb"), protocol=0)
            print_elapsed_time(t, time.time(), prefix='took')

            printl('Pickling ' + str(len(Pixel.all)) + ' pixels ', end='')
            t = time.time()
            pickle.dump({'pixels': Pixel.all, 'total_pixels': Pixel.total_pixels}, open(filename + '_pixels', "wb"),
                        protocol=0)
            print_elapsed_time(t, time.time(), prefix='took')
            done = True

            printl('Saving took:', end='')
            print_elapsed_time(t0, time.time(), prefix='')
        except RuntimeError:
            printl(
                '\nIf recursion depth has been exceeded, '
                'you may increase the maximal depth with: sys.setrecursionlimit(<newdepth>)')
            printl('The current max recursion depth is: ' + str(sys.getrecursionlimit()))
            printl(
                'Opening up an interactive console, press \'n\' then \'enter\' to load variables before interacting,'
                ' and enter \'exit\' to resume execution')
            debug()
            pass
    log.flush()


def load(filename, directory=Config.PICKLEDIR):
    if directory[-1] not in ['/', '\\']:
        slash = '/'
    else:
        slash = ''
    filename = directory + slash + filename
    t_start = time.time()
    printl('Loading from file \'' + str(filename) + str('\''))

    printl('Loading b3ds ', end='', flush=True)
    t = time.time()

    buff = pickle.load(open(filename + '_b3ds', "rb"))
    Blob3d.all = buff['b3ds']
    # if 'possible_merges' in buff:
    #     printl('Loading b3d possible merges')
    #     Blob3d.possible_merges = buff['possible_merges']
    # else:
    #     printl('Blob3d possible merges were not loaded as they were not in the save file')
    # This is a temp fix to work with old pickle files:
    if type(Blob3d.all) is list:
        buf = Blob3d.all
        Blob3d.all = dict()
        for b3d in buf:
            Blob3d.all[b3d.id] = b3d

    Blob3d.next_id = max(b3d.id for b3d in Blob3d.all.values()) + 1
    print_elapsed_time(t, time.time(), prefix='(' + str(len(Blob3d.all)) + ') took', flush=True)

    printl('Loading b2ds ', end='', flush=True)
    t = time.time()
    buff = pickle.load(open(filename + '_b2ds', "rb"))
    Blob2d.all = buff['b2ds']
    Blob2d.used_ids = buff['used_ids']
    Blob2d.total_blobs = len(Blob2d.all)
    print_elapsed_time(t, time.time(), prefix='(' + str(len(Blob2d.all)) + ') took', flush=True)

    printl('Loading pixels ', end='', flush=True)
    t = time.time()
    buff = pickle.load(open(filename + '_pixels', "rb"))
    Pixel.all = buff['pixels']
    Pixel.total_pixels = len(Pixel.all)
    print_elapsed_time(t, time.time(), prefix='(' + str(len(Pixel.all)) + ') took', flush=True)

    printl('Total time to load:', end='')
    print_elapsed_time(t_start, time.time(), prefix='')


def bloom_b3ds(blob3dlist, stitch=False, create_progress_bar=True):
    allb2ds = [Blob2d.get(b2d) for b3d in blob3dlist for b2d in b3d.blob2ds]
    printl('\nProcessing internals of ' + str(len(allb2ds)) + ' 2d blobs via \'blooming\' ', end='')
    t_start_bloom = time.time()
    num_unbloomed = len(allb2ds)
    pb = ProgressBar(max_val=sum(len(b2d.pixels) for b2d in allb2ds), increments=50)
    for bnum, blob2d in enumerate(allb2ds):  # HACK need to put the colon on the right of start_offset
        blob2d.gen_internal_blob2ds()  # NOTE will have len 0 if no blooming can be done
        pb.update(len(blob2d.pixels), set_val=False)  # set is false so that we add to an internal counter
    pb.finish()

    print_elapsed_time(t_start_bloom, time.time(), prefix='took')
    printl('Before blooming there were: ' + str(num_unbloomed) + ' b2ds contained within b3ds, there are now ' + str(
        len(Blob2d.all)))

    # Setting possible_partners
    printl('Pairing all new blob2ds with their potential partners in adjacent slides')
    max_avail_depth = max(
        b2d.recursive_depth for b2d in Blob2d.all.values())  # Note may want to adjust this later to do just some b2ds
    for cur_depth in range(max_avail_depth)[1:]:  # Skip those at depth 0
        depth = [b2d.id for b2d in Blob2d.all.values() if b2d.recursive_depth == cur_depth]
        max_h_d = max(Blob2d.all[b2d].height for b2d in depth)
        min_h_d = min(Blob2d.all[b2d].height for b2d in depth)
        ids_by_height = [[] for _ in range(max_h_d - min_h_d + 1)]
        for b2d in depth:
            ids_by_height[Blob2d.get(b2d).height - min_h_d].append(b2d)
        for height_val, h in enumerate(ids_by_height[:-1]):  # All but the last one
            for b2d in h:
                b2d = Blob2d.all[b2d]
                b2d.set_possible_partners(ids_by_height[height_val + 1])

    # Creating b3ds
    printl('Creating 3d blobs from the generated 2d blobs')
    all_new_b3ds = []
    for depth_offset in range(max_avail_depth + 1)[
                        1:]:  # Skip offset of zero, which refers to the b3ds which have already been stitched
        printd('Depth_offset: ' + str(depth_offset), Config.debug_blooming)
        new_b3ds = []

        for b3d in blob3dlist:
            all_d1_with_pp_in_this_b3d = []
            for b2d in b3d.blob2ds:
                # Note this is the alternative to storing b3dID with b2ds
                b2d = Blob2d.get(b2d)
                d_1 = [blob for blob in b2d.getdescendants() if
                       blob.recursive_depth == b2d.recursive_depth + depth_offset]
                if len(d_1):
                    for desc in d_1:
                        if len(desc.possible_partners):
                            all_d1_with_pp_in_this_b3d.append(desc.id)

            all_d1_with_pp_in_this_b3d = set(all_d1_with_pp_in_this_b3d)
            if len(all_d1_with_pp_in_this_b3d) != 0:
                printd(' Working on b3d: ' + str(b3d), Config.debug_blooming)
                printd('  Len of all_d1_with_pp: ' + str(len(all_d1_with_pp_in_this_b3d)), Config.debug_blooming)
                printd('  They are: ' + str(all_d1_with_pp_in_this_b3d), Config.debug_blooming)
                printd('   = ' + str(list(Blob2d.get(b2d) for b2d in all_d1_with_pp_in_this_b3d)),
                       Config.debug_blooming)
            for b2d in all_d1_with_pp_in_this_b3d:
                b2d = Blob2d.get(b2d)
                printd('    Working on b2d: ' + str(b2d) + ' with pp: ' + str(b2d.possible_partners),
                       Config.debug_blooming)
                if b2d.b3did == -1:  # unset
                    cur_matches = [b2d]  # NOTE THIS WAS CHANGED BY REMOVED .getdescendants() #HACK
                    for pp in b2d.possible_partners:
                        printd("     *Checking if pp:" + str(pp) + ' is in all_d1: ' + str(all_d1_with_pp_in_this_b3d),
                               Config.debug_blooming)
                        if pp in all_d1_with_pp_in_this_b3d:  # HACK REMOVED
                            printd("     Added partner: " + str(pp), Config.debug_blooming)
                            cur_matches += [Blob2d.get(b) for b in Blob2d.get(pp).getpartnerschain()]
                    if len(cur_matches) > 1:
                        printd("**LEN OF CUR_MATCHES MORE THAN 1", Config.debug_blooming)
                        new_b3d_list = [blob.id for blob in set(cur_matches) if
                                        blob.recursive_depth == b2d.recursive_depth and blob.b3did == -1]
                        if len(new_b3d_list):
                            new_b3ds.append(Blob3d(new_b3d_list, r_depth=b2d.recursive_depth))
        all_new_b3ds += new_b3ds
    printl(' Made a total of ' + str(len(all_new_b3ds)) + ' new b3ds')

    if stitch:
        # Set up shape contexts
        printl('Setting shape contexts for stitching')
        for b2d in [Blob2d.all[b2d] for b3d in all_new_b3ds for b2d in b3d.blob2ds]:
            b2d.set_shape_contexts(36)

        # Stitching
        printl('Stitching the newly generated 2d blobs')
        for b3d_num, b3d in enumerate(all_new_b3ds):
            printl(' Working on b3d: ' + str(b3d_num) + ' / ' + str(len(all_new_b3ds)))
            Pairing.stitch_blob2ds(b3d.blob2ds, debug=False)
    return all_new_b3ds


def main():
    printl('Current recusion limit: ' + str(sys.getrecursionlimit()) + ' updating to: ' + str(Config.recursion_limit))
    sys.setrecursionlimit(Config.recursion_limit)  # HACK
    if Config.test_instead_of_data:
        picklefile = 'All_test_pre_b3d_tree.pickle'  # THIS IS DONE *, and log distance base 2, now filtering on max_distance_cost of 3, max_pixels_to_stitch = 100
    else:
        if Config.swell_instead_of_c57bl6:
            picklefile = 'Swellshark_Adult_012615.pickle'
        else:
            picklefile = 'C57BL6_Adult_CerebralCortex.pickle'
    if not Config.dePickle:
        all_slides, blob3dlist = Slide.dataToSlides(stitch=Config.base_b3ds_with_stitching)
        # Reads in images and converts them to slides.
        # This process involves generating Pixels & Blob2ds & Blob3ds & Pairings
        printl("DB saving a rd0 copy!")
        save(blob3dlist, picklefile + '_rd0_only')
        log.flush()

        if Config.process_internals:
            bloomed_b3ds = bloom_b3ds(blob3dlist,stitch=Config.stitch_bloomed_b2ds) # Also sets partners + optionally stitching
            printl('Blooming resulted in ' + str(len(bloomed_b3ds)) + ' new b3ds:')
            blob3dlist = blob3dlist + bloomed_b3ds

        save(blob3dlist, picklefile)
        log.flush()
        plot_b2ds(list(Blob2d.all.values()), ids=False, stitches=True, edge=False, parentlines=Config.process_internals,
                  explode=Config.process_internals, pixel_ids=False)
        printl("Debug going to plot each blob2d individually:")
        for b2d in Blob2d.all.values():
            printl("B2d: " + str(b2d))
            plotBlob2d(b2d)

    else:
        # HACK
        load_base = False  # Note that each toggle dominates those below it due to elif
        # HACK

        if load_base:
            load(picklefile + '_rd0_only')
            blob3dlist = list(Blob3d.all.values())
            if Config.process_internals:
                bloomed_b3ds = bloom_b3ds(blob3dlist,
                                          stitch=Config.stitch_bloomed_b2ds)  # Includes setting partners, and optionally stitching
                printl('Blooming resulted in ' + str(len(bloomed_b3ds)) + ' new b3ds:')
                for b3d in bloomed_b3ds:
                    printl(b3d)
                blob3dlist = blob3dlist + bloomed_b3ds
        else:
            load(picklefile)
            blob3dlist = list(Blob3d.all.values())

        Blob3d.clean_b3ds()
        printl('Setting beads!')
        Blob3d.tag_all_beads()

        beads = list(b3d for b3d in Blob3d.all.values() if b3d.isBead)
        printl('Total number of beads: ' + str(len(beads)) + ' out of ' + str(len(Blob3d.all)) + ' total b3ds')
        plot_b2ds([b2d for b2d in Blob2d.all.values()], coloring='simple', ids=False, stitches=True, edge=True,
                  buffering=True, parentlines=True, explode=True)



        # printl('Plotting b3ds with plotly')
        # plot_plotly(blob3dlist)
        # printl('Plotting b2ds with plotly')
        # plot_plotly(list(Blob2d.all.values()), b2ds=True)
        printl('Plotting all simple:')
        plotBlob3ds(blob3dlist, color='simple')
        exit()


if __name__ == '__main__':
    try:
        if Config.mayPlot:
            from serodraw import *

            # global colors
            # global color_dict
            filter_available_colors()
        main()  # Run the main function
        log.close()
    except Exception as exc:
        printl("\nEXECUTION FAILED!\n")
        printl(traceback.format_exc())
        printl('Writing object to log')
        log.close()




        # TODO less b3ds after loading then when saving..?

        # NOTE: After updating blooming (2/28)
        # NOTE: Swell, stitched base, stitched blooming 2/28
        # Pickling 4123 b3ds took 11.19 seconds
        # Pickling 24253 b2ds took 21.77 seconds
        # Pickling 708062 pixels took 11.14 seconds
        # Saving took: 44.11 seconds
        # &
        # Loading b3ds (3829) took 4.32 seconds
        # Loading b2ds (24253) took 11.19 seconds
        # Loading pixels (708062) took 7.56 seconds
        # Total time to load: 23.08 seconds

        # NOTE: After updating blooming (2/28)
        # NOTE: C57BL6, stitched base, non-stitched blooming 2/28
        # Pickling 8293 b3ds took 27.58 seconds
        # Pickling 49891 b2ds took 26.82 seconds
        # Pickling 782067 pixels took 12.23 seconds
        # Saving took: 1 minute & 7 seconds
        # &
        # Loading b3ds (8209) took 9.49 seconds
        # Loading b2ds (49891) took 31.66 seconds
        # Loading pixels (782067) took 8.68 seconds
        # Total time to load: 49.83 seconds

        # TODO less b3ds after loading then when saving..?


        # NOTE: Post Stitch fix: (Also parallel run with the below)
        # NOTE: Swell, stitched base, non-stitched blooming 2/27
        # Pickling 8526 b3ds took 20.99 seconds
        # Pickling 24253 b2ds took 21.02 seconds
        # Pickling 708062 pixels took 11.31 seconds
        # Saving took: 53.32 seconds

        # NOTE: Post Stitch fix: (Also parallel run with the below)
        # NOTE: Swell, stitched base, stitched blooming 2/27
        # Pickling 8526 b3ds took 19.95 seconds
        # Pickling 24253 b2ds took 21.11 seconds
        # Pickling 708062 pixels took 11.50 seconds
        # Saving took: 52.57 seconds
        # &
        # Loading b3ds (8526) took 10.79 seconds
        # Loading b2ds (24253) took 40.17 seconds
        # Loading pixels (708062) took 9.72 seconds
        # Total time to load: 1 minute & 1 seconds

        # NOTE: Post Stitch fix: (Also parallel run with the above)
        # NOTE: C57BL6, stitched base, stitched blooming 2/27
        # Pickling 26514 b3ds took 39.91 seconds
        # Pickling 49891 b2ds took 50.86 seconds
        # Pickling 782067 pixels took 13.00 seconds
        # Saving took: 1 minute & 44 seconds

        # NOTE: C57BL6, stitched base, non-stitched blooming 2/22
        # Pickling 8294 b3ds took 55.75 seconds
        # Pickling 49891 b2ds took 1 minute & 1 seconds
        # Pickling 782067 pixels took 18.64 seconds
        # Saving took: 2 minutes & 15 seconds

        # NOTE: C57BL6, non-stitched base, non-blooming 2/22
        # Pickling 10118 b3ds took 0.26 seconds
        # Pickling 30815 b2ds took 1.39 seconds
        # Pickling 782067 pixels took 14.71 seconds
        # Saving took: 16.35 seconds

        # NOTE: Swell, stitched base, non-stitched blooming: 2/20
        # Loading b3ds (3851) took 15.63 seconds
        # Loading b2ds (26218) took 2 minutes & 49 seconds
        # Loading pixels (708062) took 12.96 seconds
        # Total time to load: 3 minutes & 18 seconds

        # NOTE: Swell, stitched base, non-stitched blooming: 1/25
        # Pickling 9606 b3ds took 27.79 seconds
        # Pickling 25347 b2ds took 26.07 seconds
        # Pickling 708062 pixels took 15.85 seconds
        # Saving took: 1 minute & 10 seconds

        # NOTE: C57BL6, stitched base, non-stitched blooming: 1/25
        # Pickling 25616 b3ds took 52.14 seconds
        # Pickling 50298 b2ds took 47.14 seconds
        # Pickling 782067 pixels took 15.58 seconds
        # Saving took: 1 minute & 55 seconds
