__author__ = 'gio'
import pickle  # Note uses cPickle automatically ONLY IF python 3
import traceback
import sys
import time
import pandas as pd

from Slide import Slide
from Stitches import Pairing
from myconfig import Config
from util import print_elapsed_time
from util import ProgressBar, log, printl, printd  # Log is the actual object that will be shared between files
from Blob3d import *

# NOTE ------------------------------------------------------------------------------
# This is the main file. Its execution is controlled by the parameters in myconfig.py
# NOTE ------------------------------------------------------------------------------


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
            pickle.dump({'b3ds': Blob3d.all, 'possible_merges': Blob3d.possible_merges,
                         "merged_b3ds":Blob3d.lists_of_merged_blob3ds, "merged_parent_b3ds":Blob3d.list_of_merged_blob3d_parents}, open(filename + '_b3ds', "wb"),
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
            printl('\nIf recursion depth has been exceeded, '
                   'you may increase the maximal depth with: sys.setrecursionlimit(<newdepth>)')
            printl('The current max recursion depth is: ' + str(sys.getrecursionlimit()))
            printl('Opening up an interactive console, press \'n\' then \'enter\' to load variables before interacting,'
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

    found_merged_b3ds = False
    if "merged_b3ds" in buff:
        Blob3d.lists_of_merged_blob3ds = buff["merged_b3ds"]
        found_merged_b3ds = True
    found_merged_parents = False
    if "merged_parent_b3ds" in buff:
        Blob3d.list_of_merged_blob3d_parents = buff["merged_parent_b3ds"]
        found_merged_parents = True

    Blob3d.next_id = max(b3d.id for b3d in Blob3d.all.values()) + 1
    print_elapsed_time(t, time.time(), prefix='(' + str(len(Blob3d.all)) + ') took', flush=True)
    if not found_merged_b3ds:
        print(" No lists of merged b3ds found, likely a legacy pickle file or small dataset")
    if not found_merged_parents:
        print(" No lists of merged b3 parents found, likely a legacy pickle file or small dataset")

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


def bloom_b3ds(blob3dlist, stitch=False):
    allb2ds = [Blob2d.get(b2d) for b3d in blob3dlist for b2d in b3d.blob2ds]
    printl('\nProcessing internals of ' + str(len(allb2ds)) + ' 2d blobs via \'blooming\' ', end='')
    t_start_bloom = time.time()
    num_unbloomed = len(allb2ds)
    pb = ProgressBar(max_val=sum(len(b2d.pixels) for b2d in allb2ds), increments=50)
    for bnum, blob2d in enumerate(allb2ds):
        blob2d.gen_internal_blob2ds()  # NOTE will have len 0 if no blooming can be done
        pb.update(len(blob2d.pixels), set_val=False)  # set is false so that we add to an internal counter
    pb.finish()

    print_elapsed_time(t_start_bloom, time.time(), prefix='took')
    printl('Before blooming there were: ' + str(num_unbloomed) + ' b2ds contained within b3ds, there are now ' + str(
        len(Blob2d.all)))

    # Setting possible_partners
    printl('Pairing all new blob2ds with their potential partners in adjacent slides')
    max_avail_depth = max(
        b2d.recursive_depth for b2d in Blob2d.all.values())
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


def do_stat_analysis():
    printl("Now performing statistical analysis...")
    b3d_count = len(Blob3d.all)
    base_b3ds = list(b3d for b3d in Blob3d.all.values() if b3d.recursive_depth == 0)
    beads = list(b3d for b3d in Blob3d.all.values() if b3d.isBead)  # TODO optimize

    beads_per_strand = []
    loose_beads = []  # These are beads that are solitary (not part of a strand)
    beads_in_strands = []
    strands = []

    for b3d in base_b3ds:  # TODO see if this conflicts with the current 'isBead' labeling
        buf = b3d.get_first_child_beads()
        num_children = len(buf)
        if num_children != 0:
            # Has bead children; is a strand
            if b3d.isBead:
                loose_beads.append(b3d)
            else:
                # Not a bead, so is a strand (since these are from base b3ds)
                if b3d.isBead:
                    print("WARNING adding b3d to strands, when isBead: " + str(b3d))
                strands.append(b3d)
                beads_per_strand.append(num_children)
                beads_in_strands += buf
        else:
            # No children, therefore implicitly a loose bead?
            if not b3d.isBead:
                print("WARNING adding b3d to loose beads, when not isBead: " + str(b3d))
            loose_beads.append(b3d)
    number_of_strands = len(beads_per_strand)
    printl('Total number of beads: ' + str(len(beads)) + ' out of ' + str(b3d_count) + ' total b3ds')
    printl('Total number of base b3ds: ' + str(len(base_b3ds)) + ' out of ' + str(b3d_count) + ' total b3ds')
    printl('Total number of loose beads: ' + str(len(loose_beads)) + ' out of ' + str(b3d_count) + ' total b3ds')
    printl('Total number of strands: ' + str(len(strands)) + ' out of ' + str(b3d_count) + ' total b3ds')

    plot_hist_xyz(base_b3ds)
    plot_hist_xyz(beads, type='All_Bead_b3ds')
    plot_hist_xyz(loose_beads, type='Loose_bead_b3ds')
    plot_hist_xyz(strands, type='Strand_b3ds')
    plot_hist_xyz(beads_in_strands, type='Beads_in_strand_b3ds')
    #
    plot_corr(base_b3ds)
    plot_corr(beads, type='All_Bead_b3ds')
    plot_corr(loose_beads, type='Loose_bead_b3ds')
    plot_corr(strands, type='Strand_b3ds')
    plot_corr(beads_in_strands, type='Beads_in_strand_b3ds')

    n1, bins1, patches1 = plt.hist(beads_per_strand, bins=max(beads_per_strand))
    plt.xlabel("Number of beads per strand")
    plt.ylabel("Number of b3ds")
    plt.title("Strand b3ds by number of beads")
    plt.tight_layout()
    plt.show()


def gen_skeleton(blob3d):
    #------------------------------------------------
    # NOTE OUTDATED, keeping only for reference
    #------------------------------------------------

    # Begin by creating a 3d array, with each element either None or the id of the pixel
    # Then create a second 3d array, with the distances from each internal point to the closest edge point
    # Find internals by doing all pixels - edge_pixels


    print("CALLED GEN_SKELETON!!!!")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xdim = blob3d.maxx - blob3d.minx + 1
    ydim = blob3d.maxy - blob3d.miny + 1
    zdim = blob3d.highslideheight - blob3d.lowslideheight + 1
    minx = blob3d.minx
    miny = blob3d.miny
    minz = blob3d.lowslideheight

    def pixel_to_pos(pixel):
        # Returns the (x,y,z) location of the pixel in any one of the 3d arrays
        return (pixel.x - minx, pixel.y - miny, pixel.z - minz)

    def distance_from_offset(x_offset, y_offset, z_offset):
        return math.sqrt(math.pow(x_offset, 2) + math.pow(y_offset, 2) + math.pow(z_offset, 2))

    def inBounds(x, y, z):
        if x >= 0 and y >= 0 and z >= 0 and x < xdim and y < ydim and z < zdim:
            return True
        return False

    def find_nearest_neighbor(x, y, z, arr, recur=1):
        """

        :param x: X coordinate within given arr
        :param y: Y coordinate within given arr
        :param z: Z coordinate within given arr
        :param arr: An array populated by the ids of pixels
        :param recur: How far outwards to extend the 'search cube'
        :return: (x_coor, y_coor, z_coor, distance, pixel_id)
        """
        # print("Finding nearest neighbor of: " + str((x, y, z, recur)))

        possible_coordinates = []  # Contains coordinate tuples (x,y,z)
        # Because am using cube instead of spheres, need to remember all matches and then find the best

        # arr should

        # TODO restrict the ranges so that they don't overlap

        # X restricts Y and Z
        # Y restricts Z
        # Z restricts nothing

        for x_offset in [-recur, recur]:
            curx = x + x_offset
            for y_offset in range(-recur, recur + 1):  # +1 to include endcap
                cury = y + y_offset
                for z_offset in range(-recur, recur + 1):  # +1 to include endcap
                    curz = z + z_offset

                    if inBounds(curx, cury, curz) and not np.isnan(arr[curx][cury][curz]):
                        possible_coordinates.append(
                            (curx, cury, curz, distance_from_offset(x_offset, y_offset, z_offset), int(arr[curx][cury][curz])))
                        # x, y, z, distance, pixel_id

        # print("Coordinates found: " + str(possible_coordinates))

        if len(possible_coordinates) == 0:
            # print("----Making a recursive call!")
            return find_nearest_neighbor(x, y, z, arr, recur + 1)

        # TODO Y and Z


        else:
            # Find the closest coordinate
            possible_coordinates.sort(key=lambda x_y_z_dist_id: x_y_z_dist_id[3])  # Sort by distance
            # print("SORTED POSSIBLE COORDINATES: " + str(possible_coordinates))
            return possible_coordinates[0]

    # TODO have an option to create a shell around the blob3d, by taking the highest and lowest levels, and making them all count temporarily as
    # edge pixels. This way, the effects of segmentation will be less dependent on the scan direction



    edge_pixels = blob3d.get_edge_pixels()  # Actual pixels not ids
    all_pixels = blob3d.get_pixels()
    inner_pixels = [Pixel.get(cur_pixel) for cur_pixel in (set(pixel.id for pixel in all_pixels) - set(pixel.id for pixel in edge_pixels))]

    edge_array = np.empty((xdim, ydim, zdim))  # , dtype=np.int)
    inner_array = np.empty((xdim, ydim, zdim))  # , dtype=np.int)
    distances = np.empty((xdim, ydim, zdim), dtype=np.float)
    edge_array[:] = np.nan
    inner_array[:] = np.nan
    distances[:] = np.nan

    for pixel in edge_pixels:
        x, y, z = pixel_to_pos(pixel)
        edge_array[x][y][z] = pixel.id

    for pixel in inner_pixels:
        x, y, z = pixel_to_pos(pixel)
        inner_array[x][y][z] = pixel.id

    inner_pos = np.zeros([len(inner_pixels), 3])
    near_pos = np.zeros([len(inner_pixels), 3])
    line_endpoints = np.zeros([2 * len(inner_pixels), 3])

    index_distance_id = np.zeros([len(inner_pixels), 3])

    maxdim = max([xdim, ydim, zdim])  # Adjusting z visualization

    pixel_nid_nx_ny_nz_dist = []
    marker_colors = np.zeros((len(inner_pixels), 4))
    mycolors = ['r', 'orange', 'yellow', 'white', 'lime', 'g', 'teal', 'blue', 'purple', 'pink']

    # HACK
    import vispy
    myrgba = list(vispy.color.ColorArray(color_str).rgba for color_str in mycolors)

    for index, pixel in enumerate(inner_pixels):
        # print("Pixel: " + str(pixel))
        x, y, z = pixel_to_pos(pixel)
        inner_pos[index] = x / xdim, y / ydim, z / zdim
        x_y_z_dist_id = find_nearest_neighbor(x, y, z, edge_array)

        pixel_nid_nx_ny_nz_dist.append((pixel, x_y_z_dist_id[4], x_y_z_dist_id[0], x_y_z_dist_id[1], x_y_z_dist_id[2], x_y_z_dist_id[3]))

        near_pos[index] = (x_y_z_dist_id[0] / xdim, x_y_z_dist_id[1] / ydim, x_y_z_dist_id[2] / zdim)
        marker_colors[index] = myrgba[int(x_y_z_dist_id[3]) % len(myrgba)]

        line_endpoints[2 * index] = (x_y_z_dist_id[0] / xdim, x_y_z_dist_id[1] / ydim, x_y_z_dist_id[2] / zdim)
        line_endpoints[2 * index + 1] = (x / xdim, y / ydim, z / zdim)
        # index_distance_id[index] = [index, x_y_z_dist_id[3], x_y_z_dist_id[4]]

    # Sort distances; need to maintain index for accessing id later
    pixel_nid_nx_ny_nz_dist = sorted(pixel_nid_nx_ny_nz_dist, key=lambda entry: entry[5], reverse=True)

    # TODO find ridge
    # Strategy: Iterative, use either the highest n or x percent to find cutoff..?
    ridge = [pixel_nid_nx_ny_nz_dist[0]]
    ridge_ends = [pixel_nid_nx_ny_nz_dist[0]]  # Preferable to keep at 2 if we can

    # HACK TODO
    for i in pixel_nid_nx_ny_nz_dist:
        print(' ' + str(i))
    # min_threshold = int(pixel_nid_nx_ny_nz_dist[int(.1 * len(pixel_nid_nx_ny_nz_dist))])

    # pixel_costs = dict()
    # for pnnnnd in pixel_nid_nx_ny_nz_dist:
    #     pixel_costs[pnnnnd[0].id] = pnnnnd[5] # Mapping inner pixel's id to its distance (to the closest edge_pixel)
    #
    #
    #
    #
    #
    # n, bins, patches = plt.hist([idi[5] for idi in pixel_nid_nx_ny_nz_dist], bins=np.linspace(0,10,50))
    # # hist = np.histogram( [idi[4] for idi in pixel_nid_nx_ny_nz_dist] , bins=np.arange(10))
    # plt.show()


    import vispy.io
    import vispy.scene
    from vispy.scene import visuals
    from vispy.util import keys

    canvas = vispy.scene.SceneCanvas(size=(1200, 1200), keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    view.camera = vispy.scene.cameras.FlyCamera(parent=view.scene, fov=30, name='Fly')
    # view.camera = vispy.scene.cameras.TurntableCamera(fov=0, azimuth=80, parent=view.scene, distance=1,
    #                                                   elevation=-55, name='Turntable')
    # view.camera = vispy.scene.cameras.ArcballCamera(parent=view.scene, fov=50, distance=1,
    #                                                    name='Arcball')
    inner_markers = visuals.Markers()
    near_markers = visuals.Markers()
    lines = visuals.Line(method=Config.linemethod)
    lines.set_data(pos=line_endpoints, connect='segments', color='y')

    inner_markers.set_data(inner_pos, face_color=marker_colors, size=10)
    near_markers.set_data(near_pos, face_color='g', size=10)
    print("Inner pos: " + str(inner_pos))
    print("Near pos: " + str(near_pos))

    view.add(inner_markers)
    view.add(near_markers)
    view.add(lines)
    vispy.app.run()

    print('------------')

    # print("\n\nEP: " + str(len(edge_pixels)))
    # print("Pix: " + str(len(all_pixels)))
    # print("Inner pix: " + str(len(inner_pixels)))
    # print("Children: " + str(self.children))
    # for color_index, pix_list in enumerate([edge_pixels, inner_pixels]):
    #     cur_color = ['r', 'g', 'b'][color_index]
    #     xs = [0] * len(pix_list)
    #     ys = [0] * len(pix_list)
    #     zs = [0] * len(pix_list)
    #
    #     for index, pixel in enumerate(pix_list):
    #         # print(pixel)
    #         # print('  ' + str(pixel_to_pos(pixel)))
    #         x, y, z =  pixel_to_pos(pixel)
    #         edge_array[x][y][z] = pixel.id
    #         xs[index] = x
    #         ys[index] = y
    #         zs[index] = z
    #     ax.scatter(xs, ys, zs, c=cur_color)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()


def gen_skeleton_new(blob3d):
    """
    New attempt at generating a skeleton
    :param blob3d:
    :return:
    """
    arr = blob3d.to_array()
    non_zero_positions = np.where(arr)
    non_zero_positions = [i.tolist() for i in np.dstack((non_zero_positions[0], non_zero_positions[1]))][0]
    plot_array(arr)


def main():
    # printl('Current recusion limit: ' + str(sys.getrecursionlimit()) + ' updating to: ' + str(Config.recursion_limit))
    # sys.setrecursionlimit(Config.recursion_limit)  # HACK
    if Config.test_instead_of_data:
        picklefile = 'All_test_pre_b3d_tree.pickle'
    else:
        picklefile = Config.PICKLE_FILE_PREFIX + ".pickle"
    if not Config.dePickle:
        all_slides, blob3dlist = Slide.dataToSlides(stitch=Config.base_b3ds_with_stitching)
        # Reads in images and converts them to slides.
        # This process involves generating Pixels & Blob2ds & Blob3ds & Pairings
        printl("Saving a 'recursive depth 0' (rd0) copy")
        save(blob3dlist, picklefile + '_rd0_only')
        log.flush()
        if Config.process_internals:
            bloomed_b3ds = bloom_b3ds(blob3dlist, stitch=Config.stitch_bloomed_b2ds) # Also sets partners + optionally stitching
            printl('Blooming resulted in ' + str(len(bloomed_b3ds)) + ' new b3ds:')
            blob3dlist = blob3dlist + bloomed_b3ds

        save(blob3dlist, picklefile)
        log.flush()

        # print("\n\nThere were a total of %s lists of merged b3d lists" % (len(Blob3d.lists_of_merged_blob3ds)))
        # for index, sublist in enumerate(Blob3d.lists_of_merged_blob3ds):
        #     print("Index:%s, sublist_len:%s, sublist_ids:%s, sublist:%s" % (index, len(sublist), [b3d.id for b3d in sublist], sublist))
        #     plot(sublist)

    else:
        if Config.load_base_only:
            load(picklefile + '_rd0_only')
            blob3dlist = list(Blob3d.all.values())
            if Config.process_internals:
                bloomed_b3ds = bloom_b3ds(blob3dlist, stitch=Config.stitch_bloomed_b2ds)
                # Includes setting partners, and optionally stitching
                printl('Blooming resulted in ' + str(len(bloomed_b3ds)) + ' new b3ds:')
                for b3d in bloomed_b3ds:
                    printl(b3d)
                blob3dlist = blob3dlist + bloomed_b3ds
        else:
            load(picklefile)
            blob3dlist = list(Blob3d.all.values())

    Blob3d.clean_b3ds()  # This is for safety - prioritizing successful execution to perfectly correct data
    printl('Setting beads!')
    Blob3d.tag_all_beads()

    # largest_base_b3ds = sorted(list(blob3d for blob3d in Blob3d.all.values() if blob3d.recursive_depth == 0),
    #                       key=lambda b3d: b3d.get_edge_pixel_count(), reverse=True)  # Do by recursive depth

    for b3d in blob3dlist:
        print(b3d)
        # plot([b3d], ids=False)
        gen_skeleton_new(b3d)

        print("P:" + str(b3d.get_pixels()))
        print("EP:" + str(b3d.get_edge_pixels()))
        # for child in b3d.children:
        #     print(' ' + str(Blob3d.get(child)))




    plot(blob3dlist, ids=False, stitches=True, buffering=True, parentlines=True, explode=True, show_debug_colors=True)



        # TODO Calculate and plot different statistics about the data.
        # Good examples are:
        #   Total number of b3ds, distribution of number of pixels in blob3ds
        #   Density over the 3d volume over the scans, as a density map and as 3 histograms, for:
        #         Total beads, singular beads,
        #   Average number of beads per strand
        #
    # do_stat_analysis()
    exit()


    '''
    for blob3d in largest_base_b3ds:
        printl(blob3d)
        plot_b3ds([blob3d])
        blob3d.gen_skeleton()
        # plot_b3ds([blob3d], color='simple')

    # printl('Plotting b3ds with plotly')
    # plot_plotly(blob3dlist)
    # printl('Plotting b2ds with plotly')
    # plot_plotly(list(Blob2d.all.values()), b2ds=True)
    printl('Plotting all simple:')
    plot_b3ds(blob3dlist, color='simple')
    '''


if __name__ == '__main__':
    try:
        if Config.mayPlot:
            from serodraw import *
            filter_available_colors()
        main()  # Loads or generates blobs, displays in 3d, then displays visual stats
        log.close()
    except Exception as exc:
        printl("\nEXECUTION FAILED!\n")
        printl(traceback.format_exc())
        printl('Writing object to log')
        log.close()


