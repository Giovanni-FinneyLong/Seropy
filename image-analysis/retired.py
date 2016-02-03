__author__ = 'gio'
'''
This file contains functions which have been removed from sero.py and serodraw.py but may still have some use down the road.
Any functions here are not in use within either sero or serodraw.
'''
import serodraw
import sero
import numpy as np
import matplotlib.pyplot as plt
import vispy.visuals



# SERO Functions:
def hungarianCompare(blob1, blob2):
    '''
    Uses the Hungarian Alg to find the optimal pairings of points between two Blob2D's
    '''
    # Each blob has its shape contexts (context_bins) set by here. Now for each point in b1 there is a row, and each in b2 a column.
    # One blob may have more points than the other, and so rows OR columns containing the max value of the matrix are added so that the matrix is nxn
    # TODO create a nested function to calculate cost between 2 points based on their histograms (contained in context_bins[point#)

    global debugval

    def costBetweenPoints(bins1, bins2):
        assert len(bins1) == len(bins2)
        cost = 0
        for i in range(len(bins1)):
            debug_cost = cost
            if (bins1[i] + bins2[i]) != 0:
                cost += math.pow(bins1[i] - bins2[i], 2) / (bins1[i] + bins2[i])
            # if math.isnan(cost) and not math.isnan(debug_cost):
            #     print('Became nan, old val=' + str(debug_cost) + ' Pow part:' + str(math.pow(bins1[i] - bins2[i], 2)) + ' denomerator:' + str(bins1[i] + bins2[i]))
            #     print(' bins1:' + str(bins1[i]) + ' bins2:' + str(bins2[i]))
            #     buf = bins1[i] - bins2[i]
            #     print(' Buf=' + str(buf) + ' pow:' + str(math.pow(bins1[i] - bins2[i], 2)))
        return cost / 2

    def printCostArrayLined(cost_array, row_lines, col_lines):
        print('**Lined Cost Array=')
        ndim = len(cost_array)
        for r in range(ndim):
            print('[', end='')
            for c in range(ndim):
                buf = []
                if r in row_lines:
                    buf.append('r')
                if c in col_lines:
                    buf.append('c')
                if len(buf) == 2:
                    print('+ ', end='')
                elif len(buf) == 1:
                    if 'r' in buf:
                        print('- ', end='')
                    else:
                        print('| ', end='')
                else:
                    print(cost_array[r][c], end=' ')
            print(']')
    def oldLinesMethod():
        # if row_zeros[most_zeros_row] > col_zeros[most_zeros_col]:
        #     # Set a 'line' through a row
        #     lines_used += 1
        #     zeros_covered += row_zeros[most_zeros_row]
        #     row_lines.append(most_zeros_row)
        #     row_zeros[most_zeros_row] = 0
        #     for col in range(ndim): # Updating the number of zeros in each column, as we have just removed some by creating the line
        #         if cost_array[most_zeros_row][col] == 0 and col not in col_lines:
        #             col_zeros[col] -= 1
        #             # DEBUG
        #             if col_zeros[col] < 0:
        #                 print('Error!!!! lt zero')
        #                 debug()
        # else:
        #     lines_used += 1
        #     zeros_covered += col_zeros[most_zeros_col]
        #     col_lines.append(most_zeros_col)
        #     col_zeros[most_zeros_col] = 0
        #     for row in range(ndim):
        #         if cost_array[row][most_zeros_col] == 0 and row not in row_lines:
        #             row_zeros[row] -= 1
        #             # DEBUG

        #             if row_zeros[row] < 0:
        #                 print('Error!!! lt zero')
        #                 debug()
        1



    ndim = max(len(blob1.edge_pixels), len(blob2.edge_pixels))
    cost_array = np.zeros([ndim, ndim])

    ''' # TEMP for DEBUG

    for i in range(len(blob1.edge_pixels)):
        for j in range(len(blob2.edge_pixels)):
            cost_array[i][j] = costBetweenPoints(blob1.context_bins[i], blob2.context_bins[j])
    i = len(blob1.context_bins)
    j = len(blob2.context_bins)
    print('i=' + str(i) + ' j=' + str(j))
    if i != j:
        # Need to find max value currently in the array, and use it to add rows or cols so that the matrix is square
        maxcost = np.amax(cost_array)
        if i < j:
            for r in range(i,j):
                for s in range(j):
                    cost_array[r][s] = maxcost # By convention
        else:
            for r in range(j,i):
                for s in range(i):
                    cost_array[r][s] = maxcost # By convention
    # Here the cost array is completely filled.
    # TODO run this on some simple examples provided online to confirm it is accurate.

    '''

    #HACK DEBUG
    cost_array = np.array([[10,19,8,15,19], # Note that after 10 major iterations, the generated matrix is almost exactly the same as the final matrix (which can be lined), the only difference is that the twos in the
                                            #Middle column should be ones # FIXME THIS FAILS! After the addition of the not, which removed the alternating placement of lines, this uses 5 vertical instead of 4 mixed lines, resulting in finishing early
                  [10,18,7,17,19],
                  [13,16,9,14,19],
                  [12,19,8,18,19],
                  [14,17,10,19,19]])


    # cost_array = np.array([ # This is from the prime/star method (for doing by hand) # Note SUCCESS!
    #     [1,2,3],
    #     [2,4,6],
    #     [3,6,9]
    # ])
    cost_array = np.array([ # From http://www.math.harvard.edu/archive/20_spring_05/handouts/assignment_overheads.pdf # NOTE SUCCESS!
                            # NOTE reduces correctly in 1 major iteration and 2 minor
                            # DEBUG works with Munkres
        [250,400,350],
        [400,600,350],
        [200,400,250]
    ])
    # cost_array = np.array([ # From http://www.math.harvard.edu/archive/20_spring_05/handouts/assignment_overheads.pdf # NOTE SUCCESS!
    #     [90,75,75,80],   # DEBUG works with Munkres
    #     [35,85,55,65],
    #     [125,95,90,105],
    #     [45,110,95,115]
    # ])
    #HACK HACK
    wiki_not_harvard = False # NOTE If true, use the method from http://www.wikihow.com/Use-the-Hungarian-Algorithm else use method from: http://www.math.harvard.edu/archive/20_spring_05/handouts/assignment_overheads.pdf

    # cost_array = np.array([
    #     [0,1,0,0,5],
    #     [1,0,3,4,5],
    #     [7,0,0,4,5],
    #     [9,0,3,4,5],
    #     [3,0,3,4,5]
    # ])



    '''



    ndim = len(cost_array)
    original_cost_array = np.copy(cost_array)
    #HACK
    print('NDIM=' + str(ndim))
    print('Starting cost_array=\n' + str(cost_array)) # DEBUG



    # First subtract the min of each row from that row
    row_mins = np.amin(cost_array, axis=1) # An array where the nth element is the largest number in the nth row of cost_array
    print('Row mins found to be:' + str(row_mins))
    for row in range(len(cost_array)):
        cost_array[row] -= row_mins[row]
    print('After min per row subtracted cost_array=\n' + str(cost_array)) # DEBUG

    # Now if there are any cols without a zero, subtract the min of that column from the entire column (therefore creating a zero)
    # This is the equivalent of subtracting the min of a column by itself in all cases, as all values are non-negative
    col_mins = np.amin(cost_array, axis=0) # An array where the nth element is the largest number in the nth column of cost_array
    cost_array -= col_mins
    print('After min per col subtracted cost_array=\n' + str(cost_array)) # DEBUG


    # Now cover all zero elements with a minimal number of vertical/horizontal lines
    # Maintain a list of the number of zeros in each row/col




    iteration = 0

    lines_used = 0
    while lines_used != ndim:

        col_zeros = np.zeros([ndim])
        row_zeros = np.zeros([ndim])
        for row in range(ndim):
            for col in range(ndim):
                if not cost_array[row][col]:
                    col_zeros[col] += 1
                    row_zeros[row] += 1
        print('============================================\nIteration #' + str(iteration))

        # print('DB col_zeros:' + str(col_zeros))
        # print('DB row_zeros:' + str(row_zeros))
        total_zeros = sum(col_zeros) #len(col_zeros) + len(row_zeros)
        print('DB total_zeros=' + str(total_zeros))

        if iteration > 25: # DEBUG
            debug()
        lines_used = 0
        zeros_covered = 0
        row_lines = [] # Holds the indeces of lines drawn through rows
        col_lines = [] # Holds the indeces of lines drawn through columns
        last_line_horizontal = None # To be T/F
        next_line_horizontal = None
        last_line_index = -1
        next_line_index = -1

        print('About to start setting lines, total zeros=' + str(total_zeros) + ' zeros_covered=' + str(zeros_covered))
        # Now start setting the lines
        line_drawing_iteration = -1 # DEBUG
        while total_zeros != zeros_covered:
            line_drawing_iteration += 1
            # print(' Setting lines iteration #' + str(line_drawing_iteration) + ' zeros_covered/total_zeros=' + str(zeros_covered) + '/' + str(total_zeros))
            # print(' Cost_array=\n' + str(cost_array))
            # printCostArrayLined(cost_array, row_lines, col_lines)

            # print(' Col_zeros:' + str(col_zeros))
            # print(' Row_zeros:' + str(row_zeros))
            # print(' RowLines:' + str(row_lines))
            # print(' ColLines:' + str(col_lines))
            most_zeros_row = np.argmax(row_zeros) # An index not a value
            most_zeros_col = np.argmax(col_zeros) # An index not a value
            # print(' Most zeros (r,c) = (' + str(most_zeros_row) + ',' + str(most_zeros_col) + ')')
            if line_drawing_iteration == 6:
                print('hit 6 iterations, debugging')
                debug()

            max_covered = -1
            next_line_index = -1
            max_covered_r = -1
            max_covered_c = -1
            next_line_index_c = -1
            next_line_index_r = -1
            for r in range(ndim):# http://stackoverflow.com/questions/14795111/hungarian-algorithm-how-to-cover-0-elements-with-minimum-lines
                if(row_zeros[r] > max_covered_r or (row_zeros[r] == max_covered_r and last_line_horizontal == True)):
                    next_line_index_r = r
                    next_line_horizontal = True
                    max_covered_r = row_zeros[r]
            for c in range(ndim):
                if(col_zeros[c] > max_covered_c or (col_zeros[c] == max_covered_c and last_line_horizontal == False)):
                    next_line_index_c = c
                    next_line_horizontal = False
                    max_covered_c = col_zeros[c]
            # TODO fix the above by making it so that there is a preference for vertical if just did horizontal and vice versa.
            # Should involve separate max counters for each direction (already have in most_zeros_row/col)
            # So just need max_covered both verticall and horizontally, then compare and then set.
            print(' MAX COVERED R/C=' + str(max_covered_r) + ', ' + str(max_covered_c))

            if max_covered_r == max_covered_c:
                if last_line_horizontal: # Prefer column
                    next_line_index = next_line_index_c
                    next_line_horizontal = False
                    max_covered = max_covered_c
                else:
                    next_line_index = next_line_index_r
                    next_line_horizontal = True
                    max_covered = max_covered_r
            else:
                if max_covered_r > max_covered_c:
                    next_line_index = next_line_index_r
                    next_line_horizontal = True
                else:
                    next_line_index = next_line_index_c
                    next_line_horizontal = False
                max_covered = max(max_covered_c, max_covered_r)
                #TODO set max_covered, and next_line_index and next_line_horizontal

            # TODO now is done setting line early when it should be, although looks to be making progress, as drew the horizontal line at pos 2 after the vertical at pos 2
            # NOTE Current issue is that continue adding lines even when all elements are covered correctly
            # DEBUG NOTE STEP 5 of the online slides says to subtract from each uncovered row and add to each covered column,
            #       whereas the wikihow says subtract the min element from every element in the matrix
            print(' Max_covered_r=' + str(max_covered_r) + ', Max_covered_c=' + str(max_covered_c))
            print(' After iterating r/c found the best line index to be:' + str(next_line_index) + ' and next_line_horizontal=' + str(next_line_horizontal) + ', max_covered=' + str(max_covered))
            if next_line_horizontal:
                row_zeros[next_line_index] = 0
                for c in range(ndim):
                    if cost_array[next_line_index][c] == 0:
                        col_zeros[c] -= 1
                row_lines.append(next_line_index)
            else:
                col_zeros[next_line_index] = 0
                for r in range(ndim):
                    if cost_array[r][next_line_index] == 0:
                        row_zeros[r] -= 1
                col_lines.append(next_line_index)
            zeros_covered += max_covered
            last_line_horizontal = next_line_horizontal
            last_line_index = next_line_index


            if total_zeros < zeros_covered:
                print('Error, too many zeros covered')
                debug()

        lines_used = len(col_lines) + len(row_lines)
        print('DONE SETTING LINES to cover zeros, next find min uncovered element, lines_used=' + str(lines_used))
        printCostArrayLined(cost_array, row_lines, col_lines)
        # print('RowLines:' + str(row_lines))
        # print('ColLines:' + str(col_lines))
        print('Cost_array:\n' + str(cost_array))

        # Now find the minimal UNCOVERED element, and add it to every COVERED element
        if lines_used != ndim: # Can skip this if we've already got all the lines we need (ndim)
            min_uncovered = np.amax(cost_array)
            for row in range(ndim):
                for col in range(ndim):
                    if row not in row_lines and col not in col_lines:
                        min_uncovered = min(min_uncovered, cost_array[row][col])
            print('The min_uncovered value is:' + str(min_uncovered))


            #HACK DEBUG
            if wiki_not_harvard:
                # Now add the min_uncovered to the COVERED elements
                # Note that if an element is covered twice, we add the minimal element twice
                for row in range(ndim): # TODO this could be a bit more efficient by subtracting from the whole row/col at once
                    for col in range(ndim):
                        if row in row_lines:
                            cost_array[row][col] += min_uncovered
                        if col in col_lines:
                            cost_array[row][col] += min_uncovered
                print('After increasing ONLY covered by min uncovered, cost_array=\n' + str(cost_array))

                # Now subtract the minimal element from every element in the matrix
                arr_min = np.amin(cost_array) # This can never be zero, as all zeros had the minimal uncovered value added to them
                print('Minimal value of all elements=' + str(arr_min))
                # DEBUG
                if not arr_min:
                    print('Error, contained a zero value after all zeros were added to')
                    debug()
                cost_array -= arr_min
                print('Cost_array after subtracting min_element:\n' + str(cost_array))
                # now re-cover the zero elements
            else:
                # NOTE this is the harvard method, found here: http://www.math.harvard.edu/archive/20_spring_05/handouts/assignment_overheads.pdf
                # "Determine the smallest entry not covered by any line. Subtract this entry from each uncovered row, and then add it to each covered column.
                # print('USING HARVARD METHOD!')
                for row in range(ndim):
                    if row not in row_lines:
                        for col in range(ndim):
                            cost_array[row][col] -= min_uncovered
                print('Cost array after subtracting smallest entry from each UNCOVERED row:\n' + str(cost_array))
                for col in range(ndim):
                    if col in col_lines:
                        for row in range(ndim):
                            cost_array[row][col] += min_uncovered
                print('Cost array after adding smallest entry to each COVERED col:\n' + str(cost_array))
            iteration += 1



        else:
            print('DB SKIPPED SECOND PART AS NUM LINES ALREADY PERFECT')


    print('DB Done, now find a cover such that each row or column only has one zero selected ')
    print('Original cost array\n' + str(original_cost_array))
    print('Current cost array\n' + str(cost_array))
    printCostArrayLined(cost_array, row_lines, col_lines)
    print('---SUCCESSFULLY COMPLETED hungarian method')


    '''



    # if debugval == 1:
    #     debugval = 0
    #     for row in range(len(cost_array)):
    #         for col in range(len(cost_array[row])):
    #             print(str(cost_array[row][col]) + ', ', end='')
    #         print('')

    # print('Cost array = ' + str(cost_array))
    return 1


def getIdArrays(pixels, id_counts):
    '''
    Returns a list of filled arrays, each of which corresponds to an id. If remapped, the first array is most dense
    '''
    id_arrays = [zeros([xdim, ydim]) for _ in range(len(id_counts))]  # Each entry is an (r,c) array, filled only with the maximal values from the corresponding

    if remap_ids_by_group_size:
        remap = [None] * len(id_counts)
        for id in range(len(id_counts)): # Supposedly up to 2.5x faster than using numpy's .tolist()
            remap[id_counts[id][0]] = id
        for pixel in pixels:
            id_arrays[remap[pixel.blob_id]][pixel.x][pixel.y] = int(pixel.val)
    else:
        for pixel in pixels:
            if pixel.blob_id >= id_counts:
                print('DEBUG: About to fail:' + str(pixel))
            id_arrays[pixel.blob_id][pixel.x][pixel.y] = int(pixel.val)
    return id_arrays


def KMeansClusterIntoLists(listin, num_clusters):

    def doClustering(array_in, num_clusters):
        'Take an array of tuples and returns a list of lists, each of which contains all the pixels of a cluster'
        cluster_lists = [[] for i in range(num_clusters)]
        (bookC, distortionC) = kmeans(array_in, num_clusters)
        (centLabels, centroids) = vq(array_in, bookC)
        for pixlabel in range(len(centLabels)):
            cluster = centLabels[pixlabel]
            # pix = max_pixel_array_floats[pixlabel]
            # cluster_arrays[cluster][pix[1]][pix[2]] = pix[0]
            cluster_lists[cluster].append(array_in[pixlabel])
        return cluster_lists

    max_tuples_as_arrays = np.asarray([(float(pixel.val), float(pixel.x), float(pixel.y)) for pixel in listin])
    # NOTE: Is an array of shape (#pixels, 3), where each element is an array representing a tuple.
    # NOTE: This is the required format for kmeans/vq
    tuple_array = np.asarray([(float(pixel.val), float(pixel.x), float(pixel.y)) for pixel in listin])
    return doClustering(tuple_array, num_clusters)







# SERODRAW Functions:

def plotPixels(pixellist, canvas_size=(800, 800)):
    canvas = Canvas(canvas_size)
    xmin = min(pixel.x for pixel in pixellist)
    ymin = min(pixel.y for pixel in pixellist)
    edge_pixel_array = np.zeros([len(pixellist), 3])
    for (p_num, pixel) in enumerate(pixellist):
        edge_pixel_array[p_num] = [(pixel.x - xmin) / len(pixellist), (pixel.y - ymin) / len(pixellist), pixel.z /  (config.z_compression * len(pixellist))]
    marker = visuals.Markers()
    marker.set_data(edge_pixel_array, edge_color=None, face_color=colors[0 % len(colors)], size=8)
    canvas.view.add(marker)
    vispy.app.run()

def plotPixelLists(pixellists, canvas_size=(800, 800)): # NOTE works well to show bloom results
    canvas = Canvas(canvas_size)
    xmin = min(pixel.x for pixellist in pixellists for pixel in pixellist)
    ymin = min(pixel.y for pixellist in pixellists for pixel in pixellist)
    xmax = max(pixel.x for pixellist in pixellists for pixel in pixellist)
    ymax = max(pixel.y for pixellist in pixellists for pixel in pixellist)
    zmin = min(pixel.z for pixellist in pixellists for pixel in pixellist)
    zmax = max(pixel.z for pixellist in pixellists for pixel in pixellist)
    xdim = xmax - xmin + 1
    ydim = ymax - ymin + 1
    zdim = zmax - zmin + 1

    edge_pixel_arrays = []
    # TODO plot all of a color at once


    markers_per_color = [0 for i in range(min(len(colors), len(pixellists)))]
    offsets = [0] * min(len(colors), len(pixellists))
    for blobnum, pixellist in enumerate(pixellists):
        markers_per_color[blobnum % len(markers_per_color)] += len(pixellist)
    for num,i in enumerate(markers_per_color):
        edge_pixel_arrays.append(np.zeros([i, 3]))
    for blobnum, pixellist in enumerate(pixellists):
        index = blobnum % len(markers_per_color)
        for p_num, pixel in enumerate(pixellist):
            edge_pixel_arrays[index][p_num + offsets[index]] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( config.z_compression * zdim)]
        offsets[index] += len(pixellist)

    print('NUM ARRAYS=' + str(len(edge_pixel_arrays)))
    for color_num, edge_array in enumerate(edge_pixel_arrays):
        markers = visuals.Markers()
        markers.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
        # view.add(visuals.Markers(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 ))
        canvas.view.add(markers)
    vispy.app.run()

def contrastSaturatedBlob2ds(blob2ds, minimal_edge_pixels=350):
    '''
    Used to view each blob2d with a threshold number of edge_pixels of a blob3d,
    before and after saturating the outside, with and without normalization.
    :param blob2ds: A list of blob2ds, normally from a single blob3d, which will be experimentally saturated and normalized.
    :param minimal_edge_pixels:
    :return:
    '''
    import matplotlib.pylab as plt
    from sklearn.preprocessing import normalize

    for b2d_num, blob2d in enumerate(blob2ds):
        print('Start on blob2d: ' + str(b2d_num) + ' / ' + str(len(blob2ds)) + ' which has ' + str(len(blob2d.edge_pixels)) + ' edge_pixels')
        if len(blob2d.edge_pixels) > minimal_edge_pixels: # using edge to emphasize skinny or spotty blob2d's
            before = blob2d.edgeToArray()
            saturated = blob2d.gen_saturated_array()
            normal_before = normalize(before)
            normal_saturated = normalize(saturated)
            xx, yy = saturated.shape
            print(' array dim xx,yy: ' + str(xx) + ',' + str(yy))
            fig, axes = plt.subplots(2,2, figsize=(12,12))
            for img_num, ax in enumerate(axes.flat):
                print('>>DB img_num:' + str(img_num))
                ax.set_xticks([])
                ax.set_yticks([])
                if img_num == 0:
                    ax.imshow(before, interpolation='nearest', cmap=plt.cm.jet)
                elif img_num == 1:
                    ax.imshow(saturated, interpolation='nearest', cmap=plt.cm.jet)
                elif img_num == 2:
                    ax.imshow(normal_before, interpolation='nearest', cmap=plt.cm.jet)
                elif img_num == 3:
                    ax.imshow(normal_saturated, interpolation='nearest', cmap=plt.cm.jet)
            plt.show()
        else:
            print('Skipping, as blob2d had only: ' + str(len(blob2d.edge_pixels)) + ' edge_pixels')

def showSlide(slide):
    import matplotlib.pylab as plt
    if len(slide.alive_pixels) > 0:
        maxx = max(b2d.maxx for b2d in slide.blob2dlist)
        maxy = max(b2d.maxy for b2d in slide.blob2dlist)
        minx = min(b2d.minx for b2d in slide.blob2dlist)
        miny = min(b2d.miny for b2d in slide.blob2dlist)
        array = np.zeros([maxx - minx + 1, maxy - miny + 1])
        for pixel in slide.alive_pixels:
            array[pixel.x - minx][pixel.y - miny] = pixel.val
        plt.imshow(array, cmap='rainbow', interpolation='none')
        # plt.matshow(array)
        plt.show()
    else:
        print('Cannot show slide with no pixels:' + str(slide))

def showBlob2d(b2d):
    import matplotlib.pylab as plt
    width = b2d.maxx - b2d.minx + 1
    height = b2d.maxy - b2d.miny + 1
    array = np.zeros([width, height])
    for pixel in b2d.pixels:
        array[pixel.x - b2d.minx][pixel.y - b2d.miny] = pixel.val
    plt.imshow(array, cmap='rainbow', interpolation='none')
    plt.colorbar()
    plt.show()




def PlotHist(listin, numBins):
    'Take a 1dimensional matrix or a list'
    plt.hist(listin, bins=numBins)
    plt.show()


def PlotCounterHist(counter, numBins):
    PlotHist(list(counter.values()), numBins)


def PlotCounter(counter):
    labels, values = zip(*counter.items())
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()


def PlotMatrixBinary(mat):
    plt.spy(mat, markersize=1, aspect='auto', origin='lower')
    plt.show()


def PlotMatrixColor(mat):
    PlotMatrixColorThresholds(mat, 0, 99)


def PlotMatrixColorThresholds(mat, min_thresh, max_thresh):
    plt.imshow(mat, vmin=min_thresh, vmax=max_thresh) # 0,99 are min,max defaults
    plt.colorbar()
    plt.show()


def plotMatrixPair(m1, m2):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(32,18))
    cmap = cm.jet
    # matplotlib.style.use('ggplot')
    plt.set_cmap(cmap)
    ax1.spy(m1, markersize=1, aspect='auto', origin='lower')
    ax2.spy(m2, markersize=1, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.show()


def PlotMatrixTrio(m1, m2, m3):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(32,18))
    cmap = cm.jet
    #plt.style.use('ggplot')
    plt.set_cmap(cmap)
    ax1.spy(m1, markersize=1, aspect='auto', origin='lower')
    ax2.spy(m2, markersize=1, aspect='auto', origin='lower')
    ax3.spy(m3, markersize=1, aspect='auto', origin='lower')
    plt.show()


def PlotClusterLists(list_of_lists, **kwargs):
    '''
    Takes a list of lists, each list is a the pixels of the corresponding cluster
    '''
    dimensions = kwargs.get('dim', '2D').lower()
    if (dimensions != '2d' and dimensions != '3d'):
        print('ERROR, dimensions must be 2d or 3d!!!!!')
    else:
        cluster_count = len(list_of_lists)
        cluster_arrays = []  # Each entry is an array, filled only with the maximal values from the corresponding
        for cluster in range(cluster_count):
            cluster_arrays.append(zeros([xdim, ydim]))  # (r,c)
            for pixel in list_of_lists[cluster]:
                cluster_arrays[cluster][pixel.x][pixel.y] = int(pixel.val)
        if dimensions == '2d':
            PlotListofClusterArraysColor2D(cluster_arrays, **kwargs)
        else:
            PlotListofClusterArraysColor3D(cluster_arrays, 1)


def FindBestClusterCount(array_of_floats, min, max, step):
    print('Attempting to find optimal number of clusters, range:(' + str(min) + ', ' + str(max))
    kVals = [] # The number of clusters
    distortionVSclusters = [] # The distortion per cluster
    for z in range(math.floor((max - min) / step)):
        num_clusters = (z * step) + min
        if(num_clusters == 0):
            num_clusters = 1
        print('Trying with ' + str(num_clusters) + ' clusters')
        (bookC, distortionC)  = kmeans(array_of_floats, num_clusters)
        # (centLabels, centroids) = vq(max_pixel_array_floats, bookC
        kVals.append(num_clusters)
        distortionVSclusters.append(distortionC)
    plt.plot(kVals, distortionVSclusters, marker='x')
    plt.grid(True)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Average cluster distortion')
    plt.title('Elbow method for K-Means Clustering on Pixels\nManually Record the desired K value\n')
    plt.show()


def PlotListofClusterArraysColor3D(list_of_arrays, have_divides): #have_divides is 0 to not show, otherwise show
    'Takes a list of 2D arrays, each of which is a populated cluster, and plots then in 3d.'
    # Try 3D plot
    colors2 = plt.get_cmap('gist_rainbow')
    num_clusters = len(list_of_arrays)
    cNorm = colortools.Normalize(vmin=0, vmax=num_clusters-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
    fig = plt.figure(figsize=(25,15)) # figsize=(x_inches, y_inches), default 80-dpi
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(num_clusters)])

    ax.set_xlim([0, xdim])
    ax.set_ylim([ydim, 0])
    ax.set_zlim([0, num_clusters])
    ax.view_init(elev=10., azim=0) #There is also a dist which can be set
    ax.dist = 8 # Default is 10, 0 is too low..


    for c in range(num_clusters):
        (x,y) = list_of_arrays[c].nonzero()
        ax.scatter(x,y, c, zdir='z', c=scalarMap.to_rgba(c))
        #plt.savefig("3D.png")

    if have_divides > 0:
        [xx, yy] = np.meshgrid([0, xdim], [0, ydim]) # Doing a grid with just the corners yields much better performance.
        for plane in range(len(list_of_arrays)-1):
            ax.plot_surface(xx, yy, plane+.5, alpha=.05)
    fig.tight_layout()
    plt.show()


def PlotListofClusterArraysColor2D(list_of_arrays, **kwargs):
    numbered = kwargs.get('numbered', False) # Output the pixel's blob id and order in the id list
    # Note: Numbering greatly increase draw time
    label_start_finish = kwargs.get('marked', False) # X on the first element of a blob, + on the last
    figsize = kwargs.get('figsize', (32, 32))
    markersize = kwargs.get('markersize', 30)

    colors2 = plt.get_cmap('gist_rainbow')
    num_clusters = len(list_of_arrays)
    cNorm = colortools.Normalize(vmin=0, vmax=num_clusters-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
    fig = plt.figure(figsize=figsize) # figsize=(x_inches, y_inches), default 80-dpi
    plt.clf()
    ax = fig.add_subplot(111)

    ax.set_xlim([0, xdim])
    ax.set_ylim([ydim, 0])

    for c in range(num_clusters):
        (x,y) = list_of_arrays[c].nonzero()
        ax.scatter(x,y, s=markersize, c=scalarMap.to_rgba(c), edgecolor=scalarMap.to_rgba(c))
        if label_start_finish:
            ax.plot(x[0], y[0], marker='x', markersize=markersize)
            ax.plot(x[-1], y[-1], marker='+', markersize=markersize)
            ax.annotate(str(c), xy=(x[0], y[0]))
            ax.annotate('\\'+str(c), xy=(x[-1], y[-1]))
        if numbered:
            for lab in range(len(x)):
                ax.annotate(str(c) + '.' + str(lab), xy=(x[lab], y[lab]))
        #plt.savefig("3D.png")
    fig.tight_layout()
    plt.savefig('temp/2D_Plot_of_Cluster_Arrays__' + timeNoSpaces() + '.png')
    plt.show()


def AnimateClusterArraysGif(list_of_arrays, imagefile, **kwargs):
    start_time = time.time()
    draw_divides = kwargs.get('divides', False)
    video_format = kwargs.get('format', 'MP4') # Either Mp4 or gif
    video_format = video_format.lower()
    figsize = kwargs.get('figsize', (8,4.5))

    ok_to_run = True

    if video_format != 'mp4' and video_format != 'gif':
        print('INVALID VIDEO FORMAT PROVIDED:' + str(video_format))
        ok_to_run = False
    else:
        total_frames = 617 #1940 #HACK
        speed_scale = 1 # Default is 1 (normal speed), 2 = 2x speed, **must be int for now due to range()
        total_frames = math.floor(total_frames / speed_scale)
        frame_offset = 615

        colors2 = plt.get_cmap('gist_rainbow')
        num_clusters = len(list_of_arrays)
        cNorm = colortools.Normalize(vmin=0, vmax=num_clusters-1)
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
        fig = plt.figure(figsize=figsize, dpi=100) # figsize=(x_inches, y_inches), default 80-dpi
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        t0 = time.time()

        # DEBUG
        # Frame continuitiny issues between frames:
        # 620/621 # Changed to be % 270..

        # NOTE making new directory for animation for organization:
        animation_time_string = timeNoSpaces()
        animation_folder = current_path + '\\temp\\' + animation_time_string
        os.makedirs(animation_folder)



    def animate(i):
        # i = (i * speed_scale) + frame_offset
        if i%1 == 0:
            curtime = time.time()
            temp = curtime - t0
            m = math.floor(temp / 60)
            print('Done with: ' + str(i) + '/' + str(total_frames / speed_scale) + ' frames, = %.2f percent' % (100 * ( i - frame_offset)/(total_frames / speed_scale)), end='')
            print('. Elapsed Time: ' + str(m) + ' minutes & %.0f seconds' % (temp % 60))
        if i < 360: # Rotate 360 degrees around horizontal axis
            ax.view_init(elev=10., azim=i) #There is also a dist which can be set
        elif i < 720:# 360 around vertical
            ax.view_init(elev=(10+i)%360., azim=0) #Going over
        elif i < 1080:# 360 diagonal
            ax.view_init(elev=(ax.elev + 1), azim=i%360) #There is also a dist which can be set
        elif i < 1100:# Quick rest
            #Sit for a sec to avoid sudden stop
            ax.view_init(elev=10., azim=0)
        elif i < 1250: # zoom in(to)
            d = 13 - (i-1100)/15 # 13 because 0 is to zoomed, now has min zoom of 3
            ax.dist = d
        elif i < 1790: #Spin from within, 540 degrees so reverse out backwards!
            ax.view_init(elev=(ax.elev + 1), azim=0) #Going over
            ax.dist = 1
        else: # zoom back out(through non-penetrated side)
            d = 3 + (i-1790)/15
            ax.dist = d

    def generateFrames():
        'Takes a list of arrays, each of which is a populated cluster.'

        #Elev and azim are both in degrees
        # Performance Increasers:
        ax.set_xlim([0, xdim])
        ax.set_ylim([ydim, 0])
        ax.set_zlim([0, num_clusters])

        for c in range(len(list_of_arrays)):
            (x, y) = list_of_arrays[c].nonzero()
            ax.scatter(x, y, c, zdir='z', c=scalarMap.to_rgba(c))
        if draw_divides != 0:
            [xx, yy] = np.meshgrid([0, 1600],[0, 1600]) # Doing a grid with just the corners yields much better performance.
            for plane in range(len(list_of_arrays)-1):
                ax.plot_surface(xx, yy, plane+.5, alpha=.05)
        fig.tight_layout()
        print('Generating and saving frames, start_time: ' + str(time.ctime()) + ', saving to folder: ' + str(animation_folder))
        for i in range(frame_offset,total_frames, speed_scale):

            animate(i)
            #im = fig2img(fig)
            #im.show()
            buf = i
            padding = '00000' # Hack
            buf_digits = buf
            while buf_digits >= 10:
                padding = padding[1:]
                buf_digits = buf_digits / 10
            plt.savefig(animation_folder + '/gif_frame_' + padding + str(buf) + '.png', bbox_inches='tight')
            #frames.append(im)

    def framesToGif(): # TODO convert to calling executable with: http://pastebin.com/JJ6ZuXdz
        # HACK
        # IMAGEMAGICK_CONVERT_EXEC = 'C:\\Program Files\\ImageMagick-6.9.1-Q8\\convert.exe'
        # HACK
        frame_names = animation_folder + '/*.png' # glob.glob('temp/*.png')
        #print('Frame names:' + str(frame_names))
        #frames = [Image.open(frame_name) for frame_name in frame_names]
        imagex = 10 * figsize[0]
        imagey = 10 * figsize[1]
        filename_out = (imagefile[-12:-4] + '_' + animation_time_string + '.gif')
        print('Now writing gif to:' + str(filename_out))

        command = [IMAGEMAGICK_CONVERT_EXEC, "-delay", "0", "-size", str(imagex)+'x'+str(imagey)] + [frame_names] + [filename_out]
        t1 = time.time()
        m = math.floor((t1-t0) / 60)
        s = (t1-t0) % 60
        print('It has been:' + str(m) + ' mins & ' + str(s) + ' seconds, now calling imagemagick_executable to generate gif')
        subprocess.call(command)
        t2 = time.time()
        m = math.floor((t2-t1) / 60)
        s = (t2-t1) % 60
        print('Done saving animated gif; took ' + str(m) + ' mins & ' + str(s) + ' seconds.')

        # writeGif(filename, frames, duration=100, dither=0)
        # TODO Change rotation over vertical 270 degrees
        # TODO Check that isnt also an issue horizontally
        # TODO Adjust the percentages output by animate(i)
        # TODO Check the ram issue's source; see if theres a way to view usage via debug somehow within pycharm
        # TODO Remove Anaconda3(Safely)
        print('Done writing gif')

    def GifImageMagick():
        print('Generating image-magick anim')
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=20, blit=True) # 1100 = 360 + 360 + 360 + 30
        print('Now writing gif')
        filename = imagefile[-12:-4] + '.gif'
        print('Saving ImageMagick gif as:' + str(filename))
        anim.save("test.gif", writer='imagemagick', fps=10)
        # runShell()
        print('Done writing gif')

    if ok_to_run:
        if video_format == 'mp4':
            for c in range(len(list_of_arrays)):
                (x,y) = list_of_arrays[c].nonzero()
                ax.scatter(x,y, c, zdir='z', c=scalarMap.to_rgba(c))
            if draw_divides:
                [xx, yy] = np.meshgrid([0, xdim],[0, ydim]) # Doing a grid with just the corners yields much better performance.
                for plane in range(len(list_of_arrays)-1):
                    ax.plot_surface(xx, yy, plane+.5, alpha=.05)

            plt.title('Animation_of_ ' + str(imagefile[8:-4] + '.mp4'))
            fig.tight_layout()
            anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=20, blit=True) # 1100 = 360 + 360 + 360 + 30
            print('Saving, start_time: ' + str(time.ctime()))
            anim.save('Animation_of ' + str(imagefile[8:-4]) + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            end_time = time.time()
            print('Time to save animation: ' + str(end_time - start_time))
        elif video_format == 'gif':
            generateFrames()
            framesToGif()

