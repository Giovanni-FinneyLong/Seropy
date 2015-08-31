__author__ = 'gio'
# This file includes the various functions written to visualize data from within sero.py
# These functions have been separated for convenience; they are higher volume and lower maintenance.

from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import matplotlib
import tkinter
# HACK HACK
# matplotlib.use('GTK3Agg')
# HACK HACK http://www.swharden.com/blog/2013-04-15-fixing-slow-matplotlib-in-pythonxy/
# http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
import glob
# import wand
# import cv2 # OpenCV version 2

import matplotlib.colors as colortools
from matplotlib import animation


import matplotlib.pylab as plt
# import vispy.mpl_plot as plt


import matplotlib.cm as cm
import numpy as np
import time
import math
from PIL import Image
from numpy import zeros
from visvis.vvmovie.images2gif import writeGif
# from Scripts.images2gif import writeGif
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2

import subprocess
import readline
import code
import rlcompleter
import pdb
import os
from mpl_toolkits.mplot3d import Axes3D

from config import *
from vispy import plot as vp

import vispy.io
import vispy.scene
from vispy.scene import visuals
from vispy import gloo


# NOTE  ##########################
# NOTE  Setting up global vars:
current_path = os.getcwd()
xdim = -1
ydim = -1
zdim = -1

# master_start_time = 0 # Set at the start of main # FIXME!

debug_blob_ids = False
debug_pixel_ops = False
debug_set_merge = False
remap_ids_by_group_size = True
test_instead_of_data = True
dePickle = False
debug_pixel_ops_y_depth = 500

min_val_threshold = 250
    # Recommended = 250
max_val_step = 5 # The maximum amount that two neighboring pixels can differ in val and be grouped by blob_id
    # Recommended = 5
minimal_nonzero_neighbors = 2 # The minimal amount of nzn a pixel must have to avoid being filtered; 0 = no filter
    # Recommended = 2
# NOTE  ##########################


#HACK
array_list = []
frame_num = 0
scatter = visuals.Markers()
scatter_list = []
possible_lines = visuals.Line()
context_lines = visuals.Line()
rotation = 'x'
scale = 1.
animation_time_string = ''
frames = 0
orders = [] # List of tuples of the format ('command', degrees/scaling total, number of iterations)
order_frame = 0

canvas = None
view = None

def setMasterStartTime():
    master_start_time = time.time() # FIXME!


def plotSlidesVC(slide_stack, **kwargs):
    '''
    For now, using V as a suffix to indicate that the plotting is being done via vispy, C for colored
    '''
    global frames
    global orders
    global array_list
    global scatter
    global scatter_list
    global scale
    global rotation
    global animation_time_string
    global canvas
    global view

    edges = kwargs.get('edges', False) # True if only want to plot the edge pixels of each blob
    coloring = kwargs.get('color', None)
    midpoints = kwargs.get('midpoints', True)
    partnerlines = kwargs.get('possible', False)
    contextlines = kwargs.get('context', False)
    animate = kwargs.get('animate', False)
    orders = kwargs.get('orders') # ('command', total scaling/rotation, numberofframes)
    canvas_size = kwargs.get('canvas_size', (800,800))
    gif_size = kwargs.get('gif_size', canvas_size)


    def framesToGif(frames_folder, animation_time_string): # TODO convert to calling executable with: http://pastebin.com/JJ6ZuXdz
        frame_names = frames_folder + '/*.png' # glob.glob('temp/*.png')
        filename_out = ('generated_figures/' + animation_time_string + '.gif')
        print('Now writing gif to:' + str(filename_out))

        command = [IMAGEMAGICK_CONVERT_EXEC, "-delay", "0", "-size", str(gif_size[0])+'x'+str(gif_size[1])] + [frame_names] + [filename_out]
        t1 = time.time()
        # m = math.floor((t1-t0) / 60)
        # s = (t1-t0) % 60
        print('Now calling imagemagick_executable to generate gif, start time: ' + str(timeNoSpaces()))
        subprocess.call(command)
        t2 = time.time()
        m = math.floor((t2-t1) / 60)
        s = (t2-t1) % 60
        print('Done saving animated gif; took: ' + str(m) + ' mins & ' + str(s) + ' seconds.')




    assert coloring in [None, 'blobs', 'slides']
    assert edges in [True, False]
    assert midpoints in [True, False]

    animation_time_string = timeNoSpaces()
    animation_folder = current_path + '/temp/' + animation_time_string
    os.makedirs(animation_folder)

    colors = vispy.color.get_color_names()
    index = 0
    if animate:
        frames = 0
        for order in orders:
            frames += (order[2])# + 1) # HACK +1 for the transition so that we can stop the timer
        print('There are ' + str(frames) + ' total frames')
        assert (animate is True and frames > 0 and len(orders) > 0)
        canvas = vispy.scene.SceneCanvas(show=True, size=canvas_size, resizable=False)
    else:
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, size=canvas_size)

    view = canvas.central_widget.add_view()

    if coloring == 'blobs':
        array_list = []
        for (slide_num, slide) in enumerate(slide_stack):
            print('Adding slide:' + str(slide_num) + '/' + str(len(slide_stack)))
            for blob in slide.blob2dlist:
                if edges:
                    array_list.append(np.zeros([len(blob.edge_pixels), 3]))
                    for (p_num, pix) in enumerate(blob.edge_pixels):
                        array_list[-1][p_num] = [pix.x / xdim, pix.y / ydim, slide_num / len(slide_stack)]
                else:
                    array_list.append(np.zeros([len(blob.pixels), 3]))
                    for (p_num, pix) in enumerate(blob.pixels):
                        array_list[-1][p_num] = [pix.x / xdim, pix.y / ydim, slide_num / len(slide_stack)]
    elif coloring == 'slides':
        array_list = []
        for (slide_num, slide) in enumerate(slide_stack):
            print('Adding slide:' + str(slide_num) + '/' + str(len(slide_stack)))
            if edges:
                array_list.append(np.zeros([len(slide.edge_pixels), 3]))
                for (p_num, pix) in enumerate(slide.edge_pixels):
                    array_list[-1][p_num] = [pix.x / xdim, pix.y / ydim, slide_num / len(slide_stack)]
            else:
                array_list.append(np.zeros([len(slide.alive_pixels), 3]))
                for (p_num, pix) in enumerate(slide.alive_pixels):
                    array_list[-1][p_num] = [pix.x / xdim, pix.y / ydim, slide_num / len(slide_stack)]
        # # DEBUG
        # print('Done adding colored slides:' + str(array_list))
    else: # No coloring
        total_pixels = 0
        if edges:
            for slide in slide_stack:
                total_pixels += len(slide.edge_pixels)
        else:
            for slide in slide_stack:
                total_pixels += len(slide.alive_pixels)
        array_list = np.zeros([total_pixels, 3]) # NOTE Intentional misnaming; just an array, but now can use the same calls as the colored
        index = 0
        for (slide_num, slide) in enumerate(slide_stack):
            print('Adding slide:' + str(slide_num) + '/' + str(len(slide_stack)))
            if edges:
                for pix in slide.edge_pixels:
                    array_list[index] = [pix.x / xdim, pix.y / ydim, slide_num / len(slide_stack)]
                    index += 1
            else:
                for pix in slide.alive_pixels:
                    array_list[index] = [pix.x / xdim, pix.y / ydim, slide_num / len(slide_stack)]
                    index += 1

        scatter = visuals.Markers() # TODO remove
        scatter.set_data(array_list, edge_color=None, face_color=(1, 1, 1, .5), size=5)
        view.add(scatter)

    if coloring is not None:
        for (a_num, arr) in enumerate(array_list):
            print('Array_num:' + str(a_num) + ' is colored:' + str(colors[a_num % len(colors)]))
            scatter_list.append(visuals.Markers())
            scatter_list[-1].set_data(arr, edge_color=None, face_color=colors[a_num % len(colors)], size=5)
            view.add(scatter_list[-1])


    total_blobs = 0
    for slide in slide_stack:
        total_blobs += len(slide.blob2dlist)
    print('There are a total of ' + str(total_blobs) + ' blobs in the ' + str(len(slide_stack)) + ' slides')

    if midpoints:
        avg_list = np.zeros([total_blobs, 3])
        index = 0
        for slide_num, slide in enumerate(slide_stack):
            for blob in slide.blob2dlist:
                avg_list[index] = [blob.avgx / xdim, blob.avgy / ydim, slide_num / len(slide_stack)]
                index += 1
        # avg_scatter = visuals.Markers()
        # avg_scatter.set_data(avg_list, edge_color=None, face_color=(1, 1, 1, .5), size=15, symbol='*')
        # view.add(avg_scatter)
        for num, midp in enumerate(avg_list):
            view.add(visuals.Text(str(num), pos=midp, color='white'))

    if partnerlines:
        line_count = 0 # Counting the number of line segments to be drawn
        for slide in slide_stack:
            for blob in slide.blob2dlist:
                line_count += len(blob.possible_partners)
        line_count *= 2 # Because need start and end points
        linedata = np.zeros([line_count, 1, 3])
        index = 0
        for slide_num, slide in enumerate(slide_stack[:-1]): # All but the lat slide
            for blob in slide.blob2dlist:
                # print('DB: Slide:' + str(slide_num) + ' Blob:' + str(blob) + ' partners:' + str(blob.possible_partners))
                for partner in blob.possible_partners:
                    linedata[index] = [blob.avgx / xdim, blob.avgy / ydim, slide_num / len(slide_stack)]
                    linedata[index + 1] = [partner.avgx / xdim, partner.avgy / ydim, (slide_num + 1) / len(slide_stack)]
                    index += 2
        possible_lines.set_data(pos=linedata, connect='segments')
        view.add(possible_lines)

    if contextlines: # TODO

        contextline_count = 0
        for slide in slide_stack:
            for blob in slide.blob2dlist:
                for partner in blob.partner_indeces:


                    if dePickle:
                        partner = partner[0] # NOTE HACK remove this once re-pickled
                    contextline_count += len(partner)
        contextline_count *= 2 # Because need start and end point
        contextline_data = np.zeros([contextline_count,1,3])
        index = 0
        print('Context Line Count: ' + str(contextline_count))
        for slide_num, slide in enumerate(slide_stack[:-1]):
            # print('Slide #' + str(slide_num))
            for blob_num, blob in enumerate(slide.blob2dlist):
                # print('partner_indeces: ' + str(blob.partner_indeces))
                for partner_num, partner in enumerate(blob.partner_indeces):

                    if dePickle:
                        partner = partner[0] # NOTE HACK remove this once re-pickled
                    # print('partner: ' + str(partner))
                    for edgep1, edgep2 in partner:
                        # print(str(edgep1) + ' / ' + str(len(blob.edge_pixels)) + ' : ' +  str(edgep2) + ' / ' + str(len(blob.possible_partners[partner_num].edge_pixels)))
                        if edgep1 < len(blob.edge_pixels) and edgep2 < len(blob.possible_partners[partner_num].edge_pixels):
                            contextline_data[index] = blob.edge_pixels[edgep1].x / xdim, blob.edge_pixels[edgep1].y / ydim, slide_num / len(slide_stack)
                            temp_pix = blob.possible_partners[partner_num].edge_pixels[edgep2]
                            contextline_data[index+1] = temp_pix.x / xdim, temp_pix.y / ydim, (slide_num + 1) / len(slide_stack)
                            # print('Line:' + str(contextline_data[index]) + ' : ' + str(contextline_data[index+1]) + ', index=' + str(index) + ' / ' + str(contextline_count))
                            index += 2
                        else:
                            # print('Overflow, hopefully due to matrix expansion')
                            maxEdge = max(edgep1, edgep2)
                            maxEdgePixels = max(len(blob.edge_pixels), len(blob.possible_partners[partner_num].edge_pixels))
                            if maxEdge > maxEdgePixels:
                                print('\n\n-----ERROR! Pixel number was greater than both edge_pixel lists')
                                debug()
        print('Done generating contextline_data:' + str(contextline_data))
        context_lines.set_data(pos=contextline_data, connect='segments')
        view.add(context_lines)


    view.camera = 'turntable'  # or try 'arcball'
    view.camera.elevation = -75
    view.camera.azimuth = 1
    print('VIEW = ' + str(view))
    print('VIEW.Cam = ' + str(view.camera))

    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)

    def update_points(event):
        global array_list
        global frame_num
        global scatter
        global possible_lines
        global rotation
        global scale
        global frames
        global orders
        global order_frame

        if not canvas._closed:

            # Note that orders is modified as it it is read
            draw_arrays = []
            frame_num += 1
            rotation = None
            if order_frame > orders[0][2]: # Orders = (command, total_transform, number of frames for transform)
                if len(orders) > 1:
                    orders = orders[1:]
                    order_frame = 0

            if order_frame <= orders[0][2]: # Orders = (command, total_transform, number of frames for transform)
                base_order = orders[0][0][0]
                polarity = orders[0][0][1]

                if base_order in ['x', 'y','z']:
                    angle = math.radians((order_frame / orders[0][2]) * orders[0][1])
                    # print('Angle=' + str(math.degrees(angle)))

                if polarity == '-':
                    polarity = -1
                else:
                    polarity = 1

                if base_order == 'x':
                    rotate_arr = [
                        [1,0,0],
                        [ 0, math.cos(angle), -1. * math.sin(angle)],
                        [ 0, math.sin(angle), math.cos(angle)]
                    ]
                elif base_order == 'y':
                    rotate_arr = [
                        [ math.cos(angle), 0, math.sin(angle)],
                        [0,1,0],
                        [ -1. * math.sin(angle), 0, math.cos(angle)]
                    ]
                elif base_order == 'z':
                    rotate_arr = [
                        [ math.cos(angle), -1. * math.sin(angle), 0],
                        [ math.sin(angle), math.cos(angle), 0],
                        [0,0,1]
                    ]
                else:
                    rotate_arr = [
                        [1,0,0],
                        [0,1,0],
                        [0,0,1]
                    ]


                rotate_arr = np.multiply(rotate_arr, (scale * polarity))

                possible_lines.set_data(pos=np.dot((linedata - .5), rotate_arr) + .5, connect='segments')
                for arr in array_list:
                    draw_arrays.append(np.zeros(arr.shape))
                    # buf = np.dot(draw_arrays[-1], rotate_arr)
                    draw_arrays[-1] = np.dot((arr - .5), rotate_arr) + .5

                if coloring is not None:
                    for (a_num, arr) in enumerate(draw_arrays):
                        # scatter_list[a_num].set_data(arr, edge_color=None, face_color=colors[(a_num + frame_num) % len(colors)], size=5)
                        scatter_list[a_num].set_data(arr, edge_color=None, face_color=colors[(a_num) % len(colors)], size=5)

                image = canvas.render()
                buf = '00000'
                buf_off = frame_num
                while buf_off > 9:
                    buf_off = math.floor(buf_off / 10)
                    buf = buf[1:]

                vispy.io.write_png(animation_folder + '/Vispy' + buf + str(frame_num) + '.png', image)
                # view.update()
                order_frame += 1
            else:
                    print('Done rendering animation')
                    canvas.close()
                    framesToGif(animation_folder, animation_time_string)

    def update_camera(event):
        global frames
        global orders
        global order_frame
        global frame_num

        if not canvas._closed:

            if order_frame >= orders[0][2]: # Orders = (command, total_transform, number of frames for transform)
                if len(orders) > 1:
                    orders = orders[1:]
                    order_frame = 0
                else:
                    print('Done rendering animation, about to call frames to gif from within camera update')
                    canvas.close()
                    framesToGif(animation_folder, animation_time_string)

            if order_frame < orders[0][2]: # Orders = (command, total_transform, number of frames for transform) # TODO can remove this is-else (keep contents of if block)
                base_order = orders[0][0][0]
                polarity = orders[0][0][1]
                if base_order in ['x', 'y','z']:
                    angle = orders[0][1] / orders[0][2]

                if polarity == '-':
                    angle *= -1

                if base_order == 'x' or base_order == 'z':
                    view.camera.azimuth += angle
                if base_order == 'y' or base_order == 'z':
                    if abs(view.camera.elevation + angle) > 90:
                        if view.camera.elevation + angle > 90:
                            view.camera.elevation = (view.camera.elevation + angle) - 180
                            # view.camera.azimuth += 90
                        else:
                            view.camera.elevation = (view.camera.elevation + angle) + 180
                            # view.camera.azimuth += 180

                    else:
                        view.camera.elevation = (view.camera.elevation + angle)
                image = canvas.render()
                buf = '00000'
                buf_off = frame_num
                while buf_off > 9:
                    buf_off = math.floor(buf_off / 10)
                    buf = buf[1:]
                vispy.io.write_png(animation_folder + '/Vispy' + buf + str(frame_num) + '.png', image)
                order_frame += 1
                frame_num += 1
            else:
                print('Never should have reached here')
                print('Likely error closing canvas. Perhaps try with timer\'s parent?')
        # canvas.update()

        print('Camera(' + str(frame_num) + ') : Elevation:' + str(view.camera.elevation) + ' Azimuth:' + str(view.camera.azimuth) + ' distance:' + str(view.camera.distance) + ' center' + str(view.camera.center))
        # if view.camera.distance is None:
        #     view.camera.distance = 10
        # view.camera.distance -= 1
        # view.camera.orbit(1,1)


        # Note elevation is RESTRICTED to the range (-90, 90) = Rotate camera over
        # Note  azimuth ROUND (0, 180/-179, -1) (loops at end of range) = Rotate camera around


    if animate:
        # timer = vispy.app.Timer(interval=0, connect=update_points, start=True, iterations=frames+1)
        view.camera.interactive = False
        timer = vispy.app.Timer(interval=0, connect=update_camera, start=True, iterations=frames+1) # Hack +1 fpr writing?
    # DEBUG
    print('Now starting visuals, but first, slide/ blob info:')
    for snum, slide in enumerate(slide_stack):
        print('Slide: ' + str(snum) + ' / ' + str(len(slide_stack)))
        for blobl in slide.blob2dlist:
            print(' Blob:' + str(blob) + ' which has' + str(len(blob.possible_partners)) + ' possible partners')
    vispy.app.run()




def printElapsedTime(t0, tf):
    temp = tf - t0
    m = math.floor(temp / 60)
    print('Elapsed Time: ' + str(m) + ' minutes & %.0f seconds' % (temp % 60))


def setseerodrawdims(x,y,z):
    global xdim
    global ydim
    global zdim
    xdim = x
    ydim = y
    zdim = z


def runShell():
    gvars = globals()
    gvars.update(locals())
    readline.set_completer(rlcompleter.Completer(gvars).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(gvars)
    shell.interact()

def debug():
    pdb.set_trace()

def timeNoSpaces():
    return time.ctime().replace(' ', '_').replace(':', '-')

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

def plotSlides(slide_list):
    colors2 = plt.get_cmap('gist_rainbow')
    num_slides = len(slide_list)
    # cNorm = colortools.Normalize(vmin=0, vmax=num_slides-1)
    # scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
    fig = plt.figure(figsize=(25,15)) # figsize=(x_inches, y_inches), default 80-dpi
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(num_clusters)])

    ax.set_xlim([0, xdim])
    ax.set_ylim([ydim, 0])
    ax.set_zlim([0, num_slides])
    ax.view_init(elev=10., azim=0) #There is also a dist which can be set
    ax.dist = 8 # Default is 10, 0 is too low..

    for (slide_num, slide) in enumerate(slide_list):
        print('Plotting slide:' + str(slide_num) + '/' + str(len(slide_list)) + '  @' + str(time.ctime()))
        cluster_lists = []
        num_blobs = len(slide.blob2dlist)
        cNorm = colortools.Normalize(vmin=0, vmax=num_blobs-1) # NOTE this makes a new color cycle for each slide
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
        x = []
        y = []
        # print(len(slide.blob2dlist))
        # print(slide.blob2dlist)


        for (blob_num, blob) in enumerate(slide.blob2dlist):
            print(' Plotting blob: ' + str(blob_num) + '/' + str(len(slide.blob2dlist)))
            for pixel in blob.edge_pixels: # hack ONLY PLOTTING EDGES
                x.append(pixel.x)
                y.append(pixel.y)
            ax.scatter(x, y, slide_num, c=scalarMap.to_rgba(blob_num))
    print('Making tight layout @' + str(time.ctime()))
    fig.tight_layout()
    # print('Now saving rendering @' + str(time.ctime()))
    # plt.savefig("3D.png")
    print('Now displaying rendering @' + str(time.ctime()))
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

