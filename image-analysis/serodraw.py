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
# from visvis.vvmovie.images2gif import writeGif
# from Scripts.images2gif import writeGif
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2

import subprocess
import readline
import code
import rlcompleter
import pdb
import os
from mpl_toolkits.mplot3d import Axes3D

from myconfig import *
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


# NOTE  ##########################


#HACK
array_list = []
frame_num = 0
scatter = visuals.Markers()
scatter_list = []
possible_lines = visuals.Line(method=linemethod)
context_lines = visuals.Line(method=linemethod)
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

def plotBlod3ds(blob3dlist, **kwargs):
    global canvas
    global view

    canvas_size = kwargs.get('canvas_size', (800,800))
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, size=canvas_size)

    coloring = kwargs.get('color', None)

    # Finding the maximal slide, so that the vertical dimension of the plot can be evenly divided
    total_slides = 0
    for blob3d in blob3dlist:
        if blob3d.highslide > total_slides:
            total_slides = blob3d.highslide
    total_slides += 1 # Note this is b/c numbering starts at 0

    view = canvas.central_widget.add_view()
    view.camera = 'turntable'  # or try 'arcball'
    view.camera.elevation = -75
    view.camera.azimuth = 1
    edge_pixel_arrays = [] # One array per 3d blob
    markerlist = []
    colors = vispy.color.get_color_names() # ALl possible colors
    lineendpoints = 0

    if coloring == 'blob': # Note: This is very graphics intensive.

        for blob_num, blob3d in enumerate(blob3dlist):
            edge_pixel_arrays.append(np.zeros([len(blob3d.edge_pixels), 3]))
            for (p_num, pixel) in enumerate(blob3d.edge_pixels):
                edge_pixel_arrays[-1][p_num] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( z_compression * total_slides)]
            markerlist.append(visuals.Markers())
            markerlist[-1].set_data(edge_pixel_arrays[-1], edge_color=None, face_color=colors[blob_num % len(colors)], size=8)
            view.add(markerlist[-1])
            for stitch in blob3d.stitches:
                lineendpoints += (2 * len(stitch.indeces)) # 2 as each line has 2 endpoints
    else: # All colored the same
        total_points = 0
        for blob_num, blob3d in enumerate(blob3dlist):
            total_points += len(blob3d.edge_pixels)
        edge_pixel_array = np.zeros([total_points, 3])
        index = 0
        for blob3d in blob3dlist:
            for pixel in blob3d.edge_pixels:
                edge_pixel_array[index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (z_compression * total_slides)]
                index += 1
            for stitch in blob3d.stitches:
                lineendpoints += (2 * len(stitch.indeces)) # 2 as each line has 2 endpoints

        markers = visuals.Markers()
        markers.set_data(edge_pixel_array, edge_color=None, face_color=colors[0], size=8) # TODO change color
        view.add(markers)



    lower_index = 0
    upper_index = 0
    line_index = 0
    
    lower_markers_locations = np.zeros([lineendpoints / 2, 3]) # Note changes to points_to_draw (num indeces) rather than count of pixels
    upper_markers_locations = np.zeros([lineendpoints / 2, 3])
    line_locations = np.zeros([lineendpoints, 3])

    for blob3d in blob3dlist:
        for stitch in blob3d.stitches:
            for lowerpnum, upperpnum in stitch.indeces:
                lowerpixel = stitch.lowerpixels[lowerpnum]
                upperpixel = stitch.upperpixels[upperpnum]
                lower_markers_locations[lower_index] = [lowerpixel.x / xdim, lowerpixel.y / ydim, (stitch.lowerslidenum ) / ( z_compression * total_slides)]
                upper_markers_locations[upper_index] = [upperpixel.x / xdim, upperpixel.y / ydim, (stitch.upperslidenum ) / ( z_compression * total_slides)]
                line_locations[line_index] = lower_markers_locations[lower_index]
                line_locations[line_index + 1] = upper_markers_locations[upper_index]

                lower_index += 1
                upper_index += 1
                line_index += 2
    lower_markers = visuals.Markers()
    upper_markers = visuals.Markers()
    stitch_lines = visuals.Line(method=linemethod)

    # if coloring == 'blob': # TODO optimize the above as these are not used
        # lower_markers.set_data(lower_markers_locations, edge_color=None, size=10)
        # upper_markers.set_data(upper_markers_locations, edge_color=None, size=7)
    if coloring != 'blob':
        lower_markers.set_data(lower_markers_locations, edge_color=None, face_color='yellow', size=11)
        upper_markers.set_data(upper_markers_locations, edge_color=None, face_color='green', size=11)
        lower_markers.symbol = 'ring'
        upper_markers.symbol = '+'
        view.add(lower_markers)
        view.add(upper_markers)
    stitch_lines.set_data(pos=line_locations, connect='segments')

    view.add(stitch_lines)
    vispy.app.run()





def plotSlidesVC(slide_stack, stitchlist, **kwargs):
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
    stitches = kwargs.get('stitches', False)
    polygons = kwargs.get('polygons', False)
    subpixels = kwargs.get('subpixels', False)




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
    view.camera = 'turntable'  # or try 'arcball'
    view.camera.elevation = -75
    view.camera.azimuth = 1
    print('VIEW.Cam = ' + str(view.camera))





    if coloring == 'blobs':
        array_list = []
        for (slide_num, slide) in enumerate(slide_stack):
            print('Adding slide:' + str(slide_num) + '/' + str(len(slide_stack)))
            for blob in slide.blob2dlist:
                if edges:
                    array_list.append(np.zeros([len(blob.edge_pixels), 3]))
                    for (p_num, pix) in enumerate(blob.edge_pixels):
                        array_list[-1][p_num] = [pix.x / xdim, pix.y / ydim, slide_num / ( z_compression * len(slide_stack))]
                else:
                    array_list.append(np.zeros([len(blob.pixels), 3]))
                    for (p_num, pix) in enumerate(blob.pixels):
                        array_list[-1][p_num] = [pix.x / xdim, pix.y / ydim, slide_num / ( z_compression * len(slide_stack))]
    elif coloring == 'slides':
        array_list = []
        for (slide_num, slide) in enumerate(slide_stack):
            print('Adding slide:' + str(slide_num) + '/' + str(len(slide_stack)))
            if edges:
                array_list.append(np.zeros([len(slide.edge_pixels), 3]))
                for (p_num, pix) in enumerate(slide.edge_pixels):
                    array_list[-1][p_num] = [pix.x / xdim, pix.y / ydim, slide_num / ( z_compression * len(slide_stack))]
            else:
                array_list.append(np.zeros([len(slide.alive_pixels), 3]))
                for (p_num, pix) in enumerate(slide.alive_pixels):
                    array_list[-1][p_num] = [pix.x / xdim, pix.y / ydim, slide_num / ( z_compression * len(slide_stack))]
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
                    array_list[index] = [pix.x / xdim, pix.y / ydim, slide_num / ( z_compression * len(slide_stack))]
                    index += 1
            else:
                for pix in slide.alive_pixels:
                    array_list[index] = [pix.x / xdim, pix.y / ydim, slide_num / ( z_compression * len(slide_stack))]
                    index += 1

        scatter = visuals.Markers() # TODO remove
        scatter.set_data(array_list, edge_color=None, face_color=(1, 1, 1, .5), size=5)
        view.add(scatter)

    if coloring is not None:
        for (a_num, arr) in enumerate(array_list):
            print('Array_num:' + str(a_num) + ' is colored:' + str(colors[a_num % len(colors)]))
            scatter_list.append(visuals.Markers())
            scatter_list[-1].set_data(arr, edge_color=None, face_color=colors[a_num % len(colors)], size=8)
            # print('DEBUG array data:' + str(arr))
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
                avg_list[index] = [blob.avgx / xdim, blob.avgy / ydim, slide_num / ( z_compression * len(slide_stack))]
                view.add(visuals.Text(str(blob.id) + ':' + str(index), pos=avg_list[index], color='white'))
                index += 1
        # for num, midp in enumerate(avg_list):
        #     view.add(visuals.Text(str(num), pos=midp, color='white'))


    if polygons: # See https://github.com/vispy/vispy/blob/6834a2b89e4a218f1c1783c588d5f47a781a2f7f/examples/basics/scene/surface_plot.py for surface plots
        polygonvisuals = []
        x = []
        y = []
        z = []
        for stitch in stitchlist:
            vol = np.zeros([2 * len(stitch.indeces), 3])
            print ('Vol ndim:' + str(vol.ndim))
            for index_num, (lowerpnum, upperpnum) in enumerate(stitch.indeces):
                lowerpixel = stitch.lowerpixels[lowerpnum]
                upperpixel = stitch.upperpixels[upperpnum]
                vol[2 * index_num] = [lowerpixel.x, lowerpixel.y, stitch.lowerslidenum / ( z_compression * len(slide_stack))]
                vol[2 * index_num + 1] = [upperpixel.x, upperpixel.y, stitch.upperslidenum / ( z_compression * len(slide_stack))]
                view.add(visuals.Polygon(vol)) # TODO FIXME NEEDS A 3d image
                # view.add(visuals.SurfacePlot(x=np.array(x).flatten(), y=np.array(y).flatten(), color=(0.3, 0.3, 1, 1)))
                # if index_num == len(stitch.indeces) - 1: # Wrap around, to draw the last polygon with the same edge as the first



    if stitches:
        total_lower_pixels = 0
        total_upper_pixels = 0
        points_to_draw = 0
        for stitch in stitchlist:
            total_lower_pixels += len(stitch.lowerpixels)
            total_upper_pixels += len(stitch.upperpixels)
            points_to_draw += len(stitch.indeces)
        print('DEBUG lower v upper pixels v lines:' + str(total_lower_pixels) + ' ' + str(total_upper_pixels) + ' ' + str(points_to_draw))

        lower_markers_locations = np.zeros([points_to_draw, 3]) # Note changes to points_to_draw (num indeces) rather than count of pixels
        upper_markers_locations = np.zeros([points_to_draw, 3])
        points_to_draw *= 2 # Multiply by 2 for start and end points
        line_locations = np.zeros([points_to_draw, 3])


        lower_index = 0
        upper_index = 0
        line_index = 0

        for stitch in stitchlist:
            for lowerpnum, upperpnum in stitch.indeces:
                lowerpixel = stitch.lowerpixels[lowerpnum]
                upperpixel = stitch.upperpixels[upperpnum]
                lower_markers_locations[lower_index] = [lowerpixel.x / xdim, lowerpixel.y / ydim, (stitch.lowerslidenum ) / ( z_compression * len(slide_stack))]
                upper_markers_locations[upper_index] = [upperpixel.x / xdim, upperpixel.y / ydim, (stitch.upperslidenum ) / ( z_compression * len(slide_stack))]
                line_locations[line_index] = lower_markers_locations[lower_index]
                line_locations[line_index + 1] = upper_markers_locations[upper_index]

                lower_index += 1
                upper_index += 1
                line_index += 2
        lower_markers = visuals.Markers()
        upper_markers = visuals.Markers()
        stitch_lines = visuals.Line(method=linemethod)
        # print('Stitch lower locations:' + str(lower_markers_locations))
        # print('Stitch upper locations:' + str(upper_markers_locations))

        if xdim < 300 or ydim < 300: # HACK
            lowersize = 20
            uppersize = 15
        else: # Basically for the actual slides
            lowersize = 10
            uppersize = 7


        lower_markers.set_data(lower_markers_locations, edge_color=None, face_color='yellow', size=lowersize)
        upper_markers.set_data(upper_markers_locations, edge_color=None, face_color='green', size=uppersize)
        stitch_lines.set_data(pos=line_locations, connect='segments')
        lower_markers.symbol = 'ring'
        upper_markers.symbol = '+'
        view.add(lower_markers)
        view.add(upper_markers)
        view.add(stitch_lines)

        # context_lines.set_data(pos=contextline_data, connect='segments')
        # view.add(context_lines)


        '''

        contextline_data = np.zeros([contextline_count,1,3])
        index = 0
        print('Context Line Count: ' + str(contextline_count))
        for slide_num, slide in enumerate(slide_stack[:-1]):
            # print('Slide #' + str(slide_num))
            for blob_num, blob in enumerate(slide.blob2dlist):
                # print('partner_indeces: ' + str(blob.partner_indeces))
                for partner_num, partner in enumerate(blob.partner_indeces):
                    for edgep1, edgep2 in partner:
                        # print(str(edgep1) + ' / ' + str(len(blob.edge_pixels)) + ' : ' +  str(edgep2) + ' / ' + str(len(blob.possible_partners[partner_num].edge_pixels)))
                        if edgep1 < len(blob.edge_pixels) and edgep2 < len(blob.possible_partners[partner_num].edge_pixels):
                            contextline_data[index] = blob.edge_pixels[edgep1].x / xdim, blob.edge_pixels[edgep1].y / ydim, slide_num / ( z_compression * len(slide_stack))
                            temp_pix = blob.possible_partners[partner_num].edge_pixels[edgep2]
                            contextline_data[index+1] = temp_pix.x / xdim, temp_pix.y / ydim, (slide_num + 1) / ( z_compression * len(slide_stack))
                            # print('Line:' + str(contextline_data[index]) + ' : ' + str(contextline_data[index+1]) + ', index=' + str(index) + ' / ' + str(contextline_count))
                            index += 2
                        else:
                            # print('Overflow, hopefully due to matrix expansion')
                            maxEdge = max(edgep1, edgep2)
                            maxEdgePixels = max(len(blob.edge_pixels), len(blob.possible_partners[partner_num].edge_pixels))
                            if maxEdge > maxEdgePixels:
                                print('\n\n-----ERROR! Pixel number was greater than both edge_pixel lists')
                                debug()
        '''







    if subpixels:
        total_partner_subpixels = 0
        total_self_subpixels = 0
        for slide_num, slide in enumerate(slide_stack[:-1]):
            for blob in slide.blob2dlist:
                for partner in blob.partner_subpixels:
                    print('DEBUG, counting partner length of:' + str(partner))
                    total_partner_subpixels += len(partner)
                for pixel_subset in blob.my_subpixels:
                    total_self_subpixels += len(pixel_subset)
        partner_marker_locations = np.zeros([total_partner_subpixels, 3])
        self_marker_locations = np.zeros([total_self_subpixels, 3])

        partner_index = 0
        self_index = 0
        for slide_num, slide in enumerate(slide_stack[:-1]):
            for blob in slide.blob2dlist:
                for partner_num, partner in enumerate(blob.partner_subpixels): # Each partner is a list,with each element's value representing an index in edge_pixels
                    parnter_blob = blob.possible_partners[partner_num]
                    for partner_edge_index in partner:
                        partner_edge_pixel = parnter_blob.edge_pixels[partner_edge_index]
                        partner_marker_locations[partner_index] = [partner_edge_pixel.x / xdim, partner_edge_pixel.y  / ydim, (slide_num + 1) / ( z_compression * len(slide_stack))]
                        partner_index += 1
                for subset_num, subset in enumerate(blob.my_subpixels):
                    for my_edge_index in subset:
                        my_edge_pixel = blob.edge_pixels[my_edge_index]
                        self_marker_locations[self_index] = [my_edge_pixel.x / xdim, my_edge_pixel.y  / ydim, slide_num / ( z_compression * len(slide_stack))]
                        self_index += 1

        partner_subpixel_markers = visuals.Markers()
        self_subpixel_markers = visuals.Markers()
        partner_subpixel_markers.set_data(partner_marker_locations, edge_color=None, face_color='yellow', size=20)
        self_subpixel_markers.set_data(self_marker_locations, edge_color=None, face_color='green', size=15)

        partner_subpixel_markers.symbol = 'ring'
        self_subpixel_markers.symbol = '+'

        view.add(partner_subpixel_markers)
        view.add(self_subpixel_markers)



    if partnerlines:
        line_count = 0 # Counting the number of line segments to be drawn
        for slide in slide_stack:
            for blob in slide.blob2dlist:
                line_count += len(blob.possible_partners)
        line_count *= 2 # Because need start and end points
        line_locations = np.zeros([line_count, 1, 3])
        index = 0
        for slide_num, slide in enumerate(slide_stack[:-1]): # All but the lat slide
            for blob in slide.blob2dlist:
                # print('DB: Slide:' + str(slide_num) + ' Blob:' + str(blob) + ' partners:' + str(blob.possible_partners))
                for partner in blob.possible_partners:
                    line_locations[index] = [blob.avgx / xdim, blob.avgy / ydim, slide_num / ( z_compression * len(slide_stack))]
                    line_locations[index + 1] = [partner.avgx / xdim, partner.avgy / ydim, (slide_num + 1) / ( z_compression * len(slide_stack))]
                    index += 2
        possible_lines.set_data(pos=line_locations, connect='segments')
        view.add(possible_lines)

    if contextlines: # TODO

        contextline_count = 0
        for slide in slide_stack:
            for blob in slide.blob2dlist:
                for partner in blob.partner_indeces:


                    # if dePickle:
                    #     partner = partner[0] # NOTE HACK remove this once re-pickled
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

                    # if dePickle:
                    #     partner = partner[0] # NOTE HACK remove this once re-pickled
                    # print('partner: ' + str(partner))
                    for edgep1, edgep2 in partner:
                        # print(str(edgep1) + ' / ' + str(len(blob.edge_pixels)) + ' : ' +  str(edgep2) + ' / ' + str(len(blob.possible_partners[partner_num].edge_pixels)))
                        if edgep1 < len(blob.edge_pixels) and edgep2 < len(blob.possible_partners[partner_num].edge_pixels):
                            contextline_data[index] = blob.edge_pixels[edgep1].x / xdim, blob.edge_pixels[edgep1].y / ydim, slide_num / ( z_compression * len(slide_stack))
                            temp_pix = blob.possible_partners[partner_num].edge_pixels[edgep2]
                            contextline_data[index+1] = temp_pix.x / xdim, temp_pix.y / ydim, (slide_num + 1) / ( z_compression * len(slide_stack))
                            # print('Line:' + str(contextline_data[index]) + ' : ' + str(contextline_data[index+1]) + ', index=' + str(index) + ' / ' + str(contextline_count))
                            index += 2
                        else:
                            # print('Overflow, hopefully due to matrix expansion')
                            maxEdge = max(edgep1, edgep2)
                            maxEdgePixels = max(len(blob.edge_pixels), len(blob.possible_partners[partner_num].edge_pixels))
                            if maxEdge > maxEdgePixels:
                                print('\n\n-----ERROR! Pixel number was greater than both edge_pixel lists')
                                debug()
        # print('Done generating contextline_data:' + str(contextline_data))
        context_lines.set_data(pos=contextline_data, connect='segments')
        view.add(context_lines)




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

                possible_lines.set_data(pos=np.dot((line_locations - .5), rotate_arr) + .5, connect='segments')
                for arr in array_list:
                    draw_arrays.append(np.zeros(arr.shape))
                    # buf = np.dot(draw_arrays[-1], rotate_arr)
                    draw_arrays[-1] = np.dot((arr - .5), rotate_arr) + .5

                if coloring is not None:
                    for (a_num, arr) in enumerate(draw_arrays):
                        # scatter_list[a_num].set_data(arr, edge_color=None, face_color=colors[(a_num + frame_num) % len(colors)], size=5)
                        print('Debug setting scatter data to:' + str(arr))
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
    # print('Now starting visuals, but first, slide/ blob info:')
    # for snum, slide in enumerate(slide_stack):
    #     print('Slide: ' + str(snum) + ' / ' + str(len(slide_stack)))
    #     for blob in slide.blob2dlist:
    #         print(' Blob:' + str(blob) + ' which has' + str(len(blob.possible_partners)) + ' possible partners')
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

# NOTE: all marker types:
'''
('*',
 '+',
 '-',
 '->',
 '>',
 '^',
 'arrow',
 'clobber',
 'cross',
 'diamond',
 'disc',
 'hbar',
 'o',
 'ring',
 's',
 'square',
 'star',
 'tailed_arrow',
 'triangle_down',
 'triangle_up',
 'v',
 'vbar',
 'x',
 '|')
 '''