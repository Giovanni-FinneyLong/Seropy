__author__ = 'gio'
# This file includes the various functions written to visualize data from within sero.py
# These functions have been separated for convenience; they are higher volume and lower maintenance.

from myconfig import *
import numpy as np
import time
import math
import pdb
import os
from Blob2d import Blob2d
from Pixel import Pixel


current_path = os.getcwd()
if mayPlot:
    import vispy.io
    import vispy.scene
    from vispy.scene import visuals
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



if mayPlot:
    colors = vispy.color.get_color_names() # ALl possible colors
    # note getting rid of annoying colors
    rejectwords = ['dark', 'light', 'slate', 'grey', 'white', 'pale', 'medium']
    removewords = []
    for knum, key in enumerate(colors):
        for word in rejectwords:
            if len(key) == 1:
                removewords.append(key)
                break
            elif key.find(word) != -1: # found
                removewords.append(key)
                break
    colors = list(set(colors) - set(removewords))
    colors = sorted(colors)
    colors.remove('aliceblue')
    colors.remove('azure')
    colors.remove('blanchedalmond')
    colors.remove('aquamarine')
    colors.remove('beige')
    colors.remove('bisque')
    colors.remove('black')
    colors.remove('blueviolet')
    colors.remove('brown')
    colors.remove('burlywood')
    colors.remove('cadetblue')
    colors.remove('chocolate')
    colors.remove('coral')
    colors.remove('cornsilk')
    colors.remove('cornflowerblue')
    colors.remove('chartreuse')
    colors.remove('crimson')
    colors.remove('cyan')
    colors.remove('deepskyblue')
    colors.remove('dimgray')
    colors.remove('dodgerblue')
    colors.remove('firebrick')
    colors.remove('forestgreen')
    colors.remove('fuchsia')
    colors.remove('gainsboro')
    colors.remove('gold') # Named golden
    colors.remove('goldenrod')
    colors.remove('gray')
    colors.remove('greenyellow')
    colors.remove('honeydew')
    colors.remove('hotpink')
    colors.remove('indianred')
    colors.remove('indigo')
    colors.remove('ivory')
    colors.remove('khaki')
    colors.remove('lavender')
    colors.remove('lavenderblush')
    colors.remove('lawngreen')
    colors.remove('lemonchiffon')
    colors.remove('linen')
    colors.remove('olive')
    colors.remove('olivedrab')
    colors.remove('limegreen')
    colors.remove('midnightblue')
    colors.remove('mintcream')
    colors.remove('mistyrose')
    colors.remove('moccasin')
    colors.remove('navy')
    colors.remove('orangered')
    colors.remove('orchid')
    colors.remove('papayawhip')
    colors.remove('peachpuff')
    colors.remove('peru')
    colors.remove('pink')
    colors.remove('powderblue')
    colors.remove('plum')
    colors.remove('rosybrown')
    colors.remove('saddlebrown')
    colors.remove('salmon')
    colors.remove('sandybrown')
    colors.remove('seagreen')
    colors.remove('seashell')
    colors.remove('silver')
    colors.remove('sienna')
    colors.remove('skyblue')
    colors.remove('springgreen')
    colors.remove('tan')
    colors.remove('teal')
    colors.remove('thistle')
    colors.remove('tomato')
    colors.remove('turquoise')
    colors.remove('snow')
    colors.remove('steelblue')
    colors.remove('violet')
    colors.remove('wheat')
    colors.remove('yellowgreen')
    print('There are a total of ' + str(len(colors)) + ' colors available for plotting')
    # openglconfig = vispy.gloo.wrappers.get_gl_configuration() # Causes opengl/vispy crash for unknown reasons





def showColors(canvas_size=(800,800)):
    global canvas
    global view
    global colors
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, size=canvas_size)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'  # or try 'arcball'
    view.camera.elevation = -55
    view.camera.azimuth = 1
    view.camera.distance = .5
    axis = visuals.XYZAxis(parent=view.scene)
    print(colors)
    print('There are a total of ' + str(len(colors)) + ' colors used for plotting')

    for i,color in enumerate(colors):
        view.add(visuals.Text(color, pos=np.reshape([0, 0, 1-(i / len(colors))], (1,3)), color=color, bold=True))
    vispy.app.run()

def plotPixels(pixellist, canvas_size=(800, 800)):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, size=canvas_size,
                                     title='')
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'  # or try 'arcball'
    view.camera.elevation = -55
    view.camera.azimuth = 1
    view.camera.distance = .1
    xmin = min(pixel.x for pixel in pixellist)
    ymin = min(pixel.y for pixel in pixellist)
    xmax = max(pixel.x for pixel in pixellist)
    ymax = max(pixel.y for pixel in pixellist)
    edge_pixel_array = np.zeros([len(pixellist), 3])
    for (p_num, pixel) in enumerate(pixellist):
        edge_pixel_array[p_num] = [(pixel.x - xmin) / len(pixellist), (pixel.y - ymin) / len(pixellist), pixel.z /  (z_compression * len(pixellist))]
    marker = visuals.Markers()
    marker.set_data(edge_pixel_array, edge_color=None, face_color=colors[0 % len(colors)], size=8)
    view.add(marker)

            # view.add(visuals.Markers(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 ))


    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()

def plotPixelLists(pixellists, canvas_size=(800, 800)): # NOTE works well to show bloom results
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, size=canvas_size,
                                     title='')
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'  # or try 'arcball'
    view.camera.elevation = -55
    view.camera.azimuth = 1
    view.camera.distance = .1
    xmin = min(pixel.x for pixellist in pixellists for pixel in pixellist)
    ymin = min(pixel.y for pixellist in pixellists for pixel in pixellist)
    xmax = max(pixel.x for pixellist in pixellists for pixel in pixellist)
    ymax = max(pixel.y for pixellist in pixellists for pixel in pixellist)
    zmin = min(pixel.z for pixellist in pixellists for pixel in pixellist)
    zmax = max(pixel.z for pixellist in pixellists for pixel in pixellist)
    xdim = xmax - xmin + 1
    ydim = ymax - ymin + 1
    zdim = zmax - zmin + 1

    total_pixels = sum(len(pixellist) for pixellist in pixellists)
    edge_pixel_arrays = []
    markers = []
    # TODO plot all of a color at once

    # for list_num, pixellist in enumerate(pixellists):
    #     edge_pixel_arrays.append(np.zeros([len(pixellist), 3]))
    #     markers.append(visuals.Markers())
    #     for (p_num, pixel) in enumerate(pixellist):
    #         edge_pixel_arrays[-1][p_num] = [(pixel.x - xmin) / xdim, (pixel.y - ymin) / ydim, pixel.z /  (z_compression * zdim)]
    #     markers[-1].set_data(edge_pixel_arrays[-1], edge_color=None, face_color=colors[list_num % len(colors)], size=8)
    #     view.add(markers[-1])


    markers_per_color = [0 for i in range(min(len(colors), len(pixellists)))]
    offsets = [0] * min(len(colors), len(pixellists))
    for blobnum, pixellist in enumerate(pixellists):
        markers_per_color[blobnum % len(markers_per_color)] += len(pixellist)
    for num,i in enumerate(markers_per_color):
        edge_pixel_arrays.append(np.zeros([i, 3]))
    for blobnum, pixellist in enumerate(pixellists):
        index = blobnum % len(markers_per_color)
        for p_num, pixel in enumerate(pixellist):
            edge_pixel_arrays[index][p_num + offsets[index]] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( z_compression * zdim)]
        offsets[index] += len(pixellist)

    print('NUM ARRAYS=' + str(len(edge_pixel_arrays)))
    for color_num, edge_array in enumerate(edge_pixel_arrays):
        markers = visuals.Markers()
        markers.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
        # view.add(visuals.Markers(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 ))
        view.add(markers)
    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()

def isInside(pixel_in, blob2d):
    #NOTE this requires that the blob2d has pixels and edge_pixels fully populated
    if pixel_in in blob2d.pixels: # TODO update with a recursive type call
        if pixel_in in blob2d.edge_pixels:
            return False
        else:
            return True
    else:
        return False
    # May need to optimize this, not sure how slow the above is
    # NOTE will be able to sort this later, to effectively send lines in two directions horizontally


def progressBarUpdate(value, max, min=0, last_update=0, steps=10):
    ''' # TODO not functional
    Run like so:
    updateStatus = 0
    for num in range(100):
        updateStatus = progressBarUpdate(num, 100, last_update=updateStatus)
    :param value:
    :param max:
    :param min:
    :param last_update:
    :param steps:
    :return:
    '''
    if value == min:
        print('.', end='')
    else:
        # print('DB last_update=' + str(last_update) + ' val=' + str(value))
        # print(str((value - last_update)) + ' vs ' + str(((max-min) / steps)))
        if last_update < max:
            if (value - last_update) >= ((max-min) / steps):
                last_update = value;
                print('Diff=' + str((value - last_update)) + ' stepsize:' + str(((max-min) / steps)))
                print('Val' + str(value))
                # for i in range( math.ceil((value - last_update) / ((max-min) / steps))):
                print('.', end='')
        # if value >= max:
        #     print('', end='\n')
    return last_update

def setMasterStartTime():
    master_start_time = time.time() # FIXME!

def plotBlob3d(blob3d, coloring='', b2dids=False,canvas_size=(800,800), **kwargs):
    global canvas
    global view
    global colors
    # FIXME TODO can remove this once repickled


    blob3d.minx = Blob2d.get(blob3d.blob2ds[0]).getminx()
    blob3d.miny = Blob2d.get(blob3d.blob2ds[0]).getminy()
    blob3d.maxx = Blob2d.get(blob3d.blob2ds[0]).getmaxx()
    blob3d.maxy = Blob2d.get(blob3d.blob2ds[0]).getmaxy()

    # NOTE TODO add this to blob3d
    midx = 0
    midy = 0

    for blob in blob3d.blob2ds:
        blob = Blob2d.get(blob)
        blob3d.minx = min(blob.getminx(), blob3d.minx)
        blob3d.miny = min(blob.getminy(), blob3d.miny)
        blob3d.maxx = max(blob.getmaxx(), blob3d.maxx)
        blob3d.maxy = max(blob.getmaxy(), blob3d.maxy)
        midx += blob.getavgx()
        midy += blob.getavgy()
    midx /= len(blob3d.blob2ds)
    midy /= len(blob3d.blob2ds)
    midz = (blob3d.highslideheight + blob3d.lowslideheight) / 2
    xdim = blob3d.maxx - blob3d.minx
    ydim = blob3d.maxy - blob3d.miny
    zdim = blob3d.highslideheight - blob3d.lowslideheight

    # /TODO
    canvas_size = kwargs.get('canvas_size', (800,800))
    translate = kwargs.get('translate', True) # Offset the blob towards the origin along the x,y axis
    display_costs = kwargs.get('costs', False)


    if translate:
        offsetx = blob3d.minx
        offsety = blob3d.miny
    else:
        offsetx = 0
        offsety = 0



    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, size=canvas_size)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'  # or try 'arcball'
    view.camera.elevation = -55
    view.camera.azimuth = 1
    view.camera.distance = .5

    # #DEBUG
    # camera_attr = dir(view.camera)
    # print('All camera attr: ' + str(camera_attr))

    # TODO: timer = vispy.app.Timer(interval=0, connect=update_camera, start=True, iterations=frames+1)


    print('Camera currently at distance:' + str(view.camera.distance))

    #     view.add(visuals.Text(str(blob.id) + ':' + str(index), pos=avg_list[index], color='white'))
    axis = visuals.XYZAxis(parent=view.scene)
    midpoints = []
    # FIXME This is repairing incorrect averages in blob2ds
    for b2d in blob3d.blob2ds:
        b2d = Blob2d.get(b2d)
        b2d.avgx = sum(Pixel.get(pixel).x for pixel in b2d.edge_pixels) / len(b2d.edge_pixels)
        b2d.avgy = sum(Pixel.get(pixel).y for pixel in b2d.edge_pixels) / len(b2d.edge_pixels)
    # /FIXME


    if b2dids is True:
        midpoints.append(np.zeros([1,3]))
        for b2d_num, b2d in enumerate(blob3d.blob2ds): #FIXME! For some reason overloads the ram.
            midpoints[-1] = [(b2d.avgx - offsetx) / xdim, (b2d.avgy - offsety) / ydim, b2d.height / (z_compression * zdim)]
            textStr = str(b2d_num)
            if coloring == 'blob2d':
                color = colors[b2d_num]
            else:
                color = 'yellow'
            view.add(visuals.Text(textStr, pos=midpoints[-1], color=color))


    if coloring == 'blob2d':
        edge_pixel_arrays = []
        markers = []
        for b2d_num, blob2d in enumerate(blob3d.blob2ds):
            edge_pixel_arrays.append(np.zeros([len(blob2d.edge_pixels), 3]))
            markers.append(visuals.Markers())
            for (p_num, pixel) in enumerate(blob2d.edge_pixels):
                edge_pixel_arrays[-1][p_num] = [(pixel.x - offsetx) / xdim, (pixel.y - offsety) / ydim, pixel.z /  (z_compression * zdim)]
            markers[-1].set_data(edge_pixel_arrays[-1], edge_color=None, face_color=colors[b2d_num], size=8)
            view.add(markers[-1])
    else:
        edge_pixel_array = np.zeros([blob3d.get_edge_pixel_count(), 3])

        for (p_num, pixel) in enumerate(blob3d.get_edge_pixels()):
                edge_pixel_array[p_num] = [(pixel.x - offsetx) / xdim, (pixel.y - offsety) / ydim, pixel.z /  (z_compression * zdim)]
        markers = visuals.Markers()
        markers.set_data(edge_pixel_array, edge_color=None, face_color='green', size=8)
        view.add(markers)
    lineendpoints = 0
    for pairing in blob3d.pairings:
        lineendpoints += (2 * len(pairing.indeces))


    line_index = 0
    line_locations = np.zeros([lineendpoints, 3])

    for pairing in blob3d.pairings:
        for lowerpnum, upperpnum in pairing.indeces:
            lowerpixel = pairing.lowerpixels[lowerpnum]
            upperpixel = pairing.upperpixels[upperpnum]
            line_locations[line_index] = [(lowerpixel.x - offsetx) / xdim, (lowerpixel.y - offsety) / ydim, (pairing.lowerslidenum) / ( z_compression * zdim)]
            line_locations[line_index + 1] = [(upperpixel.x - offsetx) / xdim, (upperpixel.y - offsety) / ydim, (pairing.upperslidenum) / ( z_compression * zdim)]
            line_index += 2
    lower_markers = visuals.Markers()
    upper_markers = visuals.Markers()
    stitch_lines = visuals.Line(method=linemethod)
    stitch_lines.set_data(pos=line_locations, connect='segments')
    view.add(stitch_lines)
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

def getBloomedHeight(b2d, explode, zdim):
    if explode:
        return b2d.height + b2d.recursive_depth / (zdim * max([len(b2d.getrelated()), 1]))
    else:
        return b2d.height

def plotBlob2ds(blob2ds, coloring='', canvas_size=(1080,1080), ids=False, stitches=False, titleNote='', edge=True, parentlines=False, explode=False):
    global canvas
    global view
    global colors

    assert coloring.lower() in ['blob2d', '', 'depth', 'blob3d']

    all_b2ds_are_ids = all(type(b2d) is int for b2d in blob2ds)
    all_b2d_are_blob2ds = all(type(b2d) is Blob2d for b2d in blob2ds)
    # assert(all(type(pixel) is int for b2d in blob2ds for pixel in Blob2d.get(b2d).pixels))
    assert(all_b2d_are_blob2ds or all_b2ds_are_ids)
    if all_b2ds_are_ids: # May want to change this depending on what it does to memory
        blob2ds = [Blob2d.get(b2d) for b2d in blob2ds]


    xmin = min(blob2d.minx for blob2d in blob2ds)
    ymin = min(blob2d.miny for blob2d in blob2ds)
    xmax = max(blob2d.maxx for blob2d in blob2ds)
    ymax = max(blob2d.maxy for blob2d in blob2ds)
    zmin = min(blob2d.height for blob2d in blob2ds)
    zmax = max(blob2d.height for blob2d in blob2ds)

    xdim = xmax - xmin + 1
    ydim = ymax - ymin + 1
    zdim = zmax - zmin + 1



    if explode:
        if not all(b2d.height == blob2ds[0].height for b2d in blob2ds):
            warn('Attempting to explode blob2ds that are not all at the same height')
        # for b2d in blob2ds:
        #     b2d.height += b2d.recursive_depth / (zdim * max([len(b2d.getrelated()), 1])) # HACK 1.2 to try to correct allignment, may need function worst case



    if coloring == '':
        coloring = 'blob2d' # For the canvas title
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, size=canvas_size,
                                     title='plotBlob2ds(' + str(len(blob2ds)) + '-Blob2ds, coloring=' + str(coloring) + ' canvas_size=' + str(canvas_size) + ') ' + titleNote)
    view = canvas.central_widget.add_view()
    pixel_arrays = []


    if coloring == 'blob2d':
        markers_per_color = [0 for i in range(min(len(colors), len(blob2ds)))]
        offsets = [0] * min(len(colors), len(blob2ds))
        if edge:
            for blobnum, blob2d in enumerate(blob2ds):
                markers_per_color[blobnum % len(markers_per_color)] += len(blob2d.edge_pixels)
        else:
            for blobnum, blob2d in enumerate(blob2ds):
                markers_per_color[blobnum % len(markers_per_color)] += len(blob2d.pixels)
        for num,i in enumerate(markers_per_color):
            pixel_arrays.append(np.zeros([i, 3]))
        for blobnum, blob2d in enumerate(blob2ds):
            index = blobnum % len(markers_per_color)

            if edge:
                for p_num, pixel in enumerate(blob2d.edge_pixels):
                    pixel = Pixel.get(pixel)
                    pixel_arrays[index][p_num + offsets[index]] = [(pixel.x - xmin) / xdim, (pixel.y - ymin) / ydim, (getBloomedHeight(blob2d, explode, zdim) - zmin) / ( z_compression * zdim)]
                offsets[index] += len(blob2d.edge_pixels)
            else:
                for p_num, pixel in enumerate(blob2d.pixels):
                    pixel = Pixel.get(pixel)
                    pixel_arrays[index][p_num + offsets[index]] = [(pixel.x - xmin) / xdim, (pixel.y - ymin) / ydim, (getBloomedHeight(blob2d, explode, zdim)  - zmin) / ( z_compression * zdim)]
                offsets[index] += len(blob2d.pixels)

        for color_num, edge_array in enumerate(pixel_arrays):
            buf = visuals.Markers()
            buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
            view.add(buf)
    elif coloring == 'blob3d':
        edge_pixel_arrays = [] # One array per 3d blob
        max_b3d_id = max(b2d.b3did for b2d in blob2ds)
        b3d_lists = [[] for i in range(max_b3d_id + 1)]
        for b2d in blob2ds:
            b3d_lists[b2d.b3did].append(b2d)
        b3d_lists = [b3d_list for b3d_list in b3d_lists if len(b3d_list)]
        print('Total number of b3ds from b2ds:' + str(len(b3d_lists)))

        markers_per_color = [0 for i in range(min(len(colors), len(b3d_lists)))]
        offsets = [0] * min(len(colors), len(b3d_lists))
        for blobnum, b3d_list in enumerate(b3d_lists):
            markers_per_color[blobnum % len(markers_per_color)] += sum([len(b2d.edge_pixels) for b2d in b3d_list])

        for num,i in enumerate(markers_per_color):
            edge_pixel_arrays.append(np.zeros([i, 3]))

        for blobnum, b3d_list in enumerate(b3d_lists):
            index = blobnum % len(markers_per_color)
            for p_num, pixel in enumerate(Pixel.get(pixel) for b2d in b3d_list for pixel in b2d.edge_pixels):
                edge_pixel_arrays[index][p_num + offsets[index]] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( z_compression * zdim)]
            offsets[index] += sum([len(b2d.edge_pixels) for b2d in b3d_list])
        for color_num, edge_array in enumerate(edge_pixel_arrays):
            buf = visuals.Markers()
            buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
            view.add(buf)

    else:
        # DEPTH # TODO TODO TODO FIX THIS, issue when plotting with multiple depths (plotting d0 works)
        max_depth = max(blob2d.recursive_depth for blob2d in blob2ds if hasattr(blob2d, 'recursive_depth'))
        min_depth = min(blob2d.recursive_depth for blob2d in blob2ds if hasattr(blob2d, 'recursive_depth'))
        markers_per_color = [0 for i in range(min(len(colors), max_depth + 1))]
        offsets = [0] * min(len(colors), max_depth + 1)
        for blobnum, blob2d in enumerate(blob2ds):
            markers_per_color[blob2d.recursive_depth % len(markers_per_color)] += len(blob2d.edge_pixels)
        for num,i in enumerate(markers_per_color):
            pixel_arrays.append(np.zeros([i, 3]))
        for blobnum, blob2d in enumerate(blob2ds):
            index = blob2d.recursive_depth % len(markers_per_color)
            for p_num, pixel in enumerate(blob2d.edge_pixels):
                pixel = Pixel.get(pixel)
                pixel_arrays[index][p_num + offsets[index]] = [(pixel.x - xmin) / xdim, (pixel.y - ymin) / ydim, (getBloomedHeight(blob2d, explode, zdim)  - zmin) / ( z_compression * zdim)]
            offsets[index] += len(blob2d.edge_pixels)

        for color_num, edge_array in enumerate(pixel_arrays):
            if len(edge_array) == 0:
                print('Skipping plotting depth ' + str(color_num) + ' as there are no blob2ds at that depth')
            else:
                buf = visuals.Markers()
                buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
                view.add(buf)


    if ids is True:
        midpoints = []
        midpoints.append(np.zeros([1,3]))
        for b2d_num, b2d in enumerate(blob2ds):
            midpoints[-1] = [(b2d.avgx - xmin) / xdim, (b2d.avgy - ymin) / ydim, ((getBloomedHeight(b2d, explode, zdim)  + .25 - zmin) / (z_compression * zdim))]
            textStr = str(b2d.id)
            if coloring == '' or coloring == 'blob2d':
                color = colors[b2d_num % len(colors)]
            else:
                if coloring in colors:
                    color = coloring
                else:
                    color = 'yellow'
            view.add(visuals.Text(textStr, pos=midpoints[-1], color=color, font_size=15, bold=True))
    if stitches:
        lineendpoints = 0
        for blob2d in blob2ds:
            for pairing in blob2d.pairings:
                lineendpoints += (2 * len(pairing.indeces))
        if lineendpoints != 0:
            line_index = 0
            line_locations = np.zeros([lineendpoints, 3])
            for blob2d in blob2ds:
                for pairing in blob2d.pairings:
                    # print('DB pairing indeces:' + str(pairing.indeces))
                    # print('DB pairing lowerpixels:' + str(pairing.lowerpixels))


                    for lowerpnum, upperpnum in pairing.indeces:
                        lowerpixel = Pixel.get(pairing.lowerpixels[lowerpnum])
                        upperpixel = Pixel.get(pairing.upperpixels[upperpnum])
                        line_locations[line_index] = [(lowerpixel.x - xmin) / xdim, (lowerpixel.y - ymin) / ydim, (pairing.lowerheight) / ( z_compression * zdim)]
                        line_locations[line_index + 1] = [(upperpixel.x - xmin) / xdim, (upperpixel.y - ymin) / ydim, (pairing.upperheight) / ( z_compression * zdim)]
                        line_index += 2
            stitch_lines = visuals.Line(method=linemethod)
            stitch_lines.set_data(pos=line_locations, connect='segments')
            view.add(stitch_lines)
    if parentlines:
        lineendpoints = 0
        for num,b2d in enumerate(blob2ds):
            lineendpoints += (2 * len(b2d.children))
        if lineendpoints:
            line_index = 0
            line_locations = np.zeros([lineendpoints, 3])
            for b2d in blob2ds:
                for child in b2d.children:
                    child = Blob2d.get(child)
                    line_locations[line_index] = [(b2d.avgx - xmin) / xdim, (b2d.avgy - ymin) / ydim, (getBloomedHeight(b2d, explode, zdim)  - zmin) / ( z_compression * zdim)]
                    line_locations[line_index + 1] = [(child.avgx - xmin) / xdim, (child.avgy - ymin) / ydim, (getBloomedHeight(child, explode, zdim) - zmin) / ( z_compression * zdim)]
                    line_index += 2
            parent_lines = visuals.Line(method=linemethod)
            parent_lines.set_data(pos=line_locations, connect='segments', color='y')
            view.add(parent_lines)

    axis = visuals.XYZAxis(parent=view.scene)
    view.camera = 'turntable'  # or try 'arcball'
    view.camera.elevation = -55
    view.camera.azimuth = 1
    view.camera.distance = .5
    vispy.app.run()

def plotBlob3ds(blob3dlist, stitches=True, color=None, lineColoring=None, costs=0, maxcolors=-1, b2dmidpoints=False, b3dmidpoints=False, canvas_size=(800, 800), b2d_midpoint_values=0, titleNote=''):
    global canvas
    global view
    global colors
    # canvas_size = kwargs.get('canvas_size', (800,800))
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, size=canvas_size,
                                     title='plotBlob3ds(' + str(len(blob3dlist)) + '-Blob3ds, coloring=' + str(color) + ', canvas_size=' + str(canvas_size) + ') ' + titleNote)
    if maxcolors > 0 and maxcolors < len(colors):
        colors = colors[:maxcolors]

    # Finding the maximal slide, so that the vertical dimension of the plot can be evenly divided
    total_slides = 0
    xdim = 0
    ydim = 0
    zdim = 0

    for blob3d in blob3dlist: # TODO make gen functions
        if blob3d.highslideheight > total_slides:
            total_slides = blob3d.highslideheight
        if blob3d.highslideheight > zdim:
            zdim = blob3d.highslideheight
        if blob3d.maxx > xdim:
            xdim = blob3d.maxx
        if blob3d.maxy > ydim:
            ydim = blob3d.maxy
    total_slides += 1 # Note this is b/c numbering starts at 0
    zdim += 1

    view = canvas.central_widget.add_view()
    view.camera = 'turntable'  # or try 'arcball'
    view.camera.elevation = -75
    view.camera.azimuth = 1

    axis = visuals.XYZAxis(parent=view.scene)
    edge_pixel_arrays = [] # One array per 3d blob
    markerlist = []

    lineendpoints = 0


    if color == 'blob': # Note: This is very graphics intensive.
        markers_per_color = [0 for i in range(min(len(colors), len(blob3dlist)))]
        offsets = [0] * min(len(colors), len(blob3dlist))
        for blobnum, blob3d in enumerate(blob3dlist):
            # markers_per_color[blobnum % len(markers_per_color)] += len(blob3d.edge_pixels)
            markers_per_color[blobnum % len(markers_per_color)] += blob3d.get_edge_pixel_count()


        for num,i in enumerate(markers_per_color):
            edge_pixel_arrays.append(np.zeros([i, 3]))
        for blobnum, blob3d in enumerate(blob3dlist):
            index = blobnum % len(markers_per_color)
            for p_num, pixel in enumerate(blob3d.get_edge_pixels()):
                edge_pixel_arrays[index][p_num + offsets[index]] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( z_compression * total_slides)]
            offsets[index] += blob3d.get_edge_pixel_count()
        for color_num, edge_array in enumerate(edge_pixel_arrays):
            buf = visuals.Markers()
            buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
            view.add(buf)

        # for blob_num, blob3d in enumerate(blob3dlist):
        #     edge_pixel_arrays.append(np.zeros([len(blob3d.edge_pixels), 3]))
        #     for (p_num, pixel) in enumerate(blob3d.edge_pixels):
        #         edge_pixel_arrays[-1][p_num] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( z_compression * total_slides)]
        #     # midpoints[blob_num] = [blob3d.avgx, blob3d.avgy, blob3d.avgz]
        #     markerlist.append(visuals.Markers())
        #     markerlist[-1].set_data(edge_pixel_arrays[-1], edge_color=None, face_color=colors[blob_num % len(colors)], size=8)
        #     print('DB blob #' + str(blob3d.id) + ' is colored ' + str(colors[blob_num % len(colors)]))
        #     view.add(markerlist[-1])

    elif color == 'singular':
        total_singular_points = 0
        total_multi_points = 0 # Points from blob3ds that may be part of strands
        for blob3d in blob3dlist:
            if blob3d.isSingular:
                total_singular_points += len(blob3d.edge_pixels)
            else:
                total_multi_points += len(blob3d.edge_pixels)
        singular_edge_array = np.zeros([total_singular_points, 3])
        multi_edge_array = np.zeros([total_multi_points, 3])
        singular_index = 0
        multi_index = 0
        for blob_num, blob3d in enumerate(blob3dlist):
            # print('DB processing blob:' + str(blob_num) + ' / ' + str(len(blob3dlist)))
            if blob3d.isSingular:
                for pixel in blob3d.edge_pixels:
                    singular_edge_array[singular_index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (z_compression * total_slides)]
                    singular_index += 1
            else:
                for pixel in blob3d.edge_pixels:
                    multi_edge_array[multi_index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (z_compression * total_slides)]
                    multi_index += 1
            # for stitch in blob3d.pairings:
            #     lineendpoints += (2 * len(stitch.indeces))
        singular_markers = visuals.Markers()
        multi_markers = visuals.Markers()
        singular_markers.set_data(singular_edge_array, edge_color=None, face_color='green', size=8)
        multi_markers.set_data(multi_edge_array, edge_color=None, face_color='red', size=8)
        view.add(singular_markers)
        view.add(multi_markers)
    elif color == 'depth': # Coloring based on recursive depth
        # HACK can be removed when repickled # FIXME
        # print('Coloring based on depth')
        max_depth = max(blob.recursive_depth for blob in blob3dlist)
        # NOTE because of sorting, this needs to be done before any info (like midpoints) is extracted from blob3dslist
        blob3dlist = sorted(blob3dlist, key=lambda blob: blob.recursive_depth, reverse=False) # Now sorted by depth, lowest first (primary)
        cur_depth = blob3dlist[0].recursive_depth
        current_index = 0
        b3ds_by_depth = [] # A list of lists, each sublist containing all b3ds the depth of the lists's index
        for cur_depth in range(max_depth + 1): # +1 for endcap
            b3ds_at_depth = []
            while(current_index != len(blob3dlist) and blob3dlist[current_index].recursive_depth == cur_depth):
                b3ds_at_depth.append(blob3dlist[current_index])
                current_index += 1
            b3ds_by_depth.append(b3ds_at_depth)
        for depth, depth_list in enumerate(b3ds_by_depth):
            num_edge_pixels_at_depth = sum(b3d.get_edge_pixel_count() for b3d in depth_list)
            edge_pixel_arrays.append(np.zeros([num_edge_pixels_at_depth, 3]))
            p_num = 0
            for b3d in depth_list:
                ep_buf = b3d.get_edge_pixels()
                for pixel in ep_buf:
                    edge_pixel_arrays[-1][p_num] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( z_compression * total_slides)]
                    p_num += 1

            markerlist.append(visuals.Markers())
            markerlist[-1].set_data(edge_pixel_arrays[-1], edge_color=None, face_color=colors[depth % len(colors)], size=8)
            # print('Using color: ' + str(colors[depth % len(colors)]))
            view.add(markerlist[-1])


    # all_stitches = sorted(all_stitches, key=lambda stitch: stitch.cost[2], reverse=True) # costs are (contour_cost, distance(as cost), total, distance(not as cost))


    else: # All colored the same
        total_points = 0
        for blob_num, blob3d in enumerate(blob3dlist):
            total_points += blob3d.get_edge_pixel_count()
        edge_pixel_array = np.zeros([total_points, 3])
        index = 0
        for blob3d in blob3dlist:
            ep_buf = blob3d.get_edge_pixels()
            for pixel in ep_buf:
                edge_pixel_array[index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (z_compression * total_slides)]
                index += 1
            for stitch in blob3d.pairings:
                lineendpoints += (2 * len(stitch.indeces)) # 2 as each line has 2 endpoints

        markers = visuals.Markers()
        markers.set_data(edge_pixel_array, edge_color=None, face_color=colors[0], size=8) # TODO change color
        view.add(markers)

    if costs > 0:

        number_of_costs_to_show = costs # HACK
        all_stitches = list(stitches for blob3d in blob3dlist for pairing in blob3d.pairings for stitches in pairing.stitches)
        all_stitches = sorted(all_stitches, key=lambda stitch: stitch.cost[2], reverse=True) # costs are (contour_cost, distance(as cost), total, distance(not as cost))
        midpoints = np.zeros([number_of_costs_to_show,3])
        for index,stitch in enumerate(all_stitches[:number_of_costs_to_show]): #FIXME! For some reason overloads the ram.
            midpoints[index] = [(stitch.lowerpixel.x + stitch.upperpixel.x) / (2 * xdim), (stitch.lowerpixel.y + stitch.upperpixel.y) / (2 * ydim), (stitch.lowerpixel.z + stitch.upperpixel.z) / (2 * zdim)]
            textStr = str(stitch.cost[0])[:2] + '_' +  str(stitch.cost[3])[:3] + '_' +  str(stitch.cost[2])[:2]
            view.add(visuals.Text(textStr, pos=midpoints[index], color='yellow'))


    if stitches:
        if lineColoring == 'blob3d':
            line_location_lists = []
            stitch_lines = []
            for blob_num, blob3d in enumerate(blob3dlist):
                lineendpoints = 2 * sum(len(pairing.indeces) for blob3d in blob3dlist for pairing in blob3d.pairings)
                line_location_lists.append(np.zeros([lineendpoints, 3]))
                line_index = 0
                for pairing in blob3d.pairings:
                    for stitch in pairing.stitches:
                        lowerpixel = stitch.lowerpixel
                        upperpixel = stitch.upperpixel
                        line_location_lists[-1][line_index] = [lowerpixel.x / xdim, lowerpixel.y / ydim, (pairing.lowerslidenum ) / ( z_compression * total_slides)]
                        line_location_lists[-1][line_index + 1] = [upperpixel.x / xdim, upperpixel.y / ydim, (pairing.upperslidenum ) / ( z_compression * total_slides)]
                        line_index += 2
                stitch_lines.append(visuals.Line(method=linemethod))
                stitch_lines[-1].set_data(pos=line_location_lists[-1], connect='segments', color=colors[blob_num % len(colors)])
                view.add(stitch_lines[-1])
        else:
            line_index = 0
            for blob_num, blob3d in enumerate(blob3dlist):
                for stitch in blob3d.pairings:
                    lineendpoints += (2 * len(stitch.indeces)) # 2 as each line has 2 endpoints
            line_locations = np.zeros([lineendpoints, 3])
            for blob3d in blob3dlist:
                for pairing in blob3d.pairings:
                    for stitch in pairing.stitches:
                        lowerpixel = Pixel.get(stitch.lowerpixel)
                        upperpixel = Pixel.get(stitch.upperpixel)
                        line_locations[line_index] = [lowerpixel.x / xdim, lowerpixel.y / ydim, (pairing.lowerheight ) / ( z_compression * total_slides)]
                        line_locations[line_index + 1] = [upperpixel.x / xdim, upperpixel.y / ydim, (pairing.upperheight ) / ( z_compression * total_slides)]
                        line_index += 2
            stitch_lines = visuals.Line(method=linemethod)
            stitch_lines.set_data(pos=line_locations, connect='segments')
            view.add(stitch_lines)





    if b3dmidpoints:
        b3d_midpoint_markers = []
        for blob_num, blob3d in enumerate(blob3dlist):
            b3d_midpoint_markers.append(visuals.Markers())
            b3d_midpoint_markers[-1].set_data(np.array([[blob3d.avgx / xdim, blob3d.avgy / ydim, blob3d.avgz / zdim]]), edge_color='w', face_color=colors[blob_num % len(colors)], size=25)
            b3d_midpoint_markers[-1].symbol = 'star'
            view.add(b3d_midpoint_markers[-1])
    if b2dmidpoints:
        b2d_num = 0
        b2d_count = sum(len(b3d.blob2ds) for b3d in blob3dlist)
        b2d_midpoint_pos = np.zeros([b2d_count, 3])

        for blob3d in blob3dlist:
            for blob2d in blob3d.blob2ds:
                b2d_midpoint_pos[b2d_num] = [blob2d.avgx / xdim, blob2d.avgy / ydim, blob2d.height / zdim]
                b2d_num += 1

        b2d_midpoint_markers = visuals.Markers()
        b2d_midpoint_markers.set_data(b2d_midpoint_pos, edge_color='w', face_color='yellow', size=15)
        b2d_midpoint_markers.symbol = 'diamond'
        view.add(b2d_midpoint_markers)

    if b2d_midpoint_values > 0:

        max_midpoints = b2d_midpoint_values
        print('The midpoints texts are the number of edge_pixels in the Blob2d, showing a total of ' + str(max_midpoints))
        #HACK
        b2d_count = sum(len(b3d.blob2ds) for b3d in blob3dlist)
        b2d_midpoint_textmarkers = []
        b2d_midpoint_pos = np.zeros([b2d_count, 3])
        b2d_num = 0

        blob2dlist = list(b2d for b3d in blob3dlist for b2d in b3d.blob2ds)
        blob2dlist = sorted(blob2dlist, key=lambda blob2d: len(blob2d.edge_pixels), reverse=False)
        # all_stitches = sorted(all_stitches, key=lambda stitch: stitch.cost[2], reverse=True) # costs are (contour_cost, distance(as cost), total, distance(not as cost))

        for b2d_num, b2d in enumerate(blob2dlist[0::3][:max_midpoints]): # GETTING EVERY Nth RELEVANT INDEX
            b2d_midpoint_pos[b2d_num] = [b2d.avgx / xdim, b2d.avgy / ydim, b2d.height / zdim]
            b2d_midpoint_textmarkers.append(visuals.Text(str(len(b2d.edge_pixels)), pos=b2d_midpoint_pos[b2d_num], color='yellow'))
            view.add(b2d_midpoint_textmarkers[-1])

    vispy.app.run()


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