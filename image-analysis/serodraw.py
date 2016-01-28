__author__ = 'gio'
# This file includes the various functions written to visualize data from within sero.py
# These functions have been separated for convenience; they are higher volume and lower maintenance.

from myconfig import *
import numpy as np
from Blob2d import Blob2d
from Pixel import Pixel
import vispy.io
import vispy.scene
from vispy.scene import visuals
from util import warn, debug
colors = None


# TODO sample animation code here: https://github.com/vispy/vispy/blob/master/examples/basics/scene/save_animation.py
# TODO sample event code: https://github.com/vispy/vispy/blob/master/examples/tutorial/app/app_events.py

class Canvas(vispy.scene.SceneCanvas):
    def __init__(self, canvas_size=(800,800), title='', coloring='blob2d', buffering=True): # Note may want to make buffering default to False
        vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True, size=canvas_size, title=title)
        if hasattr(self,'unfreeze') and callable(getattr(self,'unfreeze')):         #HACK # Interesting bug fix for an issue that only occurs on Envy
            self.unfreeze()
        self.view = self.central_widget.add_view()
        camera = vispy.scene.cameras.TurntableCamera(fov=0, azimuth=80, parent=self.view.scene, distance=1, elevation=-55)
        self.axis = visuals.XYZAxis(parent=self.view.scene)
        self.view.camera = camera
        self.blob2d_coloring_markers = []
        self.blob3d_coloring_markers = []
        self.depth_coloring_markers = []
        self.show()
        self.coloring = coloring.lower()
        # self.markers = [self.depth_coloring_markers, self.blob2d_coloring_markers, self.blob3d_coloring_markers]
        self.markers = []
        self.available_marker_colors = ['depth', 'blob2d', 'blob3d']
        self.available_stitch_colors = ['neutral', 'parent', 'none']
        self.current_blob_color = self.coloring
        self.buffering = buffering
        self.marker_colors = [] # Each entry corresponds to the color of the correspond 'th marker in self.view.scene.children (not all markers!)
        self.image_no = 0
        self.b2d_count = -1
        self.b3d_count = -1
        self.stitches = []
        self.current_stitch_color = 'neutral'


    def on_mouse_press(self, event):
        """Pan the view based on the change in mouse position."""
        if event.button == 1:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            print(x1,y1)
            # print (self.view.scene.node_transform(self.canvas_cs).simplified())
            # print (self.view.scene.node_transform(self.canvas_cs).map((x1,y1)))

            # for key,val in self.__dict__.items():
            #     print(str(key) + ' ' + str(val))

            # for key,val in self.view.__dict__.items():
            #     print(str(key) + ' ' + str(val))
            # for key,val in self.view.scene.__dict__.items():
            #     print(str(key) + ' ' + str(val))
            # print(type(self.view.scene.children))
            # print(len(self.view.scene.children))
            # self.view.scene.children = [child for child in self.view.scene.children if type(child) is visuals.Markers]
            # print(len(self.view.scene.children))

    def on_key_press(self, event):
        modifiers = [key.name for key in event.modifiers]
        print('Key pressed - text: %r, key: %s, modifiers: %r' % (
            event.text, event.key.name, modifiers))
        if event.key.name == 'Up': # Next color cheme
            self.update_markers(increment=1)
        elif event.key.name == 'Down': # Previous color scheme
            self.update_markers(increment=-1)
        elif event.key.name == 'S': # Save an image
            img = self.render()
            img_name = self.name_image()
            print('Writing to image file: \'' + str(img_name) + '\'')
            vispy.io.write_png(img_name, img)
        elif event.key.name == 'T': # test
            for child,coloring in self.markers:
                print(child)
                for name, val in child.__dict__.items():
                    print('   ' +str(name) + ' : ' + str(val))
                    debug()
        elif event.key.name == 'Left': # Toggle stitches
            self.update_stitches(increment=1)

    def next_marker_color(self, increment=1):
        assert(increment in [1,-1])
        return self.available_marker_colors[(self.available_marker_colors.index(self.current_blob_color) + increment) % len(self.available_marker_colors)]

    def next_stitch_color(self, increment=1):
        assert(increment in [1,-1])
        return self.available_stitch_colors[(self.available_stitch_colors.index(self.current_stitch_color) + increment) % len(self.available_stitch_colors)]

    def update_stitches(self, increment=1):
        print('Old stitch color: ' + str(self.current_stitch_color))
        self.current_stitch_color = self.next_stitch_color(increment=increment)
        print('New stitch color: ' + str(self.current_stitch_color))
        for child, color in self.stitches:
            print('Color = ' + str( color))
            if color == self.current_stitch_color:# and \
                    # not (self.current_stitch_color == 'blob3d' and self.current_blob_color == 'blob3d' and color == 'parent'): #Hides parent lines when plotting blob3d b/c exploding is turned off
                print('stitch.visible = ' + str(child.visible))
                child.visible = True
            else:
                child.visible = False

    def update_markers(self, increment=1):
        assert increment in [-1, 1]
        print('Going from ' + str(self.current_blob_color) + ' to ' + str(self.next_marker_color()))


        # if self.next_marker_color(increment=increment) == 'blob3d':
        #     if self.current_stitch_color == 'parent':
        #         for child,color in self.stitches:
        #             if color == 'parent':
        #                 child.visible = False
        # if self.current_blob_color == 'blob3d':
        #     if self.current_stitch_color == 'parent':
        #         for child,color in self.stitches:
        #             if color == 'parent':
        #                 child.visible = True



        self.current_blob_color = self.next_marker_color(increment=increment)
        # if self.current_stitch_color == 'parent':
        #     for child,color in self.stitches:
        #         if color == 'parent':
        #             child.visible = self.current_blob_color

        for child,coloring in self.markers:
            if coloring == self.current_blob_color:
                child.visible = False
            if coloring == self.next_marker_color(): #   [(self.current_blob_color + increment) % len(self.available_marker_colors)]:
                child.visible = True


        self.update_title()

    def name_image(self):
        prefix = FIGURES_DIR
        if test_instead_of_data:
            prefix += 'Test_'
        else:
            prefix += 'Data_'
        self.image_no += 1
        return prefix + str(self.image_no - 1) + '.png'

    def setup_markers(self):
        for child, coloring in self.markers:
            if coloring == self.current_blob_color:
                child.visible = True
            elif coloring in self.available_marker_colors:
                child.visible = False
        self.update_title()

    def setup_stitches(self):
        for stitch, color in self.stitches:
            self.view.add(stitch)



    def update_title(self):
        self.title =  '# B3ds: ' + str(self.b3d_count) + ', # B2ds: ' + str(self.b2d_count) + ', Coloring = ' + str(self.current_blob_color)

    def add_marker(self, marker, coloring):
        self.markers.append((marker,coloring))
        self.view.add(self.markers[-1][0]) # add the above marker


def plotBlob2ds(blob2ds, coloring='', canvas_size=(1080,1080), ids=False, stitches=False, titleNote='', edge=True, parentlines=False, explode=False, showStitchCosts=0, b2dmidpoints=False):
    global colors
    coloring = coloring.lower()
    assert coloring in ['blob2d', '', 'b2d_depth', 'blob3d']

    # This block allows the passing of ids or blob2ds
    all_b2ds_are_ids = all(type(b2d) is int for b2d in blob2ds)
    all_b2d_are_blob2ds = all(type(b2d) is Blob2d for b2d in blob2ds)
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

    if coloring == '':
        coloring = 'blob2d' # For the canvas title

    canvas = setupCanvas(canvas_size, title='plotBlob2ds(' + str(len(blob2ds)) + '-Blob2ds, coloring=' + str(coloring) + ' canvas_size=' + str(canvas_size) + ') ' + titleNote)
    canvas.b2d_count = len(blob2ds)
    # if showStitchCosts > 0: #TODO
    #     number_of_costs_to_show = showStitchCosts # HACK
    #     all_stitches = list(stitches for blob3d in blob3dlist for pairing in blob3d.pairings for stitches in pairing.stitches)
    #     all_stitches = sorted(all_stitches, key=lambda stitch: stitch.cost[2], reverse=True) # costs are (contour_cost, distance(as cost), total, distance(not as cost))
    #     midpoints = np.zeros([number_of_costs_to_show,3])
    #     for index,stitch in enumerate(all_stitches[:number_of_costs_to_show]): #FIXME! For some reason overloads the ram.
    #         midpoints[index] = [(stitch.lowerpixel.x + stitch.upperpixel.x) / (2 * xdim), (stitch.lowerpixel.y + stitch.upperpixel.y) / (2 * ydim), (stitch.lowerpixel.z + stitch.upperpixel.z) / (2 * zdim)]
    #         textStr = str(stitch.cost[0])[:2] + '_' +  str(stitch.cost[3])[:3] + '_' +  str(stitch.cost[2])[:2]
    #         canvas.view.add(visuals.Text(textStr, pos=midpoints[index], color='yellow'))

    # if b3dmidpoints:
    #     b3d_midpoint_markers = []
    #     for blob_num, blob3d in enumerate(blob3dlist):
    #         b3d_midpoint_markers.append(visuals.Markers())
    #         b3d_midpoint_markers[-1].set_data(np.array([[blob3d.avgx / xdim, blob3d.avgy / ydim, blob3d.avgz / zdim]]), edge_color='w', face_color=colors[blob_num % len(colors)], size=25)
    #         b3d_midpoint_markers[-1].symbol = 'star'
    #         canvas.view.add(b3d_midpoint_markers[-1])
    if b2dmidpoints:
        b2d_num = 0
        b2d_midpoint_pos = np.zeros([len(blob2ds), 3])

        for blob2d in blob2ds:
            b2d_midpoint_pos[b2d_num] = [blob2d.avgx / xdim, blob2d.avgy / ydim, blob2d.height / zdim]
            b2d_num += 1
        b2d_midpoint_markers = visuals.Markers()



        b2d_midpoint_markers.set_data(b2d_midpoint_pos, edge_color='w', face_color='yellow', size=15)
        b2d_midpoint_markers.symbol = 'diamond'
        # canvas.view.add(b2d_midpoint_markers)
        canvas.add_marker(b2d_midpoint_markers, 'blob2d_mid')
    #
    # if b2d_midpoint_values > 0:
    #
    #     max_midpoints = b2d_midpoint_values
    #     print('The midpoints texts are the number of edge_pixels in the Blob2d, showing a total of ' + str(max_midpoints))
    #     b2d_count = sum(len(b3d.blob2ds) for b3d in blob3dlist)
    #     b2d_midpoint_textmarkers = []
    #     b2d_midpoint_pos = np.zeros([b2d_count, 3])
    #
    #     blob2dlist = list(b2d for b3d in blob3dlist for b2d in b3d.blob2ds)
    #     blob2dlist = sorted(blob2dlist, key=lambda blob2d: len(blob2d.edge_pixels), reverse=False)
    #     for b2d_num, b2d in enumerate(blob2dlist[0::3][:max_midpoints]): # GETTING EVERY Nth RELEVANT INDEX
    #         b2d_midpoint_pos[b2d_num] = [b2d.avgx / xdim, b2d.avgy / ydim, b2d.height / zdim]
    #         b2d_midpoint_textmarkers.append(visuals.Text(str(len(b2d.edge_pixels)), pos=b2d_midpoint_pos[b2d_num], color='yellow'))
    #         canvas.view.add(b2d_midpoint_textmarkers[-1])



    if coloring == 'blob2d' or canvas.buffering:
        pixel_arrays = []
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
            if canvas.buffering:
                # canvas.blob2d_coloring_markers.append(buf)
                # canvas.marker_colors.append('blob2d')
                canvas.add_marker(buf, 'blob2d')
            # if coloring == 'blob2d':
            #     canvas.view.add(buf)

    if coloring == 'blob3d' or canvas.buffering:
        edge_pixel_arrays = [] # One array per 3d blob
        max_b3d_id = max(b2d.b3did for b2d in blob2ds)
        b2d_id_size = len(set([b2d.b3did for b2d in blob2ds]))


        b3d_lists = [[] for i in range(max_b3d_id + 2)] # +2 to allow for blobs which are unassigned
                                                        # NOTE this may cause seemingly strange results later if b3did unset!!!
                                                        # TODO find a more elegant way to display these.. maybe an edge color?
        # DEBUG
        # print('Processing ' + str(len(blob2ds)))

        for b2d in blob2ds:
            b3d_lists[b2d.b3did].append(b2d)
        if len(b3d_lists[-1]):
            warn('Plotting b2ds that weren\'t assigned ids (below)')
            # for b2d in b3d_lists[-1]:
            #     print('  ' + str(b2d))
        b3d_lists = [b3d_list for b3d_list in b3d_lists if len(b3d_list)]
        print('--- DB num b3ds: ' + str(len(b3d_lists)) + ' unique b3dids in b2ds: ' + str(b2d_id_size))



        canvas.b3d_count = len(b3d_lists)
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

            if canvas.buffering:
                # canvas.blob3d_coloring_markers.append(buf)
                # canvas.marker_colors.append('blob3d')
                canvas.add_marker(buf, 'blob3d')
            # if coloring == 'blob3d':
            #     canvas.view.add(buf) # HACK

    if coloring == 'b2d_depth' or canvas.buffering:
        pixel_arrays = []

        max_depth = max(blob2d.recursive_depth for blob2d in blob2ds if hasattr(blob2d, 'recursive_depth'))
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
                if canvas.buffering:
                    # canvas.marker_colors.append('depth')
                    # canvas.depth_coloring_markers.append(buf)
                    canvas.add_marker(buf, 'depth')
                # if coloring == 'depth':
                #     canvas.view.add(buf)

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
            canvas.view.add(visuals.Text(textStr, pos=midpoints[-1], color=color, font_size=15, bold=True))
    if stitches:
        #loading neutral stitches (all the same color)
        lineendpoints = 0
        for blob2d in blob2ds:
            for pairing in blob2d.pairings:
                lineendpoints += (2 * len(pairing.indeces))
        if lineendpoints != 0:
            line_index = 0
            line_locations = np.zeros([lineendpoints, 3])
            for blob2d in blob2ds:
                for pairing in blob2d.pairings:
                    for lowerpnum, upperpnum in pairing.indeces:
                        lowerpixel = Pixel.get(pairing.lowerpixels[lowerpnum])
                        upperpixel = Pixel.get(pairing.upperpixels[upperpnum])
                        line_locations[line_index] = [(lowerpixel.x - xmin) / xdim, (lowerpixel.y - ymin) / ydim, (pairing.lowerheight) / ( z_compression * zdim)]
                        line_locations[line_index + 1] = [(upperpixel.x - xmin) / xdim, (upperpixel.y - ymin) / ydim, (pairing.upperheight) / ( z_compression * zdim)]
                        line_index += 2
            stitch_lines = visuals.Line(method=linemethod)
            stitch_lines.set_data(pos=line_locations, connect='segments')
            canvas.stitches.append((stitch_lines, 'neutral'))
            # canvas.view.add(stitch_lines)
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
            canvas.stitches.append((parent_lines , 'parent'))
    canvas.setup_markers()
    canvas.setup_stitches()
    vispy.app.run()


def plotBlob3ds(blob3dlist, stitches=True, color=None, lineColoring=None, costs=0, maxcolors=-1, b2dmidpoints=False, b3dmidpoints=False, canvas_size=(800, 800), b2d_midpoint_values=0, titleNote=''):
    global colors
    canvas = setupCanvas(canvas_size,
                                 title='plotBlob3ds(' + str(len(blob3dlist)) + '-Blob3ds, coloring=' + str(color) + ', canvas_size=' + str(canvas_size) + ') ' + titleNote)
    if maxcolors > 0 and maxcolors < len(colors):
        colors = colors[:maxcolors]

    # Finding the maximal slide, so that the vertical dimension of the plot can be evenly divided
    zdim = 0
    xdim = 0
    ydim = 0

    for blob3d in blob3dlist: # TODO make gen functions
        if blob3d.highslideheight > zdim:
            zdim = blob3d.highslideheight
        if blob3d.maxx > xdim:
            xdim = blob3d.maxx
        if blob3d.maxy > ydim:
            ydim = blob3d.maxy

    zdim += 1 # Note this is b/c numbering starts at 0
    edge_pixel_arrays = [] # One array per 3d blob
    markerlist = []
    lineendpoints = 0

    if color == 'blob' or color == 'blob3d': # Note: This is very graphics intensive.
        markers_per_color = [0 for i in range(min(len(colors), len(blob3dlist)))]
        offsets = [0] * min(len(colors), len(blob3dlist))
        for blobnum, blob3d in enumerate(blob3dlist):
            markers_per_color[blobnum % len(markers_per_color)] += blob3d.get_edge_pixel_count()
        for num,i in enumerate(markers_per_color):
            edge_pixel_arrays.append(np.zeros([i, 3]))
        for blobnum, blob3d in enumerate(blob3dlist):
            index = blobnum % len(markers_per_color)
            for p_num, pixel in enumerate(blob3d.get_edge_pixels()):
                edge_pixel_arrays[index][p_num + offsets[index]] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( z_compression * zdim)]
            offsets[index] += blob3d.get_edge_pixel_count()
        for color_num, edge_array in enumerate(edge_pixel_arrays):
            buf = visuals.Markers()
            buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
            canvas.view.add(buf)

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
            if blob3d.isSingular:
                for pixel in blob3d.edge_pixels:
                    singular_edge_array[singular_index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (z_compression * zdim)]
                    singular_index += 1
            else:
                for pixel in blob3d.edge_pixels:
                    multi_edge_array[multi_index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (z_compression * zdim)]
                    multi_index += 1
        singular_markers = visuals.Markers()
        multi_markers = visuals.Markers()
        singular_markers.set_data(singular_edge_array, edge_color=None, face_color='green', size=8)
        multi_markers.set_data(multi_edge_array, edge_color=None, face_color='red', size=8)
        canvas.view.add(singular_markers)
        canvas.view.add(multi_markers)

    elif color == 'depth': # Coloring based on recursive depth
        max_depth = max(blob.recursive_depth for blob in blob3dlist)
        # NOTE because of sorting, this needs to be done before any info (like midpoints) is extracted from blob3dslist
        blob3dlist = sorted(blob3dlist, key=lambda blob: blob.recursive_depth, reverse=False) # Now sorted by depth, lowest first (primary)
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
                    edge_pixel_arrays[-1][p_num] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( z_compression * zdim)]
                    p_num += 1
            markerlist.append(visuals.Markers())
            markerlist[-1].set_data(edge_pixel_arrays[-1], edge_color=None, face_color=colors[depth % len(colors)], size=8)
            canvas.view.add(markerlist[-1])
    else: # All colored the same
        total_points = 0
        for blob_num, blob3d in enumerate(blob3dlist):
            total_points += blob3d.get_edge_pixel_count()
        edge_pixel_array = np.zeros([total_points, 3])
        index = 0
        for blob3d in blob3dlist:
            ep_buf = blob3d.get_edge_pixels()
            for pixel in ep_buf:
                edge_pixel_array[index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (z_compression * zdim)]
                index += 1
            for stitch in blob3d.pairings:
                lineendpoints += (2 * len(stitch.indeces)) # 2 as each line has 2 endpoints
        markers = visuals.Markers()
        markers.set_data(edge_pixel_array, edge_color=None, face_color=colors[0], size=8) # TODO change color
        canvas.view.add(markers)

    if costs > 0:
        number_of_costs_to_show = costs # HACK
        all_stitches = list(stitches for blob3d in blob3dlist for pairing in blob3d.pairings for stitches in pairing.stitches)
        all_stitches = sorted(all_stitches, key=lambda stitch: stitch.cost[2], reverse=True) # costs are (contour_cost, distance(as cost), total, distance(not as cost))
        midpoints = np.zeros([number_of_costs_to_show,3])
        for index,stitch in enumerate(all_stitches[:number_of_costs_to_show]): #FIXME! For some reason overloads the ram.
            midpoints[index] = [(stitch.lowerpixel.x + stitch.upperpixel.x) / (2 * xdim), (stitch.lowerpixel.y + stitch.upperpixel.y) / (2 * ydim), (stitch.lowerpixel.z + stitch.upperpixel.z) / (2 * zdim)]
            textStr = str(stitch.cost[0])[:2] + '_' +  str(stitch.cost[3])[:3] + '_' +  str(stitch.cost[2])[:2]
            canvas.view.add(visuals.Text(textStr, pos=midpoints[index], color='yellow'))

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
                        line_location_lists[-1][line_index] = [lowerpixel.x / xdim, lowerpixel.y / ydim, (pairing.lowerslidenum ) / ( z_compression * zdim)]
                        line_location_lists[-1][line_index + 1] = [upperpixel.x / xdim, upperpixel.y / ydim, (pairing.upperslidenum ) / ( z_compression * zdim)]
                        line_index += 2
                stitch_lines.append(visuals.Line(method=linemethod))
                stitch_lines[-1].set_data(pos=line_location_lists[-1], connect='segments', color=colors[blob_num % len(colors)])
                canvas.view.add(stitch_lines[-1])
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
                        line_locations[line_index] = [lowerpixel.x / xdim, lowerpixel.y / ydim, (pairing.lowerheight ) / ( z_compression * zdim)]
                        line_locations[line_index + 1] = [upperpixel.x / xdim, upperpixel.y / ydim, (pairing.upperheight ) / ( z_compression * zdim)]
                        line_index += 2
            stitch_lines = visuals.Line(method=linemethod)
            stitch_lines.set_data(pos=line_locations, connect='segments')
            canvas.view.add(stitch_lines)


    if b3dmidpoints:
        b3d_midpoint_markers = []
        for blob_num, blob3d in enumerate(blob3dlist):
            b3d_midpoint_markers.append(visuals.Markers())
            b3d_midpoint_markers[-1].set_data(np.array([[blob3d.avgx / xdim, blob3d.avgy / ydim, blob3d.avgz / zdim]]), edge_color='w', face_color=colors[blob_num % len(colors)], size=25)
            b3d_midpoint_markers[-1].symbol = 'star'
            canvas.view.add(b3d_midpoint_markers[-1])
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
        canvas.view.add(b2d_midpoint_markers)

    if b2d_midpoint_values > 0:

        max_midpoints = b2d_midpoint_values
        print('The midpoints texts are the number of edge_pixels in the Blob2d, showing a total of ' + str(max_midpoints))
        b2d_count = sum(len(b3d.blob2ds) for b3d in blob3dlist)
        b2d_midpoint_textmarkers = []
        b2d_midpoint_pos = np.zeros([b2d_count, 3])

        blob2dlist = list(Blob2d.get(b2d) for b3d in blob3dlist for b2d in b3d.blob2ds)
        blob2dlist = sorted(blob2dlist, key=lambda blob2d: blob2d.id, reverse=False)
        for b2d_num, b2d in enumerate(blob2dlist[0::3][:max_midpoints]): # GETTING EVERY Nth RELEVANT INDEX
            b2d_midpoint_pos[b2d_num] = [b2d.avgx / xdim, b2d.avgy / ydim, b2d.height / zdim]
            b2d_midpoint_textmarkers.append(visuals.Text(str(b2d.id), pos=b2d_midpoint_pos[b2d_num], color='yellow'))
            canvas.view.add(b2d_midpoint_textmarkers[-1])
    vispy.app.run()


def filterAvailableColors():
    global colors
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
        removecolors = ['aliceblue', 'azure', 'blanchedalmond', 'aquamarine', 'beige', 'bisque', 'black', 'blueviolet', 
                        'brown', 'burlywood', 'cadetblue', 'chocolate', 'coral', 'cornsilk', 'cornflowerblue',
                        'chartreuse', 'crimson', 'cyan', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick',
                        'forestgreen', 'fuchsia', 'gainsboro', 'gold',  'goldenrod', 'gray', 'greenyellow', 'honeydew',
                        'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush',
                        'lawngreen', 'lemonchiffon', 'linen', 'olive', 'olivedrab', 'limegreen', 'midnightblue',
                        'mintcream', 'mistyrose', 'moccasin', 'navy', 'orangered', 'orchid', 'papayawhip', 'peachpuff',
                        'peru', 'pink', 'powderblue', 'plum', 'rosybrown', 'saddlebrown', 'salmon', 'sandybrown',
                        'seagreen', 'seashell', 'silver', 'sienna', 'skyblue', 'springgreen', 'tan', 'teal', 'thistle',
                        'tomato', 'turquoise', 'snow', 'steelblue', 'violet', 'wheat', 'yellowgreen']
        for color in removecolors:
            colors.remove(color)
        print('There are a total of ' + str(len(colors)) + ' colors available for plotting')
        # openglconfig = vispy.gloo.wrappers.get_gl_configuration() # Causes opengl/vispy crash for unknown reasons

def setupCanvas(canvas_size=(800,800), title=''):
    return Canvas(canvas_size, title=title) # Todo this is getting overwritten, but might be nice to have a fallback set?

def showColors(canvas_size=(800,800)):
    global colors
    canvas = setupCanvas(canvas_size)
    print(colors)
    print('There are a total of ' + str(len(colors)) + ' colors used for plotting')
    for i,color in enumerate(colors):
        canvas.view.add(visuals.Text(color, pos=np.reshape([0, 0, 1-(i / len(colors))], (1,3)), color=color, bold=True))
    vispy.app.run()

def plotPixels(pixellist, canvas_size=(800, 800)):
    canvas = setupCanvas(canvas_size)
    xmin = min(pixel.x for pixel in pixellist)
    ymin = min(pixel.y for pixel in pixellist)
    xmax = max(pixel.x for pixel in pixellist)
    ymax = max(pixel.y for pixel in pixellist)
    edge_pixel_array = np.zeros([len(pixellist), 3])
    for (p_num, pixel) in enumerate(pixellist):
        edge_pixel_array[p_num] = [(pixel.x - xmin) / len(pixellist), (pixel.y - ymin) / len(pixellist), pixel.z /  (z_compression * len(pixellist))]
    marker = visuals.Markers()
    marker.set_data(edge_pixel_array, edge_color=None, face_color=colors[0 % len(colors)], size=8)
    canvas.view.add(marker)
    vispy.app.run()

def plotPixelLists(pixellists, canvas_size=(800, 800)): # NOTE works well to show bloom results
    canvas = setupCanvas(canvas_size)
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
        canvas.view.add(markers)
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