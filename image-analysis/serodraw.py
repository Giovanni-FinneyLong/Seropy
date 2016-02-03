__author__ = 'gio'
# This file includes the various functions written to visualize data from within sero.py
# These functions have been separated for convenience; they are higher volume and lower maintenance.

from myconfig import config
import numpy as np
from Blob2d import Blob2d
from Pixel import Pixel
import vispy.io
import vispy.scene
from vispy.scene import visuals
from vispy.util import keys
from util import warn, debug
colors = None
from Blob3d import getBlob2dOwners, Blob3d


# TODO sample animation code here: https://github.com/vispy/vispy/blob/master/examples/basics/scene/save_animation.py
# TODO sample event code: https://github.com/vispy/vispy/blob/master/examples/tutorial/app/app_events.py

class Canvas(vispy.scene.SceneCanvas):
    def __init__(self, canvas_size=(800,800), title='', coloring='blob2d', buffering=True): # Note may want to make buffering default to False
        vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True, size=canvas_size, title=title)
        if hasattr(self,'unfreeze') and callable(getattr(self,'unfreeze')):         #HACK # Interesting bug fix for an issue that only occurs on Envy
            self.unfreeze()
        self.view = self.central_widget.add_view()

        self.fov = 10

        turn_camera = vispy.scene.cameras.TurntableCamera(fov=self.fov, azimuth=80, parent=self.view.scene, distance=1, elevation=-55, name='Turntable')
        fly_camera = vispy.scene.cameras.FlyCamera(parent=self.view.scene, fov=self.fov, name='Fly')
        panzoom_camera = vispy.scene.cameras.PanZoomCamera(parent=self.view.scene, name='Panzoom')
        arcball_camera = vispy.scene.cameras.ArcballCamera(parent=self.view.scene, fov=self.fov, distance=1, name='Arcball')

        # TODO adjust _keymap of FlyCamera to tune turning to be less extreme
        print('Fly_camera keymap: ' + str(fly_camera._keymap))

        turn_speed = .6
        assert(turn_speed >= .6) # This is because below this value, the camera stops reponding to turn requests when not already moving

        # fly_camera.set_range(x=(0,1),y=(0,1),z=(0,1))
        # fly_camera.link(turn_camera) # Can't link b/c dont share rotation :/

        # Mapping that defines keys to thrusters
        fly_camera._keymap = {
            # keys.UP: (+1, 1), keys.DOWN: (-1, 1),
            # keys.RIGHT: (+1, 2), keys.LEFT: (-1, 2),
            #
            'W': (+1, 1), 'S': (-1, 1),
            'D': (+1, 2), 'A': (-1, 2),
            'F': (+1, 3), 'C': (-1, 3),
            #
            'I': (+turn_speed, 4), 'K': (-turn_speed, 4),
            'L': (+turn_speed, 5), 'J': (-turn_speed, 5),
            'Q': (+turn_speed, 6), 'E': (-turn_speed, 6),
            #
            keys.SPACE: (0, 1, 2, 3),  # 0 means brake, apply to translation
            keys.ALT: (+5, 1),  # Turbo
        }

        self.cameras = [fly_camera, turn_camera, panzoom_camera, arcball_camera]
        self.current_camera_index = 0
        self.axis = visuals.XYZAxis(parent=self.view.scene)

        self.view.camera = self.cameras[self.current_camera_index]
        self.blob2d_coloring_markers = []
        self.blob3d_coloring_markers = []
        self.depth_coloring_markers = []
        self.show()
        self.coloring = coloring.lower()
        self.markers = []
        self.available_marker_colors = ['depth', 'blob2d', 'blob3d', 'bead']
        self.available_stitch_colors = ['neutral', 'parentID', 'none', 'blob3d']
        self.current_blob_color = self.coloring
        self.buffering = buffering
        self.marker_colors = [] # Each entry corresponds to the color of the correspond 'th marker in self.view.scene.children (not all markers!)
        self.image_no = 0
        self.b2d_count = -1
        self.b3d_count = -1
        self.b3ds = []
        self.stitches = []
        self.current_stitch_color = 'neutral'


    def on_mouse_press(self, event):
        """Pan the view based on the change in mouse position."""
        if event.button == 1:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            print(x1,y1)
            # print ('  ' + str(self.view.scene.node_transform(self.canvas_cs).simplified()))
            # print ('  ' + str(self.view.scene.node_transform(self.canvas_cs).map((x1,y1))))

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

        # modifiers = [key.name for key in event.modifiers]
        # if event.key != 'Escape':
        #     print('Key pressed - text: %r, key: %s, modifiers: %r' % (
        #     event.text, event.key.name, modifiers))

        if event.key.name == 'Up': # Next color cheme
            self.update_markers(increment=1)

        elif event.key.name == 'Down': # Previous color scheme
            self.update_markers(increment=-1)

        elif event.key.name == 'Insert': # Save an image
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

        # These numerical keys are for adjusting bead values
        elif event.key.name == '1':
            print('Changing config.max_pixels_to_be_a_bead from ' + str(config.max_pixels_to_be_a_bead), end='')
            # from myconfig import set_config.max_pixels_to_be_a_bead
            config.max_pixels_to_be_a_bead = int(1.1 * config.max_pixels_to_be_a_bead) + 10
            print(' to ' + str(config.max_pixels_to_be_a_bead))# + ' which by get = ' + str(config.get_config.max_pixels_to_be_a_bead()))
            self.refresh_bead_markers()

        elif event.key.name == '2':
            print('Changing config.max_pixels_to_be_a_bead from ' + str(config.max_pixels_to_be_a_bead), end='')
            config.max_pixels_to_be_a_bead = int(config.max_pixels_to_be_a_bead / 1.1)
            print(' to ' + str(config.max_pixels_to_be_a_bead))
            self.refresh_bead_markers()

        elif event.key.name == '3':
            print('Changing config.max_subbeads_to_be_a_bead from ' + str(config.max_subbeads_to_be_a_bead), end='')
            config.max_subbeads_to_be_a_bead += 1
            print(' to ' + str(config.max_subbeads_to_be_a_bead))
            self.refresh_bead_markers()

        elif event.key.name == '4':
            print('Changing config.max_subbeads_to_be_a_bead from ' + str(config.max_subbeads_to_be_a_bead), end='')
            config.max_subbeads_to_be_a_bead -= 1
            print(' to ' + str(config.max_subbeads_to_be_a_bead))
            self.refresh_bead_markers()

        elif event.key.name == '5':
            print('Changing config.child_bead_difference from ' + str(config.child_bead_difference), end='')
            config.child_bead_difference += 1
            print(' to ' + str(config.child_bead_difference))
            self.refresh_bead_markers()

        elif event.key.name == '6':
            print('Changing config.child_bead_difference from ' + str(config.child_bead_difference), end='')
            config.child_bead_difference -= 1
            print(' to ' + str(config.child_bead_difference))
            self.refresh_bead_markers()

        elif event.key.name == 'P':
            print('Printing all b3ds on canvas:')
            for b3d in self.b3ds:
                print(b3d)
                for b2d in b3d.blob2ds:
                    print('  ' + str(Blob2d.get(b2d)))

        elif event.key.name == 'V': # Change cameras
            self.current_camera_index = (self.current_camera_index + 1) % len(self.cameras)
            self.view.camera = self.cameras[self.current_camera_index]
            self.update_title()

    def refresh_bead_markers(self):
        # from myconfig import config.max_subbeads_to_be_a_bead
        Blob3d.tag_all_beads()
        self.remove_markers_of_color('bead')
        self.add_bead_markers(self.b3ds)
        self.update()

    def remove_markers_of_color(self, color):
        remove_markers = []
        for marker, coloring in self.markers:
            if coloring == color:
                # self.view.remove(marker)
                remove_markers.append((marker,coloring))
        for marker,coloring in remove_markers:
            marker.remove_parent(marker.parent)
            self.markers.remove((marker, coloring))
            del marker

    def next_marker_color(self, increment=1):
        assert(increment in [1,-1])
        return self.available_marker_colors[(self.available_marker_colors.index(self.current_blob_color) + increment) % len(self.available_marker_colors)]

    def next_stitch_color(self, increment=1):
        assert(increment in [1,-1])
        if self.current_stitch_color in self.available_stitch_colors:
            return self.available_stitch_colors[(self.available_stitch_colors.index(self.current_stitch_color) + increment) % len(self.available_stitch_colors)]
        else:
            return self.available_stitch_colors[0]

    def set_stitch_color(self, newColor):
        self.current_stitch_color = newColor
        self.update_stitches(increment=0)

    def update_stitches(self, increment=1):
        assert increment in [-1,0,1] # 0 is a refresh
        if len(self.available_stitch_colors):
            if increment != 0:
                self.current_stitch_color = self.next_stitch_color(increment=increment)
            for stitch, color in self.stitches:
                if color == self.current_stitch_color:# and \
                        # not (self.current_stitch_color == 'blob3d' and self.current_blob_color == 'blob3d' and color == 'parentID'): #Hides parentID lines when plotting blob3d b/c exploding is turned off
                    stitch.visible = True
                else:
                    stitch.visible = False
        self.update_title()

    def update_markers(self, increment=1):
        assert increment in [-1, 0, 1] # 0 is a refresh
        if len(self.available_marker_colors):
            if increment != 0:
                self.current_blob_color = self.next_marker_color(increment=increment)
            for child,coloring in self.markers:
                if coloring == self.current_blob_color:
                    child.visible = True
                else:
                    child.visible = False
            if self.current_blob_color == 'blob3d': #for now, these are viewed flat, and so parentlines should be turned off
                for stitch, color in self.stitches:
                    if color == 'parentID':
                        stitch.visible = False
        self.update_title()

    def name_image(self):
        prefix = config.FIGURES_DIR
        if config.test_instead_of_data:
            prefix += 'Test_'
        else:
            prefix += 'Data_'
        self.image_no += 1
        return prefix + str(self.image_no - 1) + '.png'

    def setup_markers(self):
        counts = [0] * len(self.available_marker_colors)
        for marker, coloring in self.markers:
            counts[self.available_marker_colors.index(coloring)] += 1
        self.available_marker_colors = [color for (index, color) in enumerate(self.available_marker_colors) if counts[index] > 0]
        self.available_marker_colors.append(None)
        for marker, coloring in self.markers:
            if coloring == self.current_blob_color:
                marker.visible = True
            else:
                marker.visible = False
        self.update_title()

    def setup_stitches(self):
        # Going to count whether there are any stitches of each type,
        # therefore if there aren't that type can be skipped
        counts = [0] * len(self.available_stitch_colors)
        for stitch, coloring in self.stitches:
            counts[self.available_stitch_colors.index(coloring)] += 1
        self.available_stitch_colors = [color for (index, color) in enumerate(self.available_stitch_colors) if counts[index] > 0]
        self.available_stitch_colors.append(None)
        for stitch, coloring in self.stitches:
            if coloring == self.current_stitch_color:
                stitch.visible = True
            else:
                stitch.visible = False
        self.update_title()

    def add_stitch(self, stitch, coloring):
        self.stitches.append((stitch, coloring))
        self.view.add(self.stitches[-1][0])

    def update_title(self):
        self.title =  '# B3ds: ' + str(self.b3d_count) + ', # B2ds: ' + str(self.b2d_count) + ', Coloring = ' + str(self.current_blob_color) + ', Stitches = ' + str(self.current_stitch_color) + ', Camera = ' + self.extract_camera_name()

    def extract_camera_name(self):
        # buf = str(type(self.cameras[self.current_camera_index]))
        # return buf[buf.rindex("."):buf.rindex("'")]
        return self.view.camera.name

    def add_marker(self, marker, coloring):
        self.markers.append((marker,coloring))
        self.view.add(self.markers[-1][0]) # add the above marker

    def add_bead_markers(self, blob3dlist):
        total_bead_points = 0
        total_nonbead_points = 0 # Points from blob3ds that may be part of strands
        for blob3d in blob3dlist:
            if blob3d.isBead:
                total_bead_points += blob3d.get_edge_pixel_count()
            else:
                total_nonbead_points += blob3d.get_edge_pixel_count()

        # print('Total bead points: ' + str(total_bead_points))
        # print('Total nonbead points: ' + str(total_nonbead_points))
        bead_edge_array = np.zeros([total_bead_points, 3])
        nonbead_edge_array = np.zeros([total_nonbead_points, 3])
        bead_index = 0
        nonbead_index = 0
        for blob_num, blob3d in enumerate(blob3dlist):
            if blob3d.isBead:
                for pixel in blob3d.get_edge_pixels():
                    bead_edge_array[bead_index] = [pixel.x / self.xdim, pixel.y / self.ydim, pixel.z / (config.z_compression * self.zdim)]
                    bead_index += 1
            else:
                for pixel in blob3d.get_edge_pixels():
                    nonbead_edge_array[nonbead_index] = [pixel.x / self.xdim, pixel.y / self.ydim, pixel.z / (config.z_compression * self.zdim)]
                    nonbead_index += 1
        bead_markers = visuals.Markers()
        nonbead_markers = visuals.Markers()
        bead_markers.set_data(bead_edge_array, edge_color=None, face_color='red', size=8)
        nonbead_markers.set_data(nonbead_edge_array, edge_color=None, face_color='green', size=8)
        self.add_marker(bead_markers, 'bead')
        self.add_marker(nonbead_markers, 'bead')


def plotBlob2ds(blob2ds, coloring='', canvas_size=(1080,1080), ids=False, stitches=False, titleNote='', edge=True, parentlines=False, explode=False, showStitchCosts=0, b2dmidpoints=False):
    global colors
    coloring = coloring.lower()
    assert coloring in ['blob2d', '', 'b2d_depth', 'blob3d', 'bead']

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

    canvas = Canvas(canvas_size, coloring='blob2d', title='plotBlob2ds(' + str(len(blob2ds)) + '-Blob2ds, coloring=' + str(coloring) + ' canvas_size=' + str(canvas_size) + ') ' + titleNote)
    canvas.b2d_count = len(blob2ds)
    canvas.xdim = xdim
    canvas.ydim = ydim
    canvas.zdim = zdim

    # canvas.available_marker_colors = ['depth', 'blob2d', 'blob3d', 'bead']
    # canvas.available_stitch_colors = ['neutral', 'parentID', 'none']#, 'blob3d']

    blob3dlist = getBlob2dOwners(blob2ds, ids=False)
    print('setting canvas.b3ds to a b3dlist of len: ' + str(len(canvas.b3ds)))
    canvas.b3ds = blob3dlist

    if canvas.buffering or coloring == 'bead':
        canvas.add_bead_markers(canvas.b3ds)

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
                    pixel_arrays[index][p_num + offsets[index]] = [(pixel.x - xmin) / xdim, (pixel.y - ymin) / ydim, (getBloomedHeight(blob2d, explode, zdim) - zmin) / ( config.z_compression * zdim)]
                offsets[index] += len(blob2d.edge_pixels)
            else:
                for p_num, pixel in enumerate(blob2d.pixels):
                    pixel = Pixel.get(pixel)
                    pixel_arrays[index][p_num + offsets[index]] = [(pixel.x - xmin) / xdim, (pixel.y - ymin) / ydim, (getBloomedHeight(blob2d, explode, zdim)  - zmin) / ( config.z_compression * zdim)]
                offsets[index] += len(blob2d.pixels)

        for color_num, edge_array in enumerate(pixel_arrays):
            buf = visuals.Markers()
            buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
            if canvas.buffering:
                canvas.add_marker(buf, 'blob2d')


    if coloring == 'blob3d' or canvas.buffering:
        edge_pixel_arrays = [] # One array per 3d blob
        max_b3d_id = max(b2d.b3did for b2d in blob2ds)
        b2d_id_size = len(set([b2d.b3did for b2d in blob2ds]))


        b3d_lists = [[] for i in range(max_b3d_id + 2)] # +2 to allow for blobs which are unassigned
                                                        # NOTE this may cause seemingly strange results later if b3did unset!!!
                                                        # TODO find a more elegant way to display these.. maybe an edge color?
        for b2d in blob2ds:
            b3d_lists[b2d.b3did].append(b2d)
        if len(b3d_lists[-1]):
            warn('Plotting b2ds that weren\'t assigned ids (below)')
        b3d_lists = [b3d_list for b3d_list in b3d_lists if len(b3d_list)]
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
                edge_pixel_arrays[index][p_num + offsets[index]] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( config.z_compression * zdim)]
            offsets[index] += sum([len(b2d.edge_pixels) for b2d in b3d_list])
        for color_num, edge_array in enumerate(edge_pixel_arrays):
            buf = visuals.Markers()
            buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
            canvas.add_marker(buf, 'blob3d')

        if coloring == 'bead' or canvas.buffering:
            total_bead_points = 0
            total_nonbead_points = 0 # Points from blob3ds that may be part of strands

            for blob3d in blob3dlist:
                if blob3d.isBead:
                    total_bead_points += blob3d.get_edge_pixel_count()
                else:
                    total_nonbead_points += blob3d.get_edge_pixel_count()

            print('Total bead points: ' + str(total_bead_points))
            print('Total nonbead points: ' + str(total_nonbead_points))
            bead_edge_array = np.zeros([total_bead_points, 3])
            nonbead_edge_array = np.zeros([total_nonbead_points, 3])
            bead_index = 0
            nonbead_index = 0
            for blob_num, blob3d in enumerate(blob3dlist):
                if blob3d.isBead:
                    for pixel in blob3d.get_edge_pixels():
                        bead_edge_array[bead_index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (config.z_compression * zdim)]
                        bead_index += 1
                else:
                    for pixel in blob3d.get_edge_pixels():
                        nonbead_edge_array[nonbead_index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (config.z_compression * zdim)]
                        nonbead_index += 1
            bead_markers = visuals.Markers()
            nonbead_markers = visuals.Markers()
            bead_markers.set_data(bead_edge_array, edge_color=None, face_color='green', size=8)
            nonbead_markers.set_data(nonbead_edge_array, edge_color=None, face_color='red', size=8)
            canvas.add_marker(bead_markers, 'bead')
            canvas.add_marker(nonbead_markers, 'bead')


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
                pixel_arrays[index][p_num + offsets[index]] = [(pixel.x - xmin) / xdim, (pixel.y - ymin) / ydim, (getBloomedHeight(blob2d, explode, zdim)  - zmin) / ( config.z_compression * zdim)]
            offsets[index] += len(blob2d.edge_pixels)

        for color_num, edge_array in enumerate(pixel_arrays):
            if len(edge_array) == 0:
                print('Skipping plotting depth ' + str(color_num) + ' as there are no blob2ds at that depth')
            else:
                buf = visuals.Markers()
                buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
                canvas.add_marker(buf, 'depth')

    if ids is True:
        midpoints = []
        midpoints.append(np.zeros([1,3]))
        for b2d_num, b2d in enumerate(blob2ds):
            midpoints[-1] = [(b2d.avgx - xmin) / xdim, (b2d.avgy - ymin) / ydim, ((getBloomedHeight(b2d, explode, zdim)  + .25 - zmin) / (config.z_compression * zdim))]
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
                        line_locations[line_index] = [(lowerpixel.x - xmin) / xdim, (lowerpixel.y - ymin) / ydim, (pairing.lowerheight) / ( config.z_compression * zdim)]
                        line_locations[line_index + 1] = [(upperpixel.x - xmin) / xdim, (upperpixel.y - ymin) / ydim, (pairing.upperheight) / ( config.z_compression * zdim)]
                        line_index += 2
            stitch_lines = visuals.Line(method=config.linemethod)
            stitch_lines.set_data(pos=line_locations, connect='segments')

            # canvas.stitches.append((stitch_lines, 'neutral'))
            canvas.add_stitch(stitch_lines, 'neutral')

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
                    line_locations[line_index] = [(b2d.avgx - xmin) / xdim, (b2d.avgy - ymin) / ydim, (getBloomedHeight(b2d, explode, zdim)  - zmin) / ( config.z_compression * zdim)]
                    line_locations[line_index + 1] = [(child.avgx - xmin) / xdim, (child.avgy - ymin) / ydim, (getBloomedHeight(child, explode, zdim) - zmin) / ( config.z_compression * zdim)]
                    line_index += 2
            parent_lines = visuals.Line(method=config.linemethod)
            parent_lines.set_data(pos=line_locations, connect='segments', color='y')
            canvas.add_stitch(parent_lines, 'parentID')
    canvas.setup_markers()
    canvas.setup_stitches()
    vispy.app.run()


def plotBlob3ds(blob3dlist, stitches=True, color='blob3d', lineColoring=None, costs=0, maxcolors=-1, b2dmidpoints=False, b3dmidpoints=False, canvas_size=(800, 800), b2d_midpoint_values=0, titleNote=''):
    global colors
    canvas = Canvas(canvas_size, coloring='blob3d',
                       title='plotBlob3ds(' + str(len(blob3dlist)) + '-Blob3ds, coloring=' + str(color) + ', canvas_size=' + str(canvas_size) + ') ' + titleNote)
    if maxcolors > 0 and maxcolors < len(colors):
        colors = colors[:maxcolors]

    # Finding the maximal slide, so that the vertical dimension of the plot can be evenly divided
    zdim = 0
    xdim = 0
    ydim = 0
    # canvas.available_marker_colors = ['blob3d', 'bead', 'depth']
    # canvas.available_stitch_colors = ['neutral', 'none', 'blob3d']

    for blob3d in blob3dlist: # TODO make gen functions
        if blob3d.highslideheight > zdim:
            zdim = blob3d.highslideheight
        if blob3d.maxx > xdim:
            xdim = blob3d.maxx
        if blob3d.maxy > ydim:
            ydim = blob3d.maxy

    zdim += 1 # Note this is b/c numbering starts at 0

    canvas.xdim = xdim
    canvas.ydim = ydim
    canvas.zdim = zdim

    if color == 'bead' or canvas.buffering:
        canvas.add_bead_markers(blob3dlist)

    edge_pixel_arrays = [] # One array per 3d blob
    markerlist = []
    lineendpoints = 0
    canvas.b3d_count = len(blob3dlist)
    canvas.b2d_count = sum(len(b3d.blob2ds) for b3d in blob3dlist)
    canvas.b3ds = blob3dlist

    if color == 'blob' or color == 'blob3d' or canvas.buffering: # Note: This is very graphics intensive.
        markers_per_color = [0 for i in range(min(len(colors), len(blob3dlist)))]
        offsets = [0] * min(len(colors), len(blob3dlist))
        for blobnum, blob3d in enumerate(blob3dlist):
            markers_per_color[blobnum % len(markers_per_color)] += blob3d.get_edge_pixel_count()
        for num,i in enumerate(markers_per_color):
            edge_pixel_arrays.append(np.zeros([i, 3]))
        for blobnum, blob3d in enumerate(blob3dlist):
            index = blobnum % len(markers_per_color)
            for p_num, pixel in enumerate(blob3d.get_edge_pixels()):
                edge_pixel_arrays[index][p_num + offsets[index]] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( config.z_compression * zdim)]
            offsets[index] += blob3d.get_edge_pixel_count()
        for color_num, edge_array in enumerate(edge_pixel_arrays):
            marker = visuals.Markers()
            marker.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
            # canvas.view.add(buf)
            canvas.add_marker(marker, 'blob3d')

    if color == 'depth' or canvas.buffering: # Coloring based on recursive depth
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
                    edge_pixel_arrays[-1][p_num] = [pixel.x / xdim, pixel.y / ydim, pixel.z / ( config.z_compression * zdim)]
                    p_num += 1
            markers = visuals.Markers()
            markers.set_data(edge_pixel_arrays[-1], edge_color=None, face_color=colors[depth % len(colors)], size=8)
            canvas.add_marker(markers, 'depth')


    else: # All colored the same
        total_points = 0
        for blob_num, blob3d in enumerate(blob3dlist):
            total_points += blob3d.get_edge_pixel_count()
        edge_pixel_array = np.zeros([total_points, 3])
        index = 0
        for blob3d in blob3dlist:
            ep_buf = blob3d.get_edge_pixels()
            for pixel in ep_buf:
                edge_pixel_array[index] = [pixel.x / xdim, pixel.y / ydim, pixel.z / (config.z_compression * zdim)]
                index += 1
            for stitch in blob3d.pairings:
                lineendpoints += (2 * len(stitch.indeces)) # 2 as each line has 2 endpoints
        markers = visuals.Markers()
        markers.set_data(edge_pixel_array, edge_color=None, face_color=colors[0], size=8) # TODO change color
        # canvas.view.add(markers)
        canvas.add_marker(markers, 'neutral')

    if costs > 0:
        number_of_costs_to_show = costs # HACK
        all_stitches = list(stitches for blob3d in blob3dlist for pairing in blob3d.pairings for stitches in pairing.stitches)
        all_stitches = sorted(all_stitches, key=lambda stitch: stitch.cost[2], reverse=True) # costs are (contour_cost, distance(as cost), total, distance(not as cost))
        midpoints = np.zeros([number_of_costs_to_show,3])
        for index,stitch in enumerate(all_stitches[:number_of_costs_to_show]): #FIXME! For some reason overloads the ram.
            midpoints[index] = [(stitch.lowerpixel.x + stitch.upperpixel.x) / (2 * xdim), (stitch.lowerpixel.y + stitch.upperpixel.y) / (2 * ydim), (stitch.lowerpixel.z + stitch.upperpixel.z) / (2 * zdim)]
            textStr = str(stitch.cost[0])[:2] + '_' +  str(stitch.cost[3])[:3] + '_' +  str(stitch.cost[2])[:2]
            canvas.view.add(visuals.Text(textStr, pos=midpoints[index], color='yellow'))

    if stitches or canvas.buffering:
        if lineColoring == 'blob3d' or canvas.buffering: # TODO need to change this so that stitchlines of the same color are the same object
            if config.test_instead_of_data:
                line_location_lists = []
                for blob_num, blob3d in enumerate(blob3dlist):
                    lineendpoints = 2 * sum(len(pairing.indeces) for blob3d in blob3dlist for pairing in blob3d.pairings)
                    line_location_lists.append(np.zeros([lineendpoints, 3]))
                    line_index = 0
                    for pairing in blob3d.pairings:
                        for stitch in pairing.stitches:
                            lowerpixel = Pixel.get(stitch.lowerpixel)
                            upperpixel = Pixel.get(stitch.upperpixel)
                            line_location_lists[-1][line_index] = [lowerpixel.x / xdim, lowerpixel.y / ydim, (pairing.lowerheight ) / (config.z_compression * zdim)]
                            line_location_lists[-1][line_index + 1] = [upperpixel.x / xdim, upperpixel.y / ydim, (pairing.upperheight ) / (config.z_compression * zdim)]
                            line_index += 2
                    stitch_lines = (visuals.Line(method=config.linemethod))
                    stitch_lines.set_data(pos=line_location_lists[-1], connect='segments', color=colors[blob_num % len(colors)])
                    canvas.add_stitch(stitch_lines, 'blob3d')
            else:
                print(' For now, skipping adding b3d colored stitch lines, because it will overload video memory..')
        if lineColoring == 'neutral' or canvas.buffering:
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
                        line_locations[line_index] = [lowerpixel.x / xdim, lowerpixel.y / ydim, (pairing.lowerheight ) / ( config.z_compression * zdim)]
                        line_locations[line_index + 1] = [upperpixel.x / xdim, upperpixel.y / ydim, (pairing.upperheight ) / ( config.z_compression * zdim)]
                        line_index += 2
            stitch_lines = visuals.Line(method=config.linemethod)
            stitch_lines.set_data(pos=line_locations, connect='segments')
            canvas.add_stitch(stitch_lines, 'neutral')
            # canvas.view.add(stitch_lines)


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
    canvas.setup_markers()
    canvas.setup_stitches()
    vispy.app.run()


def filter_available_colors():
    global colors
    if config.mayPlot:
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

def showColors(canvas_size=(800,800)):
    global colors
    canvas = Canvas(canvas_size)
    print(colors)
    print('There are a total of ' + str(len(colors)) + ' colors used for plotting')
    for i,color in enumerate(colors):
        canvas.view.add(visuals.Text(color, pos=np.reshape([0, 0, 1-(i / len(colors))], (1,3)), color=color, bold=True))
    vispy.app.run()

def getBloomedHeight(b2d, explode, zdim):
    if explode:
        return b2d.height + b2d.recursive_depth / (zdim * max([len(b2d.getrelated()), 1]))
    else:
        return b2d.height




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