__author__ = 'gio'
# This file includes the various functions written to visualize data from within sero.py
# These functions have been separated for convenience; they are higher volume and lower maintenance.

from myconfig import Config
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
import math
from munkres import Munkres
import sys

# TODO sample animation code here: https://github.com/vispy/vispy/blob/master/examples/basics/scene/save_animation.py
# TODO sample event code: https://github.com/vispy/vispy/blob/master/examples/tutorial/app/app_events.py



def plot_plotly(bloblist, b2ds=False):
    import plotly.plotly as py
    import plotly.graph_objs as go
    x = []
    y = []
    z = []
    if not b2ds:
        for b3d in bloblist:
            x.append(b3d.avgx)
            y.append(b3d.avgy)
            z.append(b3d.avgz)
    else:
        for b2d in bloblist:
            x.append(b2d.avgx)
            y.append(b2d.avgy)
            z.append(b2d.height)



    all_trace = go.Scatter3d(x=x, y=y, z=z,mode='markers', marker=dict(
        size=6,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    ))
    data = [all_trace]
    layout = go.Layout(margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ))
    fig = go.Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename='simple-3d-scatter')



class Canvas(vispy.scene.SceneCanvas):
    def __init__(self, canvas_size=(800,800), title='', coloring='simple', buffering=True): # Note may want to make buffering default to False
        vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True, size=canvas_size, title=title)
        if hasattr(self,'unfreeze') and callable(getattr(self,'unfreeze')):         #HACK # Interesting bug fix for an issue that only occurs on Envy
            self.unfreeze()
        self.view = self.central_widget.add_view()

        self.fov = 50 # Must be 0 < fov < 180

        turn_camera = vispy.scene.cameras.TurntableCamera(fov=0, azimuth=80, parent=self.view.scene, distance=1, elevation=-55, name='Turntable')
        # Using a fov of zero b/c fov makes turn harder to use
        fly_camera = vispy.scene.cameras.FlyCamera(parent=self.view.scene, fov=self.fov, name='Fly')
        panzoom_camera = vispy.scene.cameras.PanZoomCamera(parent=self.view.scene, name='Panzoom')
        arcball_camera = vispy.scene.cameras.ArcballCamera(parent=self.view.scene, fov=self.fov, distance=1, name='Arcball')

        # TODO adjust _keymap of FlyCamera to tune turning to be less extreme
        # print('Fly_camera keymap: ' + str(fly_camera._keymap))

        turn_speed = .6
        assert(turn_speed >= .6) # This is because below this value, the camera stops reponding to turn requests when not already moving

        fly_camera.set_range(x=(0,1),y=(0,1),z=(0,1)) # Doesn't seem to improve performance..
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


        self.view.add(visuals.Text('X', pos=[1,0,0], font_size=15, bold=False, color='w'))
        self.view.add(visuals.Text('Y', pos=[0,1,0], font_size=15, bold=False, color='w'))
        self.view.add(visuals.Text('Z', pos=[0,0,1], font_size=15, bold=False, color='w'))


        self.view.camera = self.cameras[self.current_camera_index]
        self.blob2d_coloring_markers = []
        self.blob3d_coloring_markers = []
        self.depth_coloring_markers = []
        self.show()
        self.coloring = coloring.lower()

        print('----Starting canvas color: ' + str(self.coloring))
        self.markers = []
        self.available_marker_colors = ['simple', 'depth', 'blob2d', 'blob3d', 'bead', 'b2d_depth', 'neutral']
        self.available_stitch_colors = ['simple', 'neutral', 'parentID', 'blob3d']
        self.current_blob_color = self.coloring
        self.buffering = buffering
        self.marker_colors = [] # Each entry corresponds to the color of the correspond 'th marker in self.view.scene.children (not all markers!)
        self.image_no = 0
        self.b3ds = []
        self.b2ds = []
        self.stitches = []
        self.current_stitch_color = 'simple'
        self.plot_call = '' # Used to remember which function created the canvas

        # self.measure_fps()


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

        # modifiers = [key.log_name for key in event.modifiers]
        # if event.key != 'Escape':
        #     print('Key pressed - text: %r, key: %s, modifiers: %r' % (
        #     event.text, event.key.log_name, modifiers))

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
            print('Changing config.max_pixels_to_be_a_bead from ' + str(Config.max_pixels_to_be_a_bead), end='')
            # from myconfig import set_config.max_pixels_to_be_a_bead
            Config.max_pixels_to_be_a_bead = int(1.1 * Config.max_pixels_to_be_a_bead) + 10
            print(' to ' + str(Config.max_pixels_to_be_a_bead))# + ' which by get = ' + str(config.get_config.max_pixels_to_be_a_bead()))
            self.refresh_bead_markers()

        elif event.key.name == '2':
            print('Changing config.max_pixels_to_be_a_bead from ' + str(Config.max_pixels_to_be_a_bead), end='')
            Config.max_pixels_to_be_a_bead = int(Config.max_pixels_to_be_a_bead / 1.1)
            print(' to ' + str(Config.max_pixels_to_be_a_bead))
            self.refresh_bead_markers()

        elif event.key.name == '3':
            print('Changing config.max_subbeads_to_be_a_bead from ' + str(Config.max_subbeads_to_be_a_bead), end='')
            Config.max_subbeads_to_be_a_bead += 1
            print(' to ' + str(Config.max_subbeads_to_be_a_bead))
            self.refresh_bead_markers()

        elif event.key.name == '4':
            print('Changing config.max_subbeads_to_be_a_bead from ' + str(Config.max_subbeads_to_be_a_bead), end='')
            Config.max_subbeads_to_be_a_bead -= 1
            print(' to ' + str(Config.max_subbeads_to_be_a_bead))
            self.refresh_bead_markers()

        elif event.key.name == '5':
            print('Changing config.child_bead_difference from ' + str(Config.child_bead_difference), end='')
            Config.child_bead_difference += 1
            print(' to ' + str(Config.child_bead_difference))
            self.refresh_bead_markers()

        elif event.key.name == '6':
            print('Changing config.child_bead_difference from ' + str(Config.child_bead_difference), end='')
            Config.child_bead_difference -= 1
            print(' to ' + str(Config.child_bead_difference))
            self.refresh_bead_markers()

        elif event.key.name == 'P':
            print('Printing all b3ds on canvas:')
            for b3d in self.b3ds:
                print(b3d)
                for b2d in b3d.blob2ds:
                    print('  ' + str(Blob2d.get(b2d)))
            print('Printing all b2ds on canvas:')
            for b2d in self.b2ds:
                print(b2d)

        elif event.key.name == 'V': # Change cameras
            self.current_camera_index = (self.current_camera_index + 1) % len(self.cameras)
            self.view.camera = self.cameras[self.current_camera_index]
            self.update_title()

        elif event.key.name == '=': # Increase fov for all cameras
            self.fov = (self.fov + 10) % 180
            for camera in self.cameras:
                if camera.name in ['Fly','Arcball']:# and hasattr(camera, 'fov'):
                    camera.fov = self.fov
            self.update_title()

        elif event.key.name == '-': # Decrease fov for all cameras
            if self.fov - 10 < 0:
                self.fov = 0
            else:
                self.fov -= 10
            for camera in self.cameras:
                if camera.name in ['Fly', 'Arcball']:#if hasattr(camera, 'fov'):
                    camera.fov = self.fov
            self.update_title()

    def refresh_bead_markers(self):
        self.remove_markers_of_color('bead')
        self.add_bead_markers(self.b3ds) # Also re-tags
        self.update()

    def remove_markers_of_color(self, color):
        remove_markers = []
        for marker, coloring in self.markers:
            if coloring == color:
                # self.view.remove(marker)
                remove_markers.append((marker,coloring))
        for marker,coloring in remove_markers:
            try: #HACK
                marker.remove_parent(marker.parent) # FIXME
            except:
                pass
            self.markers.remove((marker, coloring))
            del marker

    def next_marker_color(self, increment=1):
        assert(increment in [1,-1])
        if self.current_blob_color in self.available_marker_colors:
            return self.available_marker_colors[(self.available_marker_colors.index(self.current_blob_color) + increment) % len(self.available_marker_colors)]
        else:
            return self.available_marker_colors[0]

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
        prefix = Config.FIGURES_DIR
        if Config.test_instead_of_data:
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
        self.title = str(self.plot_call) +  ': # B3ds: ' + str(len(self.b3ds)) + ', # B2ds: ' + str(len(self.b2ds)) + ', Coloring = ' + str(self.current_blob_color) + ', Stitches = ' + str(self.current_stitch_color) + ', Camera = ' + self.extract_camera_name() + ', fov = ' + str(self.fov)

    def extract_camera_name(self):
        # buf = str(type(self.cameras[self.current_camera_index]))
        # return buf[buf.rindex("."):buf.rindex("'")]
        return self.view.camera.name

    def add_marker(self, marker, coloring):
        self.markers.append((marker,coloring))
        self.view.add(self.markers[-1][0]) # add the above marker

    def add_bead_markers(self, blob3dlist):
        # Tagging beads for safety
        Blob3d.tag_all_beads()
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
                    bead_edge_array[bead_index] = [pixel.x / self.xdim, pixel.y / self.ydim, pixel.z / (Config.z_compression * self.zdim)]
                    bead_index += 1
            else:
                for pixel in blob3d.get_edge_pixels():
                    nonbead_edge_array[nonbead_index] = [pixel.x / self.xdim, pixel.y / self.ydim, pixel.z / (Config.z_compression * self.zdim)]
                    nonbead_index += 1
        bead_markers = visuals.Markers()
        nonbead_markers = visuals.Markers()
        if total_bead_points !=0:
            bead_markers.set_data(bead_edge_array, edge_color=None, face_color='red', size=8)
            self.add_marker(bead_markers, 'bead')
        if total_nonbead_points !=0:
            nonbead_markers.set_data(nonbead_edge_array, edge_color=None, face_color='green', size=8)
            self.add_marker(nonbead_markers, 'bead')

    def add_blob3d_markers(self, blob3dlist): # Note that for now, this only does edges
        edge_pixel_arrays = [] # One array per 3d blob
        markers_per_color = [0 for i in range(min(len(colors), len(blob3dlist)))]
        offsets = [0] * min(len(colors), len(blob3dlist))
        for blobnum, blob3d in enumerate(blob3dlist):
            markers_per_color[blobnum % len(markers_per_color)] += blob3d.get_edge_pixel_count()
        for num,i in enumerate(markers_per_color):
            edge_pixel_arrays.append(np.zeros([i, 3]))
        for blobnum, blob3d in enumerate(blob3dlist):
            index = blobnum % len(markers_per_color)
            for p_num, pixel in enumerate(blob3d.get_edge_pixels()):
                edge_pixel_arrays[index][p_num + offsets[index]] = [pixel.x / self.xdim, pixel.y / self.ydim, pixel.z / (Config.z_compression * self.zdim)]
            offsets[index] += blob3d.get_edge_pixel_count()
        for color_num, edge_array in enumerate(edge_pixel_arrays):
            marker = visuals.Markers()
            marker.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
            self.add_marker(marker, 'blob3d')

    def add_depth_markers(self, blob3dlist):
        edge_pixel_arrays = [] # One array per 3d blob
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
                    edge_pixel_arrays[-1][p_num] = [pixel.x / self.xdim, pixel.y / self.ydim, pixel.z / (Config.z_compression * self.zdim)]
                    p_num += 1
            markers = visuals.Markers()
            markers.set_data(edge_pixel_arrays[-1], edge_color=None, face_color=colors[depth % len(colors)], size=8)
            self.add_marker(markers, 'depth')

    def add_depth_markers_from_blob2ds(self, blob2dlist, offset=False, explode=True):
        pixel_arrays = []
        max_depth = max(blob2d.recursive_depth for blob2d in blob2dlist if hasattr(blob2d, 'recursive_depth'))
        markers_per_color = [0 for i in range(min(len(colors), max_depth + 1))]
        offsets = [0] * min(len(colors), max_depth + 1) # This is not related to offset below
        if offset:
            xoffset = self.xmin
            yoffset = self.ymin
            zoffset = self.zmin
        else:
            xoffset = 0
            yoffset = 0
            zoffset = 0

        for blobnum, blob2d in enumerate(blob2dlist):
            markers_per_color[blob2d.recursive_depth % len(markers_per_color)] += len(blob2d.edge_pixels)
        for num,i in enumerate(markers_per_color):
            pixel_arrays.append(np.zeros([i, 3]))
        for blobnum, blob2d in enumerate(blob2dlist):
            index = blob2d.recursive_depth % len(markers_per_color)
            for p_num, pixel in enumerate(blob2d.edge_pixels):
                pixel = Pixel.get(pixel)
                pixel_arrays[index][p_num + offsets[index]] = [(pixel.x - xoffset) / self.xdim, (pixel.y - yoffset) / self.ydim, (getBloomedHeight(blob2d, explode, self.zdim)  - zoffset) / (Config.z_compression * self.zdim)]
            offsets[index] += len(blob2d.edge_pixels)

        for color_num, edge_array in enumerate(pixel_arrays):
            if len(edge_array) == 0:
                print('Skipping plotting depth ' + str(color_num) + ' as there are no blob2ds at that depth')
            else:
                buf = visuals.Markers()
                buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
                self.add_marker(buf, 'b2d_depth')

    def add_blob2d_markers(self, blob2dlist, edge=True, offset=False, explode=True):
        pixel_arrays = []
        markers_per_color = [0 for i in range(min(len(colors), len(blob2dlist)))]
        offsets = [0] * min(len(colors), len(blob2dlist))
        if offset:
            xoffset = self.xmin
            yoffset = self.ymin
            zoffset = self.zmin
        else:
            xoffset = 0
            yoffset = 0
            zoffset = 0

        # print("When adding blob2d markers, using offset=" + str(offset))

        if edge:
            for blobnum, blob2d in enumerate(blob2dlist):
                markers_per_color[blobnum % len(markers_per_color)] += len(blob2d.edge_pixels)
        else:
            for blobnum, blob2d in enumerate(blob2dlist):
                markers_per_color[blobnum % len(markers_per_color)] += len(blob2d.pixels)
        for num,i in enumerate(markers_per_color):
            pixel_arrays.append(np.zeros([i, 3]))
        for blobnum, blob2d in enumerate(blob2dlist):
            index = blobnum % len(markers_per_color)
            if edge:
                for p_num, pixel in enumerate(blob2d.edge_pixels):
                    pixel = Pixel.get(pixel)
                    pixel_arrays[index][p_num + offsets[index]] = [(pixel.x - xoffset) / self.xdim, (pixel.y - yoffset) / self.ydim, (getBloomedHeight(blob2d, explode, self.zdim) - zoffset) / (Config.z_compression * self.zdim)]
                offsets[index] += len(blob2d.edge_pixels)
            else:
                for p_num, pixel in enumerate(blob2d.pixels):
                    pixel = Pixel.get(pixel)
                    pixel_arrays[index][p_num + offsets[index]] = [(pixel.x - xoffset) / self.xdim, (pixel.y - yoffset) / self.ydim, (getBloomedHeight(blob2d, explode, self.zdim)  - zoffset) / (Config.z_compression * self.zdim)]
                offsets[index] += len(blob2d.pixels)

        for color_num, edge_array in enumerate(pixel_arrays):
            buf = visuals.Markers()
            buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8)
            # print("Adding blob2d markers with locations: " + str(edge_array))
            self.add_marker(buf, 'blob2d')

    def add_blob3d_markers_from_blob2ds(self, blob2dlist): # This makes sure that the whole b3d isnt shown, instead just the relevant b2ds
        edge_pixel_arrays = [] # One array per 3d blob
        max_b3d_id = max(b2d.b3did for b2d in blob2dlist)
        b3d_lists = [[] for i in range(max_b3d_id + 2)] # +2 to allow for blobs which are unassigned
        for b2d in blob2dlist:
            b3d_lists[b2d.b3did].append(b2d)
        if len(b3d_lists[-1]):
            warn('Plotting b2ds that weren\'t assigned ids (below)')
        b3d_lists = [b3d_list for b3d_list in b3d_lists if len(b3d_list)]
        markers_per_color = [0 for i in range(min(len(colors), len(b3d_lists)))]
        offsets = [0] * min(len(colors), len(b3d_lists))
        for blobnum, b3d_list in enumerate(b3d_lists):
            markers_per_color[blobnum % len(markers_per_color)] += sum([len(b2d.edge_pixels) for b2d in b3d_list])
        for num,i in enumerate(markers_per_color):
            edge_pixel_arrays.append(np.zeros([i, 3]))
        for blobnum, b3d_list in enumerate(b3d_lists):
            index = blobnum % len(markers_per_color)
            for p_num, pixel in enumerate(Pixel.get(pixel) for b2d in b3d_list for pixel in b2d.edge_pixels):
                edge_pixel_arrays[index][p_num + offsets[index]] = [pixel.x / self.xdim, pixel.y / self.ydim, pixel.z / (Config.z_compression * self.zdim)]
            offsets[index] += sum([len(b2d.edge_pixels) for b2d in b3d_list])
        for color_num, edge_array in enumerate(edge_pixel_arrays):
            buf = visuals.Markers()
            buf.set_data(pos=edge_array, edge_color=None, face_color=colors[color_num % len(colors)], size=8 )
            self.add_marker(buf, 'blob3d')

    def add_color_markers(self, blob3dlist, colorindex=0, markertype='neutral'):
        total_points = 0
        line_endpoints = 0
        for blob_num, blob3d in enumerate(blob3dlist):
            total_points += blob3d.get_edge_pixel_count()
        edge_pixel_array = np.zeros([total_points, 3])
        index = 0
        for blob3d in blob3dlist:
            ep_buf = blob3d.get_edge_pixels()
            for pixel in ep_buf:
                edge_pixel_array[index] = [pixel.x / self.xdim, pixel.y / self.ydim, pixel.z / (Config.z_compression * self.zdim)]
                index += 1
            for stitch in blob3d.pairings:
                line_endpoints += (2 * len(stitch.indeces)) # 2 as each line has 2 endpoints
        markers = visuals.Markers()
        markers.set_data(edge_pixel_array, edge_color=None, face_color=colors[colorindex], size=8) # TODO change color
        # canvas.view.add(markers)
        self.add_marker(markers, markertype)

    def add_simple_beads(self, blob3dlist):
        non_beads = [b3d for b3d in blob3dlist if not b3d.isBead or (b3d.recursive_depth == 0 and b3d.isBead)]
        print('Number of non_beads = ' + str(len(non_beads)) + ' / ' + str(len(blob3dlist)))
        bead_groups = []
        for b3d in non_beads:
            # print('Workin on b3d: ' + str(b3d))
            if b3d.isBead:
                first_children = [b3d] # Because this is a bead at recursive depth 0
            else:
                first_children = b3d.get_first_child_beads()
                if not len(first_children):
                    print('Found non_bead b3d without any first children beads ' + str(b3d)) # FIXME TODO
            # print('First_children ' + str(len(first_children)) + ' = ' + str(first_children))
                else:
                    bead_groups.append(first_children)

        # print('Bead groups: ' + str(bead_groups))
        # print('DB ' + str(self.xdim) + ' ' + str(self.ymin) + ' ' + str(self.zdim))

        for index, bg in enumerate(bead_groups):
            # print('BG=' + str(bg))
            # print(' ' + str(len(bg)))
            marker_midpoints = np.zeros([len(bg), 3])
            if len(bg) > 1:
                line_endpoints_len = 2 * len(bg)
                line_endpoints = np.zeros([2 * len(bg), 3])
                line_index = 0
            for index, b3d in enumerate(bg):
                val = [b3d.avgx / self.xdim, b3d.avgy / self.ydim, b3d.avgz / (Config.z_compression * self.zdim)]
                marker_midpoints[index] = val
                if len(bg) > 1:
                    line_endpoints[line_index] = val
                    if index !=0 and index != len(bg)-1:
                        line_endpoints[line_index + 1] = val
                        line_index += 2
                    else:
                        line_index += 1

            markers = visuals.Markers()
            if len(marker_midpoints) == 1:
                edge_color = 'y'
            else:
                edge_color = None

            markers.set_data(marker_midpoints, face_color=colors[index % len(colors)], size=15, edge_color=edge_color)
            # print('Adding simple markers from pos: ' + str(marker_midpoints))
            if len(bg) > 1:
                lines = visuals.Line(method=Config.linemethod)
                lines.set_data(pos=line_endpoints, connect='segments')
                # print('Adding stitches from pos:' + str(line_endpoints))
                self.add_stitch(lines, 'simple')
            self.add_marker(markers, 'simple')

        #HACK
        # from scipy.sparse.csgraph import floyd_warshall
        # for bg in bead_groups:
        #     dim = len(bg)
        #     if dim > 1: # TODO 0,1 cases
        #         cost_array = np.zeros([dim, dim])
        #         print('Dim = ' + str(dim))
        #         for i in range(dim):
        #             blob1 = bg[i]
        #             for j in range(dim):
        #                 blob2 = bg[j]
        #                 cost_array[i][j] = math.sqrt(math.pow(blob1.avgx - blob2.avgx, 2)
        #                                    + math.pow(blob1.avgy - blob2.avgy, 2)
        #                                    + math.pow(blob1.avgz - blob2.avgz, 2))
        #         # Make it so that we don't stitch to ourselves!
        #         for ij in range(dim):
        #             cost_array[ij][ij] = np.inf
        #         print('CA:' + str(cost_array))
        #         dist_matrix = floyd_warshall(cost_array, directed=False, unweighted=False)
        #         print('DM:' + str(dist_matrix))
        #         print('CA2:' + str(cost_array))

    def add_blob3d_stitches(self, blob3dlist): # FIXME
        num_markers = min(len(blob3dlist), len(colors))
        lines_per_color = [0] * num_markers
        offsets = [0] * num_markers
        line_location_lists = []
        for blob_num, b3d in enumerate(blob3dlist):
            lines_per_color[blob_num % num_markers] += sum(len(pairing.indeces) for blob3d in blob3dlist for pairing in blob3d.pairings)
        for line_count in lines_per_color:
            line_location_lists.append(np.zeros([2 * line_count, 3])) # 2x because we are storing endpoints
        for blob_num, blob3d in enumerate(blob3dlist):
            for pairing in blob3d.pairings:
                for stitch_no, stitch in enumerate(pairing.stitches):
                    lowerpixel = Pixel.get(stitch.lowerpixel)
                    upperpixel = Pixel.get(stitch.upperpixel)
                    blob_color_index = blob_num % num_markers
                    line_location_lists[blob_color_index][offsets[blob_color_index] + (2 * stitch_no)] = [lowerpixel.x / self.xdim, lowerpixel.y / self.ydim, (pairing.lowerheight ) / (Config.z_compression * self.zdim)]
                    line_location_lists[blob_color_index][offsets[blob_color_index] + (2 * stitch_no) + 1] = [upperpixel.x / self.xdim, upperpixel.y / self.ydim, (pairing.upperheight ) / (Config.z_compression * self.zdim)]
                offsets[blob_color_index] += 2 * len(pairing.stitches)
        # print(' DB adding a total of ' + str(len(line_location_lists)) + ' blob3d stitch line groups')
        for list_no, line_location_list in enumerate(line_location_lists):
            stitch_lines = (visuals.Line(method=Config.linemethod))
            stitch_lines.set_data(pos=line_location_list, connect='segments', color=colors[list_no % len(colors)])
            self.add_stitch(stitch_lines, 'blob3d')

    def add_neutral_stitches(self, blob3dlist):
        line_end_points = 0
        line_index = 0
        for blob_num, blob3d in enumerate(blob3dlist):
            for stitch in blob3d.pairings:
                line_end_points += (2 * len(stitch.indeces)) # 2 as each line has 2 endpoints
        line_locations = np.zeros([line_end_points, 3])
        for blob3d in blob3dlist:
            for pairing in blob3d.pairings:
                for stitch in pairing.stitches:
                    lowerpixel = Pixel.get(stitch.lowerpixel)
                    upperpixel = Pixel.get(stitch.upperpixel)
                    line_locations[line_index] = [lowerpixel.x / self.xdim, lowerpixel.y / self.ydim, (pairing.lowerheight ) / (Config.z_compression * self.zdim)]
                    line_locations[line_index + 1] = [upperpixel.x / self.xdim, upperpixel.y / self.ydim, (pairing.upperheight ) / (Config.z_compression * self.zdim)]
                    line_index += 2
        stitch_lines = visuals.Line(method=Config.linemethod)
        stitch_lines.set_data(pos=line_locations, connect='segments')
        self.add_stitch(stitch_lines, 'neutral')

    def add_neutral_stitches_from_blob2ds(self, blob2dlist, offset=False):
        if offset:
            xoffset = self.xmin
            yoffset = self.ymin
            zoffset = self.zmin
        else:
            xoffset = 0
            yoffset = 0
            zoffset = 0

        lineendpoints = 0
        for blob2d in blob2dlist:
            for pairing in blob2d.pairings:
                lineendpoints += (2 * len(pairing.indeces))
        if lineendpoints != 0:
            line_index = 0
            line_locations = np.zeros([lineendpoints, 3])
            for blob2d in blob2dlist:
                for pairing in blob2d.pairings:
                    for lowerpnum, upperpnum in pairing.indeces:
                        lowerpixel = Pixel.get(pairing.lowerpixels[lowerpnum])
                        upperpixel = Pixel.get(pairing.upperpixels[upperpnum])
                        line_locations[line_index] = [(lowerpixel.x - xoffset) / self.xdim, (lowerpixel.y - yoffset) / self.ydim, (pairing.lowerheight - zoffset) / (Config.z_compression * self.zdim)]
                        line_locations[line_index + 1] = [(upperpixel.x - xoffset) / self.xdim, (upperpixel.y - yoffset) / self.ydim, (pairing.upperheight - zoffset) / (Config.z_compression * self.zdim)]
                        line_index += 2
            stitch_lines = visuals.Line(method=Config.linemethod)
            stitch_lines.set_data(pos=line_locations, connect='segments')
            self.add_stitch(stitch_lines, 'neutral')

    def add_parent_lines(self, blob2dlist, offset=False, explode=True):
        if offset:
            xoffset = self.xmin
            yoffset = self.ymin
            zoffset = self.zmin
        else:
            xoffset = 0
            yoffset = 0
            zoffset = 0
        lineendpoints = 0
        for num,b2d in enumerate(blob2dlist):
            lineendpoints += (2 * len(b2d.children))
        if lineendpoints:
            line_index = 0
            line_locations = np.zeros([lineendpoints, 3])
            for b2d in blob2dlist:
                for child in b2d.children:
                    child = Blob2d.get(child)
                    line_locations[line_index] = [(b2d.avgx - xoffset) / self.xdim, (b2d.avgy - yoffset) / self.ydim, (getBloomedHeight(b2d, explode, self.zdim)  - zoffset) / (Config.z_compression * self.zdim)]
                    line_locations[line_index + 1] = [(child.avgx - xoffset) / self.xdim, (child.avgy - yoffset) / self.ydim, (getBloomedHeight(child, explode, self.zdim) - zoffset) / (Config.z_compression * self.zdim)]
                    line_index += 2
            parent_lines = visuals.Line(method=Config.linemethod)
            parent_lines.set_data(pos=line_locations, connect='segments', color='y')
            self.add_stitch(parent_lines, 'parentID')

    def set_blobs(self, bloblist):
        '''
        Setups the necessary parts of the canvas according to supplied blob2ds or blob3ds (exclusive)
        Inculding self.b2ds, self.b3ds (regardless of whether given b2ds or b3ds will set both)
        Then goes on to set the x,y,z info of the canvas based on the blobs supplied
        :param bloblist: A list of entirely blob2ds or entirely blob3ds
        :return:
        '''
        print("-- Called set_blobs")
        if all(type(blob) is Blob3d for blob in bloblist):
            # Given a list of blob3ds
            self.b3ds = bloblist
            self.b2ds = [blob2d for blob3d in bloblist for blob2d in blob3d.blob2ds]
            self.set_canvas_bounds(b2ds_not_b3ds=False)
        elif all(type(blob) is Blob2d for blob in bloblist):
            # Given a list of blob2ds
            self.b2ds = bloblist
            self.b3ds = getBlob2dOwners(bloblist, ids=False)
            self.set_canvas_bounds(b2ds_not_b3ds=True)
        else:
            warn('Given a list not of blob2ds or blob3ds entirely!!!')
            print(bloblist)
            debug()
            exit(1)

    def set_canvas_bounds(self, b2ds_not_b3ds):
        '''
        Sets (x,y,z) (min, max, dim) using either canvas.b2ds or canvas.b3ds
        :return:
        '''
        if b2ds_not_b3ds:
            xmin = self.b2ds[0].minx
            ymin = self.b2ds[0].miny
            zmin = self.b2ds[0].height
            xmax = self.b2ds[0].maxx
            ymax = self.b2ds[0].maxy
            zmax = self.b2ds[0].height

            for b2d in self.b2ds:
                xmin = min(xmin, b2d.minx)
                xmax = max(xmax, b2d.maxx)
                ymin = min(ymin, b2d.miny)
                ymax = max(ymax, b2d.maxy)
                zmin = min(zmin, b2d.height)
                zmax = max(zmax, b2d.height)
        else:
            xmin = self.b3ds[0].minx
            ymin = self.b3ds[0].miny
            zmin = self.b3ds[0].lowslideheight
            xmax = self.b3ds[0].maxx
            ymax = self.b3ds[0].maxy
            zmax = self.b3ds[0].highslideheight

            for blob3d in self.b3ds:
                xmin = min(xmin, blob3d.minx)
                xmax = max(xmax, blob3d.maxx)
                ymin = min(ymin, blob3d.miny)
                ymax = max(ymax, blob3d.maxy)
                zmin = min(zmin, blob3d.lowslideheight)
                zmax = max(zmax, blob3d.highslideheight)

            # zdim += 1 # Note this is b/c numbering starts at 0
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.xdim = xmax - xmin + 1
        self.ydim = ymax - ymin + 1
        self.zdim = zmax - zmin + 1
        print("xmin: " + str(self.xmin) + ", xmax: " + str(self.xmax) + ", ymin: " + str(self.ymin) + ", ymax: " + str(self.ymax) + ", zmin: " + str(self.zmin) + ", zmax: " + str(self.zmax))
        print("xdim: " + str(self.xdim) + ', ydim: ' + str(self.ydim) + ', zdim: ' + str(self.zdim))



def plotBlob2ds(blob2ds, coloring='', canvas_size=(1080,1080), ids=False, stitches=False, titleNote='', edge=True,
                parentlines=False, explode=False, showStitchCosts=0, b2dmidpoints=False, offset=False, pixel_ids=False, images_and_heights=None):
    global colors
    coloring = coloring.lower()
    assert coloring in ['blob2d', '', 'b2d_depth', 'blob3d', 'bead']

    # This block allows the passing of ids or blob2ds
    if len(blob2ds) == 0:
        warn('Tried to plot 0 blob2ds')
    else:
        all_b2ds_are_ids = all(type(b2d) is int for b2d in blob2ds)
        all_b2d_are_blob2ds = all(type(b2d) is Blob2d for b2d in blob2ds)
        assert(all_b2d_are_blob2ds or all_b2ds_are_ids)
        if all_b2ds_are_ids: # May want to change this depending on what it does to memory
            blob2ds = [Blob2d.get(b2d) for b2d in blob2ds]


        if coloring == '':
            coloring = 'blob2d' # For the canvas title

        canvas = Canvas(canvas_size, coloring='blob2d', title='plotBlob2ds(' + str(len(blob2ds)) + '-Blob2ds, coloring=' + str(coloring) + ' canvas_size=' + str(canvas_size) + ') ' + titleNote)
        canvas.plot_call = 'PlotBlob2ds'
        canvas.set_blobs(blob2ds)


        if coloring == 'blob2d' or canvas.buffering:
            canvas.add_blob2d_markers(blob2ds, edge=edge, offset=offset, explode=explode)

        if coloring == 'blob3d' or canvas.buffering:
            canvas.add_blob3d_markers_from_blob2ds(blob2ds)

        if coloring == 'bead' or canvas.buffering:
            canvas.add_bead_markers(canvas.b3ds) # HACK

        if coloring == 'b2d_depth' or canvas.buffering:
            canvas.add_depth_markers_from_blob2ds(blob2ds)

        if stitches or canvas.buffering:
            canvas.add_neutral_stitches_from_blob2ds(blob2ds, offset=offset)

        if parentlines or canvas.buffering:
            canvas.add_parent_lines(blob2ds, offset=offset, explode=explode)

        # if pixel_ids:
        #     print("\nWARNING adding ids for every pixel, this could overload if not a small dataset, so skipping if not a test_set!!")
        #     if Config.test_instead_of_data:
        #         for blob2d in blob2ds:
        #                 for pixel in blob2d.pixels:
        #                     pixel = Pixel.get(pixel)
        #                     canvas.view.add(visuals.Text(str(pixel.id), pos=[(pixel.x) / canvas.xdim,(pixel.y) / canvas.ydim,(pixel.z) / canvas.zdim], font_size=4, bold=False, color='w'))


        # if b2dmidpoints:
        #     b2d_num = 0
        #     b2d_midpoint_pos = np.zeros([len(blob2ds), 3])
        #     for blob2d in blob2ds:
        #         b2d_midpoint_pos[b2d_num] = [blob2d.avgx / xdim, blob2d.avgy / ydim, blob2d.height / zdim]
        #         b2d_num += 1
        #     b2d_midpoint_markers = visuals.Markers()
        #     b2d_midpoint_markers.set_data(b2d_midpoint_pos, edge_color='w', face_color='yellow', size=15)
        #     b2d_midpoint_markers.symbol = 'diamond'
        #     canvas.add_marker(b2d_midpoint_markers, 'blob2d_mid')

        if ids:
            print("\nWARNING adding ids for every blob2d, this could overload if not a small dataset, so skipping if not a test_set!!")
            if Config.test_instead_of_data:

                midpoints= np.zeros([1,3])
                for b2d_num, b2d in enumerate(blob2ds):
                    if b2d.recursive_depth == 0:
                        midpoints = [(b2d.avgx - canvas.xmin) / canvas.xdim, (b2d.avgy - canvas.ymin) / canvas.ydim, ((getBloomedHeight(b2d, explode, canvas.zdim)  + .25 - canvas.zmin) / (Config.z_compression * canvas.zdim))]
                        textStr = str(b2d.id)
                        # if coloring == '' or coloring == 'blob2d':
                        color = colors[b2d.id % len(colors)]
                        # else:
                        #     if coloring in colors:
                        #         color = coloring
                        #     else:
                        #         color = 'yellow'
                        canvas.view.add(visuals.Text(textStr, pos=midpoints, color=color, font_size=8, bold=False))

        # if images_and_heights is not None:
        #     for image, height in images_and_heights:
        #         new_arr = np.zeros([image.shape[0], image.shape[1], 3])
        #         print('DIM:' + str(new_arr.shape))
        #         for i in range(image.shape[0]):
        #             for j in range(image.shape[1]):
                        # print(new_arr[i][j])
                        # print(image[i][j])

                # height_dim = np.ones([image.shape[0],image.shape[1]])
                # print('Height dim:' + str(height_dim.shape))
                # print('Image dim:' + str(image.shape))
                # print('Combined shape:' + str(np.dstack((new_arr, height_dim)).shape))
                # displayed_image = visuals.Image(image, )


        canvas.setup_markers()
        canvas.setup_stitches()
        vispy.app.run()


def plotBlob3ds(blob3dlist, stitches=True, color='blob3d', lineColoring=None, costs=0, maxcolors=-1, b2dmidpoints=False, b3dmidpoints=False, canvas_size=(800, 800), b2d_midpoint_values=0, titleNote=''):
    global colors
    canvas = Canvas(canvas_size, coloring=color)
    if maxcolors > 0 and maxcolors < len(colors):
        colors = colors[:maxcolors]
    canvas.plot_call = 'PlotBlob3ds'
    canvas.set_blobs(blob3dlist)


    # HACK
    canvas.add_simple_beads(blob3dlist)
    # /HACK

    if color == 'bead' or canvas.buffering:
        canvas.add_bead_markers(blob3dlist)

    if color == 'blob' or color == 'blob3d' or canvas.buffering: # Note: This is very graphics intensive.
        canvas.add_blob3d_markers(blob3dlist)

    if color == 'depth' or canvas.buffering: # Coloring based on recursive depth
        canvas.add_depth_markers(blob3dlist)

    if color == 'neutral' or canvas.buffering:
        canvas.add_color_markers(blob3dlist, markertype='neutral')

    if stitches or canvas.buffering:
        if lineColoring == 'blob3d' or canvas.buffering: # TODO need to change this so that stitchlines of the same color are the same object
            if Config.test_instead_of_data:
                canvas.add_blob3d_stitches(blob3dlist)
            else:
                print('Skipping adding blob3d stitches as it overloads ram (for now)')
        if lineColoring == 'neutral' or canvas.buffering:
            canvas.add_neutral_stitches(blob3dlist)

    if costs > 0:
        number_of_costs_to_show = costs # HACK
        all_stitches = list(stitches for blob3d in blob3dlist for pairing in blob3d.pairings for stitches in pairing.stitches)
        all_stitches = sorted(all_stitches, key=lambda stitch: stitch.cost[2], reverse=True) # costs are (contour_cost, distance(as cost), total, distance(not as cost))
        midpoints = np.zeros([number_of_costs_to_show,3])
        for index,stitch in enumerate(all_stitches[:number_of_costs_to_show]): #FIXME! For some reason overloads the ram.
            midpoints[index] = [(stitch.lowerpixel.x + stitch.upperpixel.x) / (2 * xdim), (stitch.lowerpixel.y + stitch.upperpixel.y) / (2 * ydim), (stitch.lowerpixel.z + stitch.upperpixel.z) / (2 * zdim)]
            textStr = str(stitch.cost[0])[:2] + '_' +  str(stitch.cost[3])[:3] + '_' +  str(stitch.cost[2])[:2]
            canvas.view.add(visuals.Text(textStr, pos=midpoints[index], color='yellow'))

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

def plotBlob2d(b2d, canvas_size=(1080,1080)):
    #Automatically labels all pixels
    global colors
    canvas = Canvas(canvas_size, coloring='blob2d')
    canvas.plot_call = 'PlotBlob2ds'
    canvas.set_blobs([b2d])
    canvas.add_blob2d_markers([b2d], edge=False, offset=True, explode=False)
    for pixel in b2d.pixels:
        pixel = Pixel.get(pixel)
        canvas.view.add(visuals.Text(str(pixel.id), pos=[(pixel.x - canvas.xmin) / canvas.xdim,(pixel.y - canvas.ymin) / canvas.ydim,(pixel.z - canvas.zmin) / canvas.zdim], font_size=15, bold=False, color='w'))


    canvas.setup_markers()
    canvas.setup_stitches()
    vispy.app.run()
    print("EXITING plotb2d")

def filter_available_colors():
    global colors
    if Config.mayPlot:
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