__author__ = 'gio'
# This file includes the various functions written to visualize data from within sero.py
# These functions have been separated for convenience; they are higher volume and lower maintenance.

import numpy as np
import vispy.io
import vispy.scene
from vispy.scene import visuals
from vispy.util import keys

from Blob2d import Blob2d
from Pixel import Pixel
from myconfig import Config
from util import warn, debug

colors = None
color_dict = None
rgba_colors = None

from Blob3d import get_blob2ds_b3ds, Blob3d


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

    all_trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(
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
    printl("Plot is available at url: " + str(plot_url))


class Canvas(vispy.scene.SceneCanvas):
    def __init__(self, canvas_size=(800, 800), title='', coloring='simple',
                 buffering=True):  # Note may want to make buffering default to False
        vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True, size=canvas_size, title=title)
        if hasattr(self, 'unfreeze') and callable(
                getattr(self, 'unfreeze')):  # HACK # Interesting bug fix for an issue that only occurs on Envy
            self.unfreeze()
        self.view = self.central_widget.add_view()

        self.fov = 50  # Must be 0 < fov < 180

        turn_camera = vispy.scene.cameras.TurntableCamera(fov=0, azimuth=80, parent=self.view.scene, distance=1,
                                                          elevation=-55, name='Turntable')
        # Using a fov of zero b/c fov makes turn harder to use
        fly_camera = vispy.scene.cameras.FlyCamera(parent=self.view.scene, fov=self.fov, name='Fly')
        panzoom_camera = vispy.scene.cameras.PanZoomCamera(parent=self.view.scene, name='Panzoom')
        arcball_camera = vispy.scene.cameras.ArcballCamera(parent=self.view.scene, fov=self.fov, distance=1,
                                                           name='Arcball')

        # TODO adjust _keymap of FlyCamera to tune turning to be less extreme
        # print('Fly_camera keymap: ' + str(fly_camera._keymap))

        turn_speed = .6
        assert (
            turn_speed >= .6)  # This is because below this value, the camera stops reponding to turn requests when not already moving

        fly_camera.set_range(x=(0, 1), y=(0, 1), z=(0, 1))  # Doesn't seem to improve performance..
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
        self.xmin = self.ymin = self.zmin = self.xmax = self.ymax = self.zmax = self.xdim = self.ydim = self.zdim = None

        self.view.add(visuals.Text('X', pos=[1, 0, 0], font_size=15, bold=False, color='w'))
        self.view.add(visuals.Text('Y', pos=[0, 1, 0], font_size=15, bold=False, color='w'))
        self.view.add(visuals.Text('Z', pos=[0, 0, 1], font_size=15, bold=False, color='w'))

        self.view.camera = self.cameras[self.current_camera_index]
        self.blob2d_coloring_markers = []
        self.blob3d_coloring_markers = []
        self.depth_coloring_markers = []
        self.show()
        self.coloring = coloring.lower()

        # print('----Starting canvas color: ' + str(self.coloring))
        self.markers = []
        self.available_marker_colors = ['simple', 'blob3d', 'bead', 'depth', 'blob2d', 'b2d_depth', 'neutral', 'blob2d_to_3d']
        self.available_stitch_colors = ['simple', 'neutral', 'parent_id', 'blob3d']
        self.current_blob_color = self.coloring
        self.buffering = buffering
        self.marker_colors = []  # Each entry corresponds to the color of the correspond 'th marker in self.view.scene.children (not all markers!)
        self.image_no = 0
        self.b3ds = []
        self.b2ds = []
        self.stitches = []
        self.current_stitch_color = 'simple'
        self.plot_call = ''  # Used to remember which function created the canvas

        # self.measure_fps()

    @staticmethod
    def on_mouse_press(event):
        """Pan the view based on the change in mouse position.
        :param event:
        """
        if event.button == 1:
            # x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            print(x1, y1)
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

        if event.key.name == 'Up':  # Next color cheme
            self.update_markers(increment=1)

        elif event.key.name == 'Down':  # Previous color scheme
            self.update_markers(increment=-1)

        elif event.key.name == 'Insert':  # Save an image
            img = self.render()
            img_name = self.name_image()
            print('Writing to image file: \'' + str(img_name) + '\'')
            vispy.io.write_png(img_name, img)

        elif event.key.name == 'T':  # test
            for child, coloring in self.markers:
                print(child)
                for name, val in child.__dict__.items():
                    print('   ' + str(name) + ' : ' + str(val))
                    debug()
        elif event.key.name == 'Left':  # Toggle stitches
            self.update_stitches(increment=1)

        # These numerical keys are for adjusting bead values
        elif event.key.name == '1':
            print('Changing config.max_pixels_to_be_a_bead from ' + str(Config.max_pixels_to_be_a_bead), end='')
            # from myconfig import set_config.max_pixels_to_be_a_bead
            Config.max_pixels_to_be_a_bead = int(1.1 * Config.max_pixels_to_be_a_bead) + 10
            print(' to ' + str(
                Config.max_pixels_to_be_a_bead))  # + ' which by get = ' + str(config.get_config.max_pixels_to_be_a_bead()))
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

        elif event.key.name == 'V':  # Change cameras
            self.current_camera_index = (self.current_camera_index + 1) % len(self.cameras)
            self.view.camera = self.cameras[self.current_camera_index]
            self.update_title()

        elif event.key.name == '=':  # Increase fov for all cameras
            self.fov = (self.fov + 10) % 180
            for camera in self.cameras:
                if camera.name in ['Fly', 'Arcball']:  # and hasattr(camera, 'fov'):
                    camera.fov = self.fov
            self.update_title()

        elif event.key.name == '-':  # Decrease fov for all cameras
            if self.fov - 10 < 0:
                self.fov = 0
            else:
                self.fov -= 10
            for camera in self.cameras:
                if camera.name in ['Fly', 'Arcball']:  # if hasattr(camera, 'fov'):
                    camera.fov = self.fov
            self.update_title()

    def refresh_bead_markers(self):
        self.remove_markers_of_color('bead')
        self.add_bead_markers(self.b3ds)  # Also re-tags
        self.update()

    def remove_markers_of_color(self, color):
        remove_markers = []
        for marker, coloring in self.markers:
            if coloring == color:
                # self.view.remove(marker)
                remove_markers.append((marker, coloring))
        for marker, coloring in remove_markers:
            # try:  # HACK
            marker.remove_parent(marker.parent)  # FIXME
            # except:
            #     pass
            self.markers.remove((marker, coloring))
            del marker

    def next_marker_color(self, increment=1):
        assert (increment in [1, -1])
        if self.current_blob_color in self.available_marker_colors:
            return self.available_marker_colors[
                (self.available_marker_colors.index(self.current_blob_color) + increment) % len(
                    self.available_marker_colors)]
        else:
            return self.available_marker_colors[0]

    def next_stitch_color(self, increment=1):
        assert (increment in [1, -1])
        if self.current_stitch_color in self.available_stitch_colors:
            return self.available_stitch_colors[
                (self.available_stitch_colors.index(self.current_stitch_color) + increment) % len(
                    self.available_stitch_colors)]
        else:
            return self.available_stitch_colors[0]

    def set_stitch_color(self, new_color):
        self.current_stitch_color = new_color
        self.update_stitches(increment=0)

    def update_stitches(self, increment=1):
        assert increment in [-1, 0, 1]  # 0 is a refresh
        if len(self.available_stitch_colors):
            if increment != 0:
                self.current_stitch_color = self.next_stitch_color(increment=increment)
            for stitch, color in self.stitches:
                if color == self.current_stitch_color:  # and \
                    # not (self.current_stitch_color == 'blob3d' and self.current_blob_color == 'blob3d' and color == 'parent_id'): #Hides parent_id lines when plotting blob3d b/c exploding is turned off
                    stitch.visible = True
                else:
                    stitch.visible = False
        self.update_title()

    def update_markers(self, increment=1):
        assert increment in [-1, 0, 1]  # 0 is a refresh
        if len(self.available_marker_colors):
            if increment != 0:
                self.current_blob_color = self.next_marker_color(increment=increment)
            for child, coloring in self.markers:
                if coloring == self.current_blob_color:
                    child.visible = True
                else:
                    child.visible = False
            if self.current_blob_color == 'blob3d':  # for now, these are viewed flat, and so parentlines should be turned off
                for stitch, color in self.stitches:
                    if color == 'parent_id':
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

    def update_title(self):
        self.title = str(self.plot_call) + ': # B3ds: ' + str(len(self.b3ds)) + ', # B2ds: ' + str(
            len(self.b2ds)) + ', Coloring = ' + str(self.current_blob_color) + ', Stitches = ' + str(
            self.current_stitch_color) + ', Camera = ' + self.extract_camera_name() + ', fov = ' + str(self.fov)

    def extract_camera_name(self):
        # buf = str(type(self.cameras[self.current_camera_index]))
        # return buf[buf.rindex("."):buf.rindex("'")]
        return self.view.camera.name

    def setup_markers(self):
        counts = [0] * len(self.available_marker_colors)
        for marker, coloring in self.markers:
            counts[self.available_marker_colors.index(coloring)] += 1
        self.available_marker_colors = [color for (index, color) in enumerate(self.available_marker_colors) if
                                        counts[index] > 0]
        for marker, coloring in self.markers:
            if coloring == self.current_blob_color:
                marker.visible = True
            else:
                marker.visible = False
        self.update_title()

    def add_marker(self, marker, coloring):
        self.markers.append((marker, coloring))
        self.view.add(self.markers[-1][0])  # add the above marker

    def add_markers_from_groups(self, b3d_groups, type_string, midpoints=True, list_of_colors=colors, size=8, explode=True):
        # Midpoints true if just doing midpoints, otherwise do edge_pixels
        my_rgba_colors = list(color_to_rgba(color) for color in list_of_colors)

        if midpoints:
            point_count = sum(len(b3d_group) for b3d_group in b3d_groups)
        else:
            if type_string in ['b2d_depth', 'blob2d', 'blob2d_to_3d']:
                point_count = sum(len(b2d.edge_pixels) for b2d_group in b3d_groups for b2d in b2d_group)
            else:
                point_count = sum(b3d.get_edge_pixel_count() for b3d_group in b3d_groups for b3d in b3d_group)
        marker_pos = np.zeros((point_count, 3))
        marker_colors = np.zeros((point_count, 4))
        index = 0
        for group_index, b3d_group in enumerate(b3d_groups):
            for blob in b3d_group:
                if midpoints:
                    marker_pos[index] = [blob.avgx / self.xdim, blob.avgy / self.ydim, blob.avgz / (Config.z_compression * self.zdim)]
                    marker_colors[index] = my_rgba_colors[group_index % len(my_rgba_colors)]
                    index += 1
                else:
                    if type_string in ['b2d_depth', 'blob2d', 'blob2d_to_3d']:
                        cur_ep = [Pixel.get(pixel) for pixel in blob.edge_pixels]
                        for pixel in cur_ep:
                            marker_pos[index] = [pixel.x / self.xdim, pixel.y / self.ydim,
                                                 getBloomedHeight(blob, explode, self.zdim) / (Config.z_compression * self.zdim)]
                            marker_colors[index] = my_rgba_colors[group_index % len(my_rgba_colors)]
                            index += 1
                    else:
                        cur_ep = blob.get_edge_pixels()
                        for pixel in cur_ep:
                            marker_pos[index] = [pixel.x / self.xdim, pixel.y / self.ydim,
                                                 pixel.z / (Config.z_compression * self.zdim)]
                            marker_colors[index] = my_rgba_colors[group_index % len(my_rgba_colors)]
                            index += 1

        markers = visuals.Markers()
        # print("Result of marker_pos for type: " + str(type_string) + ' : ' + str(marker_pos))
        # print("Result of marker_colors for type: " + str(type_string) + ' : ' + str(marker_colors))
        # print("Point count: " + str(point_count))
        markers.set_data(marker_pos, face_color=marker_colors, size=size) #, edge_color=edge_color)
        self.add_marker(markers, type_string)

    def add_bead_markers(self, blob3dlist):
        Blob3d.tag_all_beads()         # Tagging beads for safety
        beads_nonbeads = [[], []]
        for blob3d in blob3dlist:
            beads_nonbeads[not blob3d.isBead].append(blob3d) # Inverted so that green is drawn after red,
                                                             #  to make it easier to see in denser regions
        self.add_markers_from_groups(beads_nonbeads, 'bead', midpoints=False, list_of_colors=['red','green'], size=8)

    def add_blob3d_markers(self, blob3dlist):  # Note that for now, this only does edges
        b3d_groups = [[] for _ in range(min(len(colors), len(blob3dlist)))]
        for blobnum, blob3d in enumerate(blob3dlist):
            b3d_groups[blobnum % len(b3d_groups)].append(blob3d)
        self.add_markers_from_groups(b3d_groups, 'blob3d', midpoints=False, list_of_colors=colors, size=8)

    def add_depth_markers(self, blob3dlist):
        max_depth = max(blob3d.recursive_depth for blob3d in blob3dlist)
        b3ds_by_depth = [[] for _ in range(max_depth + 1)]
        for b3d in blob3dlist:
            b3ds_by_depth[b3d.recursive_depth].append(b3d)
        self.add_markers_from_groups(b3ds_by_depth, 'depth', midpoints=False, list_of_colors=colors, size=8)

    def add_depth_markers_from_blob2ds(self, blob2dlist, offset=False, explode=True):
        max_depth = max(blob2d.recursive_depth for blob2d in blob2dlist)
        b2ds_by_depth = [[] for _ in range(max_depth + 1)]
        print(len(b2ds_by_depth))
        for b2d in blob2dlist:
            b2ds_by_depth[b2d.recursive_depth].append(b2d)
        self.add_markers_from_groups(b2ds_by_depth, 'b2d_depth', midpoints=False, list_of_colors=colors, size=8, explode=True)

    def add_blob2d_markers(self, blob2dlist, explode=True):
        num_colors_used = min(len(colors), len(blob2dlist))
        b2ds_by_color = [[] for _ in range(num_colors_used + 1)]
        for index, b2d in enumerate(blob2dlist):
            b2ds_by_color[index % num_colors_used].append(b2d)
        self.add_markers_from_groups(b2ds_by_color, 'blob2d', midpoints=False, list_of_colors=colors[:num_colors_used],size=8, explode=explode)

    def add_blob3d_markers_from_blob2ds(self, blob2dlist):  # This makes sure that the whole b3d isnt shown, instead just the relevant b2ds
        max_b3d_id = max(b2d.b3did for b2d in blob2dlist)
        b3d_lists = [[] for _ in range(max_b3d_id + 2)]  # +2 to allow for blobs which are unassigned
        for b2d in blob2dlist:
            b3d_lists[b2d.b3did].append(b2d)
        if len(b3d_lists[-1]):
            warn('Plotting b2ds that weren\'t assigned ids (while converting to b3ds)')
        b3d_lists = [b3d_list for b3d_list in b3d_lists if len(b3d_list)]
        self.add_markers_from_groups(b3d_lists, 'blob2d_to_3d', midpoints=False, list_of_colors=colors, size=8, explode=False)

    def add_neutral_markers(self, blob3dlist, color='aqua'):
        self.add_markers_from_groups([blob3dlist], 'neutral', midpoints=False, list_of_colors=[color], size=8, explode=False)


    def add_simple_beads(self, blob3dlist):  # NOTE only doing strands for now for simplicity
        base_b3ds = [b3d for b3d in blob3dlist if b3d.recursive_depth == 0]  # and not b3d.isBead]
        print(' Number of base_b3ds = ' + str(len(base_b3ds)) + ' / ' + str(len(blob3dlist)))
        bead_groups = []
        all_first_children = []
        for b3d in base_b3ds:
            if b3d.isBead is not True and b3d.isBead is not None:
                first_children = [blob3d for blob3d in b3d.get_first_child_beads() if blob3d.isBead]
                if not len(first_children):
                    warn('Found non_bead b3d without any first children beads ' + str(b3d))  # FIXME TODO
                else:
                    bead_groups.append(first_children)
                    all_first_children += first_children
            else:
                bead_groups.append([b3d])
        print(" Number of bead groups: " + str(len(bead_groups)))
        bg_of_one = []
        bg_more_than_one = []
        for bg in bead_groups:
            if len(bg) > 1:
                bg_more_than_one.append(bg)
            else:
                bg_of_one.append(bg[0])

        # Adding beads that were in groups of their own all together as one group:
        # This is done ONLY for speed reasons
        one_marker_midpoints = np.zeros([len(bg_of_one), 3])
        one_markers = visuals.Markers()
        for index,b3d in enumerate(bg_of_one):
            one_marker_midpoints[index] = [b3d.avgx / self.xdim, b3d.avgy / self.ydim, b3d.avgz / (Config.z_compression * self.zdim)]
        one_markers.set_data(one_marker_midpoints, face_color=colors[0], size=8, edge_color='yellow')
        self.add_marker(one_markers, 'simple')


        all_stitch_arr = np.zeros((0,3))

        num_markers = sum(len(bg) for bg in bg_more_than_one)
        connections = np.empty((num_markers, 2))
        stitch_colors = np.empty((num_markers, 4))
        marker_index = 0

        self.add_markers_from_groups(bg_more_than_one, 'simple', list_of_colors=colors, size=12)
        for index, bg in enumerate(bg_more_than_one):
            bg = sorted(bg, key=lambda blob3d: (blob3d.avgx, blob3d.avgy)) # TODO improve this or do segmentation of sorts..
            # Maybe look for two long lines, that share an endpoint, and replace on of them with a link to the other's other point
            marker_midpoints = np.zeros([len(bg), 3])
            for group_index, b3d in enumerate(bg):
                if group_index == len(bg) - 1:
                    connections[marker_index] = [marker_index, marker_index]
                else:
                    connections[marker_index] = [marker_index, marker_index + 1]
                stitch_colors[marker_index] = color_to_rgba(colors[index % len(colors)])
                marker_midpoints[group_index] = [b3d.avgx / self.xdim, b3d.avgy / self.ydim, b3d.avgz / (Config.z_compression * self.zdim)]
                marker_index += 1
            all_stitch_arr = np.concatenate((all_stitch_arr, marker_midpoints))
        all_lines = visuals.Line(method=Config.linemethod, color=stitch_colors, width=3)
        all_lines.set_data(pos=all_stitch_arr, connect=connections)
        self.add_stitch(all_lines, 'simple')


    def setup_stitches(self):
        # Going to count whether there are any stitches of each type,
        # therefore if there aren't that type can be skipped
        counts = [0] * len(self.available_stitch_colors)
        for stitch, coloring in self.stitches:
            counts[self.available_stitch_colors.index(coloring)] += 1
        self.available_stitch_colors = [color for (index, color) in enumerate(self.available_stitch_colors) if
                                        counts[index] > 0]
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

    def add_blob3d_stitches(self, blob3dlist):  # FIXME
        num_markers = min(len(blob3dlist), len(colors))
        lines_per_color = [0] * num_markers
        offsets = [0] * num_markers
        line_location_lists = []
        for blob_num, b3d in enumerate(blob3dlist):
            lines_per_color[blob_num % num_markers] += sum(
                len(pairing.indeces) for blob3d in blob3dlist for pairing in blob3d.pairings)
        for line_count in lines_per_color:
            line_location_lists.append(np.zeros([2 * line_count, 3]))  # 2x because we are storing endpoints
        for blob_num, blob3d in enumerate(blob3dlist):
            for pairing in blob3d.pairings:
                for stitch_no, stitch in enumerate(pairing.stitches):
                    lowerpixel = Pixel.get(stitch.lowerpixel)
                    upperpixel = Pixel.get(stitch.upperpixel)
                    blob_color_index = blob_num % num_markers
                    line_location_lists[blob_color_index][offsets[blob_color_index] + (2 * stitch_no)] = [
                        lowerpixel.x / self.xdim, lowerpixel.y / self.ydim,
                        pairing.lowerheight / (Config.z_compression * self.zdim)]
                    line_location_lists[blob_color_index][offsets[blob_color_index] + (2 * stitch_no) + 1] = [
                        upperpixel.x / self.xdim, upperpixel.y / self.ydim,
                        pairing.upperheight / (Config.z_compression * self.zdim)]
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
                line_end_points += (2 * len(stitch.indeces))  # 2 as each line has 2 endpoints
        line_locations = np.zeros([line_end_points, 3])
        for blob3d in blob3dlist:
            for pairing in blob3d.pairings:
                for stitch in pairing.stitches:
                    lowerpixel = Pixel.get(stitch.lowerpixel)
                    upperpixel = Pixel.get(stitch.upperpixel)
                    line_locations[line_index] = [lowerpixel.x / self.xdim, lowerpixel.y / self.ydim,
                                                  pairing.lowerheight / (Config.z_compression * self.zdim)]
                    line_locations[line_index + 1] = [upperpixel.x / self.xdim, upperpixel.y / self.ydim,
                                                      pairing.upperheight / (Config.z_compression * self.zdim)]
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
                        line_locations[line_index] = [(lowerpixel.x - xoffset) / self.xdim,
                                                      (lowerpixel.y - yoffset) / self.ydim,
                                                      (pairing.lowerheight - zoffset) / (
                                                          Config.z_compression * self.zdim)]
                        line_locations[line_index + 1] = [(upperpixel.x - xoffset) / self.xdim,
                                                          (upperpixel.y - yoffset) / self.ydim,
                                                          (pairing.upperheight - zoffset) / (
                                                              Config.z_compression * self.zdim)]
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
        for num, b2d in enumerate(blob2dlist):
            lineendpoints += (2 * len(b2d.children))
        if lineendpoints:
            line_index = 0
            line_locations = np.zeros([lineendpoints, 3])
            for b2d in blob2dlist:
                for child in b2d.children:
                    child = Blob2d.get(child)
                    line_locations[line_index] = [(b2d.avgx - xoffset) / self.xdim, (b2d.avgy - yoffset) / self.ydim,
                                                  (getBloomedHeight(b2d, explode, self.zdim) - zoffset) / (
                                                      Config.z_compression * self.zdim)]
                    line_locations[line_index + 1] = [(child.avgx - xoffset) / self.xdim,
                                                      (child.avgy - yoffset) / self.ydim,
                                                      (getBloomedHeight(child, explode, self.zdim) - zoffset) / (
                                                          Config.z_compression * self.zdim)]
                    line_index += 2
            parent_lines = visuals.Line(method=Config.linemethod)
            parent_lines.set_data(pos=line_locations, connect='segments', color='y')
            self.add_stitch(parent_lines, 'parent_id')

    def set_blobs(self, bloblist):
        """
        Setups the necessary parts of the canvas according to supplied blob2ds or blob3ds (exclusive)
        Inculding self.b2ds, self.b3ds (regardless of whether given b2ds or b3ds will set both)
        Then goes on to set the x,y,z info of the canvas based on the blobs supplied
        :param bloblist: A list of entirely blob2ds or entirely blob3ds
        :return:
        """
        if all(type(blob) is Blob3d for blob in bloblist):
            # Given a list of blob3ds
            self.b3ds = bloblist
            self.b2ds = [blob2d for blob3d in bloblist for blob2d in blob3d.blob2ds]
            self.set_canvas_bounds(b2ds_not_b3ds=False)
        elif all(type(blob) is Blob2d for blob in bloblist):
            # Given a list of blob2ds
            self.b2ds = bloblist
            self.b3ds = get_blob2ds_b3ds(bloblist, ids=False)
            self.set_canvas_bounds(b2ds_not_b3ds=True)
        else:
            warn('Given a list not of blob2ds or blob3ds entirely!!!')
            print(bloblist)
            debug()
            exit(1)

    def set_canvas_bounds(self, b2ds_not_b3ds):
        """
        Sets (x,y,z) (min, max, dim) using either canvas.b2ds or canvas.b3ds
        :param b2ds_not_b3ds:
        :return:
        """
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
        print("xmin: " + str(self.xmin) + ", xmax: " + str(self.xmax) + ", ymin: " + str(self.ymin) + ", ymax: " + str(
            self.ymax) + ", zmin: " + str(self.zmin) + ", zmax: " + str(self.zmax))
        print("xdim: " + str(self.xdim) + ', ydim: ' + str(self.ydim) + ', zdim: ' + str(self.zdim))


def plot_b2ds(blob2ds, coloring='', canvas_size=(800, 800), ids=False, stitches=False, titleNote='', edge=True,
              buffering=True, parentlines=False, explode=False, showStitchCosts=0, b2dmidpoints=False, offset=False,
              pixel_ids=False):
    global colors
    coloring = coloring.lower()
    assert coloring in ['blob2d', '', 'b2d_depth', 'blob3d', 'bead', 'simple']

    # This block allows the passing of ids or blob2ds
    if len(blob2ds) == 0:
        warn('Tried to plot 0 blob2ds')
    else:
        all_b2ds_are_ids = all(type(b2d) is int for b2d in blob2ds)
        all_b2d_are_blob2ds = all(type(b2d) is Blob2d for b2d in blob2ds)
        assert (all_b2d_are_blob2ds or all_b2ds_are_ids)
        if all_b2ds_are_ids:  # May want to change this depending on what it does to memory
            blob2ds = [Blob2d.get(b2d) for b2d in blob2ds]

        if coloring == '':
            coloring = 'blob2d'  # For the canvas title

        canvas = Canvas(canvas_size, coloring='blob2d', buffering=buffering,
                        title='plotBlob2ds(' + str(len(blob2ds)) + '-Blob2ds, coloring=' + str(
                            coloring) + ' canvas_size=' + str(canvas_size) + ') ' + titleNote)
        canvas.plot_call = 'PlotBlob2ds'
        canvas.set_blobs(blob2ds)
        # TODO
        if coloring == 'simple' or canvas.buffering:
            canvas.add_simple_beads(canvas.b3ds)
        # TODO


        if coloring == 'blob2d' or canvas.buffering:
            canvas.add_blob2d_markers(blob2ds, explode=explode)

        if coloring == 'blob3d' or canvas.buffering:
            canvas.add_blob3d_markers_from_blob2ds(blob2ds)

        if coloring == 'bead' or canvas.buffering:
            canvas.add_bead_markers(canvas.b3ds)  # HACK

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
            print(
                "\nWARNING adding ids for every blob2d, this could overload if not a small dataset, so skipping if not a test_set!!")
            if Config.test_instead_of_data:
                for b2d_num, b2d in enumerate(blob2ds):
                    # if b2d.recursive_depth == 0:
                    midpoints = [(b2d.avgx - canvas.xmin) / canvas.xdim, (b2d.avgy - canvas.ymin) / canvas.ydim, (
                        (getBloomedHeight(b2d, explode, canvas.zdim) + .25 - canvas.zmin) / (
                            Config.z_compression * canvas.zdim))]
                    textStr = str(b2d.id)
                    color = colors[b2d.id % len(colors)]
                    canvas.view.add(visuals.Text(textStr, pos=midpoints, color=color, font_size=8, bold=False))

        canvas.setup_markers()
        canvas.setup_stitches()
        vispy.app.run()


def plot_b3ds(blob3dlist, stitches=True, color='blob3d', lineColoring=None, costs=0, maxcolors=-1, b2dmidpoints=False,
              b3dmidpoints=False, canvas_size=(800, 800), b2d_midpoint_values=0, titleNote=''):
    global colors
    canvas = Canvas(canvas_size, coloring=color)
    if 0 < maxcolors < len(colors):
        colors = colors[:maxcolors]
    canvas.plot_call = 'PlotBlob3ds'
    canvas.set_blobs(blob3dlist)

    # HACK
    canvas.add_simple_beads(blob3dlist)
    # /HACK

    if color == 'bead' or canvas.buffering:
        canvas.add_bead_markers(blob3dlist)

    if color == 'blob' or color == 'blob3d' or canvas.buffering:  # Note: This is very graphics intensive.
        canvas.add_blob3d_markers(blob3dlist)

    if color == 'depth' or canvas.buffering:  # Coloring based on recursive depth
        canvas.add_depth_markers(blob3dlist)

    if color == 'neutral' or canvas.buffering:
        canvas.add_neutral_markers(blob3dlist)

    if stitches or canvas.buffering:
        if lineColoring == 'blob3d' or canvas.buffering:  # TODO need to change this so that stitchlines of the same color are the same object
            if Config.test_instead_of_data:
                canvas.add_blob3d_stitches(blob3dlist)
            else:
                print('Skipping adding blob3d stitches as it overloads ram (for now)')
        if lineColoring == 'neutral' or canvas.buffering:
            canvas.add_neutral_stitches(blob3dlist)

    canvas.setup_markers()
    canvas.setup_stitches()
    vispy.app.run()


def plotBlob2d(b2d, canvas_size=(1080, 1080)):
    # Automatically labels all pixels
    global colors
    canvas = Canvas(canvas_size, coloring='blob2d')
    canvas.plot_call = 'PlotBlob2ds'
    canvas.set_blobs([b2d])
    canvas.add_blob2d_markers([b2d], edge=False, offset=True, explode=False)
    for pixel in b2d.pixels:
        pixel = Pixel.get(pixel)
        canvas.view.add(visuals.Text(str(pixel.id),
                                     pos=[(pixel.x - canvas.xmin) / canvas.xdim, (pixel.y - canvas.ymin) / canvas.ydim,
                                          (pixel.z - canvas.zmin) / canvas.zdim], font_size=15, bold=False, color='w'))
    canvas.setup_markers()
    canvas.setup_stitches()
    vispy.app.run()


def filter_available_colors():
    global colors
    global color_dict
    global rgba_colors
    if Config.mayPlot:
        colors = vispy.color.get_color_names()  # ALl possible colors
        color_dict = vispy.color.get_color_dict()
        # note getting rid of annoying colors
        rejectwords = ['dark', 'light', 'slate', 'grey', 'white', 'pale', 'medium']
        removewords = []
        for knum, key in enumerate(colors):
            for word in rejectwords:
                if len(key) == 1:
                    removewords.append(key)
                    break
                elif key.find(word) != -1:  # found
                    removewords.append(key)
                    break
        colors = list(set(colors) - set(removewords))
        rgba_colors = list()
        removecolors = ['aliceblue', 'azure', 'blanchedalmond', 'aquamarine', 'beige', 'bisque', 'black', 'blueviolet',
                        'brown', 'burlywood', 'cadetblue', 'chocolate', 'coral', 'cornsilk', 'cornflowerblue',
                        'chartreuse', 'crimson', 'cyan', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick',
                        'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'gray', 'greenyellow', 'honeydew',
                        'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush',
                        'lawngreen', 'lemonchiffon', 'linen', 'olive', 'olivedrab', 'limegreen', 'midnightblue',
                        'mintcream', 'mistyrose', 'moccasin', 'navy', 'orangered', 'orchid', 'papayawhip', 'peachpuff',
                        'peru', 'pink', 'powderblue', 'plum', 'rosybrown', 'saddlebrown', 'salmon', 'sandybrown',
                        'seagreen', 'seashell', 'silver', 'sienna', 'skyblue', 'springgreen', 'tan', 'teal', 'thistle',
                        'tomato', 'turquoise', 'snow', 'steelblue', 'violet', 'wheat', 'yellowgreen']
        for color in removecolors:
            colors.remove(color)
        for color in colors:
            rgba_colors.append(color_to_rgba(color))
        colors = sorted(colors)
        print('There are a total of ' + str(len(colors)) + ' colors available for plotting')
        # openglconfig = vispy.gloo.wrappers.get_gl_configuration() # Causes opengl/vispy crash for unknown reasons

def color_to_rgba(color_str):
    return vispy.color.ColorArray(color_dict[color_str]).rgba


def showColors(canvas_size=(800, 800)):
    global colors
    canvas = Canvas(canvas_size)
    print(colors)
    print('There are a total of ' + str(len(colors)) + ' colors used for plotting')
    for i, color in enumerate(colors):
        canvas.view.add(
            visuals.Text(color, pos=np.reshape([0, 0, 1 - (i / len(colors))], (1, 3)), color=color, bold=True))
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
