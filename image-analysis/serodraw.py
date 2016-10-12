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
from util import *
import matplotlib.pyplot as plt
import pandas as pd
from Blob3d import *

colors = None
color_dict = None
rgba_colors = None



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
    def __init__(self, canvas_size=(1000, 1000), coloring='blob3d',
                 buffering=True, debug_colors=False):  # Note may want to make buffering default to False
        vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True, size=canvas_size)
        if hasattr(self, 'unfreeze') and callable(
                getattr(self, 'unfreeze')):  # HACK # Interesting bug fix for an issue that only occurs on Envy
            self.unfreeze()
        self.view = self.central_widget.add_view()
        self.debug_colors = debug_colors # Whether or not to add and show merged_blob3ds and merged_parents
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

        turn_speed = .6 # This is because below this value, the camera stops reponding to turn requests when not already moving

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

        self.markers = []
        self.available_marker_colors = ['blob3d', 'simple', 'bead', 'depth', 'blob2d', 'b2d_depth', 'neutral', 'blob2d_to_3d']
        if self.debug_colors:
            self.available_marker_colors += ['merged', 'merged_parents']

        self.available_stitch_colors = ['simple', 'neutral', 'parent_id', 'blob3d']
        self.current_blob_color = self.coloring
        self.buffering = buffering
        self.marker_colors = []  # Each entry corresponds to the color of the correspond 'th marker in self.view.scene.children (not all markers!)
        self.image_no = 0
        self.b3ds = []
        self.b2ds = []
        self.stitches = []
        self.current_stitch_color = 'neutral'
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
            point_count = sum(len(b3d_group) for b3d_group in b3d_groups)  # Lists of blob3ds
        else:
            if type_string in ['b2d_depth', 'blob2d', 'blob2d_to_3d']:  # Lists of blob2ds
                point_count = sum(len(b2d.edge_pixels) for b2d_group in b3d_groups for b2d in b2d_group)
            else:  # Lists of blob3ds
                point_count = sum(b3d.get_edge_pixel_count() for b3d_group in b3d_groups for b3d in b3d_group)
        marker_pos = np.zeros((point_count, 3))
        marker_colors = np.zeros((point_count, 4))
        index = 0
        for group_index, b3d_group in enumerate(b3d_groups):
            for blob in b3d_group:
                if midpoints:
                    marker_pos[index] = [(blob.avgx - self.xmin) / self.xdim, (blob.avgy - self.ymin) / self.ydim, (blob.avgz - self.zmin) / (Config.z_compression * self.zdim)]
                    marker_colors[index] = my_rgba_colors[group_index % len(my_rgba_colors)]
                    index += 1
                else:
                    if type_string in ['b2d_depth', 'blob2d', 'blob2d_to_3d']:
                        cur_ep = [Pixel.get(pixel) for pixel in blob.edge_pixels]
                        for pixel in cur_ep:
                            marker_pos[index] = [(pixel.x - self.xmin) / self.xdim, (pixel.y - self.ymin) / self.ydim,
                                                 (getBloomedHeight(blob, explode, self.zdim) - self.zmin) / (Config.z_compression * self.zdim)]
                            marker_colors[index] = my_rgba_colors[group_index % len(my_rgba_colors)]
                            index += 1
                    else:  # Lists of blob3ds
                        cur_ep = blob.get_edge_pixels()
                        for pixel in cur_ep:
                            marker_pos[index] = [(pixel.x - self.xmin) / self.xdim, (pixel.y - self.ymin) / self.ydim,
                                                 (pixel.z - self.zmin) / (Config.z_compression * self.zdim)]
                            marker_colors[index] = my_rgba_colors[group_index % len(my_rgba_colors)]
                            index += 1

        markers = visuals.Markers()
        # print("Result of marker_pos for type: " + str(type_string) + ' : ' + str(marker_pos)) # DEBUG
        # print("Result of marker_colors for type: " + str(type_string) + ' : ' + str(marker_colors))
        # print("Point count: " + str(point_count))
        markers.set_data(marker_pos, face_color=marker_colors, size=size) #, edge_color=edge_color)
        self.add_marker(markers, type_string)

    def add_bead_markers(self):
        Blob3d.tag_all_beads()         # Tagging beads for safety
        beads_nonbeads = [[], []]
        for blob3d in self.b3ds:
            beads_nonbeads[not blob3d.isBead].append(blob3d) # Inverted so that green is drawn after red,
                                                             #  to make it easier to see in denser regions
        self.add_markers_from_groups(beads_nonbeads, 'bead', midpoints=False, list_of_colors=['red','green'], size=8)

    def add_blob2d_markers(self, explode=True):
        num_colors_used = min(len(colors), len(self.b2ds))
        b2ds_by_color = [[] for _ in range(num_colors_used + 1)]
        for index, b2d in enumerate(self.b2ds):
            b2ds_by_color[index % num_colors_used].append(b2d)
        self.add_markers_from_groups(b2ds_by_color, 'blob2d', midpoints=False,
                                     list_of_colors=colors[:num_colors_used], size=8, explode=explode)

    def add_blob3d_markers(self):  # Note that for now, this only does edges
        b3d_groups = [[] for _ in range(min(len(colors), len(self.b3ds)))]
        for blobnum, blob3d in enumerate(self.b3ds):
            b3d_groups[blobnum % len(b3d_groups)].append(blob3d)
        self.add_markers_from_groups(b3d_groups, 'blob3d', midpoints=False, list_of_colors=colors, size=8)

    def add_depth_markers(self, explode=False):
        max_depth = max(blob3d.recursive_depth for blob3d in self.b3ds)
        b3ds_by_depth = [[] for _ in range(max_depth + 1)]
        for b3d in self.b3ds:
            b3ds_by_depth[b3d.recursive_depth].append(b3d)
        self.add_markers_from_groups(b3ds_by_depth, 'depth', midpoints=False, list_of_colors=colors, size=8, explode=explode)

    def add_merged_markers(self):
        """
        Adds markers for blob3ds that have been merged, where each color (may cycle) represents a group that has been combined into a single blob3d
        This is a good way to visualize the effects of merging, and confirm that it is a) worthwhile and b) effective / low error
        :return:
        """
        if hasattr(Blob3d, "lists_of_merged_blob3ds") and len(Blob3d.lists_of_merged_blob3ds):  # Check for legacy pickle files and presence of blob3ds that have been merged
            self.add_markers_from_groups(Blob3d.lists_of_merged_blob3ds, 'merged', midpoints=False, list_of_colors=colors, size=8, explode=False)
        else:
            printl("Skipping adding markers for merged blob3ds, this is likely from a legacy pickle file, or a small dataset")  # Legacy

    def add_merged_parent_markers(self):
        if hasattr(Blob3d, "list_of_merged_blob3d_parents") and len(Blob3d.list_of_merged_blob3d_parents):  # Check for legacy pickle files and presence of blob3ds that have been merged
            self.add_markers_from_groups([[parent] for parent_list in Blob3d.list_of_merged_blob3d_parents for parent in parent_list], 'merged_parents', midpoints=False, list_of_colors=colors, size=8, explode=False)
        else:
            printl("Skipping adding markers for merged blob3d parents, this is likely from a legacy pickle file, or a small dataset")  # Legacy

            # Note adding them all seperately, as opposed to the style used in the original merged_markers

    def add_neutral_markers(self, color='aqua'):
        self.add_markers_from_groups([self.b3ds], 'neutral', midpoints=False, list_of_colors=[color], size=8, explode=False)

    def add_simple_beads(self):  # NOTE only doing strands for now for simplicity
        base_b3ds = [b3d for b3d in self.b3ds if b3d.recursive_depth == 0]  # and not b3d.isBead]
        print(' Number of base_b3ds = ' + str(len(base_b3ds)) + ' / ' + str(len(self.b3ds)))
        bead_groups = []
        all_first_children = []
        for b3d in base_b3ds:
            if b3d.isBead is not True and b3d.isBead is not None:
                first_children = [blob3d for blob3d in b3d.get_first_child_beads() if blob3d.isBead]
                if not len(first_children):
                    warn('Found non_bead b3d without any first children beads ' + str(b3d))  # TODO
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
            one_marker_midpoints[index] = [(b3d.avgx - self.xmin) / self.xdim, (b3d.avgy - self.ymin) / self.ydim, (b3d.avgz - self.zmin) / (Config.z_compression * self.zdim)]
        one_markers.set_data(one_marker_midpoints, face_color=colors[0], size=8, edge_color='yellow')
        self.add_marker(one_markers, 'simple')

        all_stitch_arr = np.zeros((0, 3))

        num_markers = sum(len(bg) for bg in bg_more_than_one)
        connections = np.empty((num_markers, 2))
        stitch_colors = np.empty((num_markers, 4))
        marker_index = 0

        print('bg of more than one: ' + str(bg_more_than_one))
        if len(bg_more_than_one):
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
                    marker_midpoints[group_index] = [(b3d.avgx - self.xmin) / self.xdim, (b3d.avgy - self.ymin) / self.ydim, (b3d.avgz - self.zmin) / (Config.z_compression * self.zdim)]
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

    def add_blob3d_stitches(self):  # FIXME
        num_markers = min(len(self.b3ds), len(colors))
        lines_per_color = [0] * num_markers
        offsets = [0] * num_markers
        line_location_lists = []
        for blob_num, b3d in enumerate(self.b3ds):
            lines_per_color[blob_num % num_markers] += sum(
                len(pairing.indeces) for blob3d in self.b3ds for pairing in blob3d.pairings)
        for line_count in lines_per_color:
            line_location_lists.append(np.zeros([2 * line_count, 3]))  # 2x because we are storing endpoints
        for blob_num, blob3d in enumerate(self.b3ds):
            for pairing in blob3d.pairings:
                for stitch_no, stitch in enumerate(pairing.stitches):
                    lowerpixel = Pixel.get(stitch.lowerpixel)
                    upperpixel = Pixel.get(stitch.upperpixel)
                    blob_color_index = blob_num % num_markers
                    line_location_lists[blob_color_index][offsets[blob_color_index] + (2 * stitch_no)] = [
                        (lowerpixel.x - self.xmin) / self.xdim, (lowerpixel.y - self.ymin) / self.ydim,
                        (pairing.lowerheight - self.zmin)/ (Config.z_compression * self.zdim)]
                    line_location_lists[blob_color_index][offsets[blob_color_index] + (2 * stitch_no) + 1] = [
                        (upperpixel.x - self.xmin) / self.xdim, (upperpixel.y - self.ymin) / self.ydim,
                        (pairing.upperheight - self.zmin) / (Config.z_compression * self.zdim)]
                offsets[blob_color_index] += 2 * len(pairing.stitches)
        print(' DB adding a total of ' + str(len(line_location_lists)) + ' blob3d stitch line groups')
        for list_no, line_location_list in enumerate(line_location_lists):
            stitch_lines = (visuals.Line(method=Config.linemethod))
            stitch_lines.set_data(pos=line_location_list, connect='segments', color=colors[list_no % len(colors)])
            self.add_stitch(stitch_lines, 'blob3d')

    def add_neutral_stitches(self):
        line_end_points = 0
        line_index = 0
        for blob_num, blob3d in enumerate(self.b3ds):
            for stitch in blob3d.pairings:
                line_end_points += (2 * len(stitch.indeces))  # 2 as each line has 2 endpoints
        line_locations = np.zeros([line_end_points, 3])
        for blob3d in self.b3ds:
            for pairing in blob3d.pairings:
                for stitch in pairing.stitches:
                    lowerpixel = Pixel.get(stitch.lowerpixel)
                    upperpixel = Pixel.get(stitch.upperpixel)
                    # line_locations[line_index] = [lowerpixel.x / self.xdim, lowerpixel.y / self.ydim,
                    #                               pairing.lowerheight / (Config.z_compression * self.zdim)]
                    # line_locations[line_index + 1] = [upperpixel.x / self.xdim, upperpixel.y / self.ydim,
                    #                                   pairing.upperheight / (Config.z_compression * self.zdim)]
                    line_locations[line_index] = [(lowerpixel.x - self.xmin) / self.xdim,
                                                  (lowerpixel.y - self.ymin) / self.ydim,
                                                  (pairing.lowerheight - self.zmin) / (
                                                      Config.z_compression * self.zdim)]
                    line_locations[line_index + 1] = [(upperpixel.x - self.xmin) / self.xdim,
                                                      (upperpixel.y - self.ymin) / self.ydim,
                                                      (pairing.upperheight - self.zmin) / (
                                                          Config.z_compression * self.zdim)]
                    line_index += 2
        stitch_lines = visuals.Line(method=Config.linemethod)
        stitch_lines.set_data(pos=line_locations, connect='segments')
        self.add_stitch(stitch_lines, 'neutral')

    def add_parent_lines(self, explode=True):
        lineendpoints = 0
        for num, b2d in enumerate(self.b2ds):
            lineendpoints += (2 * len(b2d.children))
        if lineendpoints:
            line_index = 0
            line_locations = np.zeros([lineendpoints, 3])
            for b2d in self.b2ds:
                for child in b2d.children:
                    child = Blob2d.get(child)
                    line_locations[line_index] = [(b2d.avgx - self.xmin) / self.xdim, (b2d.avgy - self.ymin) / self.ydim,
                                                  (getBloomedHeight(b2d, explode, self.zdim) - self.zmin) / (
                                                      Config.z_compression * self.zdim)]
                    line_locations[line_index + 1] = [(child.avgx - self.xmin) / self.xdim,
                                                      (child.avgy - self.ymin) / self.ydim,
                                                      (getBloomedHeight(child, explode, self.zdim) - self.zmin) / (
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
        assert type(bloblist) is list
        if len(bloblist) == 0:
            raise Exception("Added an empty list to canvas; requires a list of blob2d or blob3ds")
        if all(type(blob) is Blob3d for blob in bloblist):
            # Given a list of blob3ds
            self.b3ds = bloblist
            self.b2ds = [Blob2d.get(blob2d) for blob3d in bloblist for blob2d in blob3d.blob2ds]
            self.set_canvas_bounds(b2ds_not_b3ds=False)
            self.plot_call = 'PlotBlob3ds'
        elif all(type(blob) is Blob2d for blob in bloblist):
            # Given a list of blob2ds
            self.b2ds = bloblist
            self.b3ds = get_blob3ds_from_blob2ds(bloblist, ids=False)
            self.set_canvas_bounds(b2ds_not_b3ds=True)
            self.plot_call = 'PlotBlob2ds'
        else:
            warn('Given a list not of blob2ds or blob3ds entirely!!!')
            print(bloblist)
            debug()
            exit(1)

    def set_canvas_bounds(self, b2ds_not_b3ds):
        """
        Sets (x,y,z) (min, max, dim) using either canvas.b2ds or canvas.b3ds
        :param b2ds_not_b3ds: True if using blob2ds, False if using blob3ds
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
                for b2d in blob3d.blob2ds:
                    b2d = Blob2d.get(b2d)
                    xmin = min(xmin, b2d.minx)
                    xmax = max(xmax, b2d.maxx)
                    ymin = min(ymin, b2d.miny)
                    ymax = max(ymax, b2d.maxy)
                    zmin = min(zmin, b2d.height)
                    zmax = max(zmax, b2d.height)

                # Note changed the below temporarily because of issues between the ranges of b3ds and their constituent b2ds
                # xmin = min(xmin, blob3d.minx)
                # xmax = max(xmax, blob3d.maxx)
                # ymin = min(ymin, blob3d.miny)
                # ymax = max(ymax, blob3d.maxy)
                # zmin = min(zmin, blob3d.lowslideheight)
                # zmax = max(zmax, blob3d.highslideheight)

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


def plot(blob3ds_or_blob2ds, coloring=None, line_coloring=None, canvas_size=(800, 800), ids=False, stitches=False,
         buffering=True, parentlines=False, explode=False, maxcolors=-1, show_debug_colors=False):
    global colors
    assert coloring in [None, 'blob2d', 'depth', 'blob3d', 'bead', 'simple', 'neutral']
    assert line_coloring in [None, 'blob3d', 'neutral']
    if coloring is None:  # Revert to default color
        coloring = 'blob3d'
    else:
        coloring = coloring.lower()
    if line_coloring is None:  # Revert to default color, use stitches = False to avoid drawing any lines
        line_coloring = 'neutral'
    else:
        line_coloring = line_coloring.lower()
    if 0 < maxcolors < len(colors):
        colors = colors[:maxcolors]
    if len(blob3ds_or_blob2ds) == 0:
        raise Exception("Tried to plot a list of zero blob2ds / blob3ds")
    canvas = Canvas(canvas_size, coloring=coloring, buffering=buffering, debug_colors=show_debug_colors)
    canvas.set_blobs(blob3ds_or_blob2ds)

    if coloring == 'simple' or canvas.buffering:
        canvas.add_simple_beads() # NOTE this also adds simple stitches

    if coloring == 'blob2d' or canvas.buffering:
        canvas.add_blob2d_markers(explode=explode)

    if coloring == 'blob3d' or canvas.buffering:
        canvas.add_blob3d_markers()

    if coloring == 'bead' or canvas.buffering:
        canvas.add_bead_markers()

    if coloring == 'depth' or canvas.buffering: # Note same as b2d_depth
        canvas.add_depth_markers()

    if coloring == 'neutral' or canvas.buffering:
        canvas.add_neutral_markers()

    if coloring == 'merged' or (canvas.debug_colors and canvas.buffering):
        canvas.add_merged_markers()
        canvas.add_merged_parent_markers()

    if stitches or canvas.buffering:
        if line_coloring == 'blob3d' or canvas.buffering:  # TODO need to change this so that stitchlines of the same color are the same object
            # if Config.test_instead_of_data:
            #     canvas.add_blob3d_stitches()
            # else:
            print('Skipping adding blob3d stitches as it overloads ram (for now)')  # Took 32Gb of ram, 2.75GB of video memory for swellshark dataset - ouch!
        if line_coloring == 'neutral' or canvas.buffering:
            canvas.add_neutral_stitches()

        if parentlines or canvas.buffering:
            canvas.add_parent_lines(explode=explode)

    if ids:
        print(
            "\nWARNING adding ids for every blob2d, this could overload if not a small dataset, so skipping if not a test_set!!")
        if Config.test_instead_of_data:
            for b2d_num, b2d in enumerate(canvas.b2ds):
                midpoints = [(b2d.avgx - canvas.xmin) / canvas.xdim, (b2d.avgy - canvas.ymin) / canvas.ydim, (
                    (getBloomedHeight(b2d, explode, canvas.zdim) + .25 - canvas.zmin) / (
                        Config.z_compression * canvas.zdim))]
                textStr = str(b2d.id)
                color = colors[b2d.id % len(colors)]
                canvas.view.add(visuals.Text(textStr, pos=midpoints, color=color, font_size=8, bold=False))

    canvas.setup_markers()
    canvas.setup_stitches()
    vispy.app.run()


def plotBlob2d(b2d, canvas_size=(1080, 1080)):
    # Automatically labels all pixels
    global colors
    canvas = Canvas(canvas_size, coloring='blob2d')
    canvas.plot_call = 'PlotBlob2ds'
    canvas.set_blobs([b2d])
    canvas.add_blob2d_markers([b2d], explode=False)
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


def plot_hist_xyz(b3ds, xname='Avgx', yname='Avgy', zname='Avgz', type='Base_B3ds', num_1d_bins=75,
                  num_2d_bins=150):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)

    xs = list(b3d.avgx for b3d in b3ds)
    ys = list(b3d.avgy for b3d in b3ds)
    zs = list(b3d.avgz for b3d in b3ds)

    # 1D Histograms
    n1, bins1, patches1 = ax1.hist(xs, bins=num_1d_bins)
    ax1.set_xlabel(xname + " of " + type)
    ax1.set_ylabel("Number of " + type)
    ax1.set_title(type + " by " + xname)

    n2, bins2, patches2 = ax2.hist(ys, bins=num_1d_bins)
    ax2.set_xlabel(yname + " of " + type)
    ax2.set_ylabel("Number of " + type)
    ax2.set_title(type + " by " + yname)

    n3, bins3, patches3 = ax3.hist(zs, bins=num_1d_bins)
    ax3.set_xlabel(zname + " of " + type)
    ax3.set_ylabel("Number of " + type)
    ax3.set_title(type + " by " + zname)

    # 2D Histograms
    axres4 = ax4.hist2d(xs, ys, bins=num_2d_bins)
    ax4.set_xlabel(xname + " of " + type)
    ax4.set_ylabel(yname + " of " + type)
    ax4.set_title(type + " " + xname + " by " + yname)
    cbar4 = fig.colorbar(axres4[3], ax=ax4, orientation='vertical')

    axres5 = ax5.hist2d(xs, zs, bins=num_2d_bins)
    ax5.set_xlabel(xname + " of " + type)
    ax5.set_ylabel(zname + " of " + type)
    ax5.set_title(type + " " + yname + " by " + zname)
    cbar5 = fig.colorbar(axres5[3], ax=ax5, orientation='vertical')

    axres6 = ax6.hist2d(ys, zs, bins=num_2d_bins)
    ax6.set_xlabel(yname + " of " + type)
    ax6.set_ylabel(zname + " of " + type)
    ax6.set_title(type + " " + yname + " by " + zname)
    cbar6 = fig.colorbar(axres6[3], ax=ax6, orientation='vertical')

    buf = list(len(b3d.blob2ds) for b3d in b3ds)
    n7, bins7, patches7 = ax7.hist(buf, bins=min(num_1d_bins, max(buf)))
    ax7.set_xlabel("Number of b2ds per " + type)
    ax7.set_ylabel("Number of " + type)
    ax7.set_title(type + " by number of b2ds")

    n8, bins8, patches8 = ax8.hist(list(len(b3d.get_pixels()) for b3d in b3ds), bins=num_1d_bins)
    ax8.set_xlabel("Number of pixels per " + type)
    ax8.set_ylabel("Number of " + type)
    ax8.set_title(type + " by number of pixels")

    n9, bins9, patches9 = ax9.hist(list(b3d.get_edge_pixel_count() for b3d in b3ds), bins=num_1d_bins)
    ax9.set_xlabel("Number of edge_pixels per " + type)
    ax9.set_ylabel("Number of " + type)
    ax9.set_title(type + " by number of edge_pixels")

    figManager = plt.get_current_fig_manager()  # http://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    figManager.window.showMaximized()

    window = plt.gcf()  # http://stackoverflow.com/questions/5812960/change-figure-window-title-in-pylab
    window.canvas.set_window_title(type + '(' + str(len(xs)) + ') by ' + xname + ' ' + yname + ' & ' + zname)
    fig.tight_layout()
    plt.show()


def plot_corr(b3ds, type='Base_B3ds'):

    # Now building correlation matrix
    attr_names = ['avgx', 'avgy', 'avgz', 'minx', 'miny', 'minz', 'maxx', 'maxy', 'maxz', 'recur_depth',
                  '# child b3ds', '# b2ds', '# pixels', '# edge_pixels', '# pairings', 'isSingle', 'isBead']
    num_attr = len(attr_names)
    mat = [[] for i in range(num_attr)]
    for index, b3d in enumerate(b3ds):
        mat[0].append(b3d.avgx)
        mat[1].append(b3d.avgy)
        mat[2].append(b3d.avgz)
        mat[3].append(b3d.minx)
        mat[4].append(b3d.miny)
        mat[5].append(b3d.lowslideheight)
        mat[6].append(b3d.maxx)
        mat[7].append(b3d.maxy)
        mat[8].append(b3d.highslideheight)
        mat[9].append(b3d.recursive_depth)
        mat[10].append(len(b3d.children))
        mat[11].append(len(b3d.blob2ds))
        mat[12].append(len(b3d.get_pixels()))
        mat[13].append(b3d.get_edge_pixel_count())
        mat[14].append(len(b3d.pairings))
        mat[15].append(int(b3d.isSingular))
        mat[16].append(int(b3d.isBead))

    corr = np.corrcoef(mat)
    adj_corr = np.copy(corr)

    for r in range(num_attr):
        for c in range(num_attr - r):
            adj_corr[r][num_attr - c - 1] = 0.

    desired_width = 500
    pd.set_option('display.width', desired_width)

    index_names = ['B3d_' + str(i) for i in range(len(b3ds))]

    df = pd.DataFrame(data=np.transpose(mat), index=index_names, columns=attr_names)
    print(df.describe())

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.matshow(adj_corr, cmap=plt.cm.seismic, interpolation='none', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks([i for i in range(num_attr)])
    ax.set_yticks([i for i in range(num_attr)])
    ax.set_xticklabels(attr_names)
    ax.set_yticklabels(attr_names)
    ax.grid()
    figManager = plt.get_current_fig_manager()  # http://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    figManager.window.showMaximized()
    window = plt.gcf()  # http://stackoverflow.com/questions/5812960/change-figure-window-title-in-pylab
    window.canvas.set_window_title("Correlation matrix of " + type)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

