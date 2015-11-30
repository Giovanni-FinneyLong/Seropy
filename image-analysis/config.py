__author__ = 'gio'

#RENAME THIS FILE 'myconfig.py'; this file is a template

  ### Switches ###
debug_blob_ids = False
debug_pixel_ops = False
debug_set_merge = False
remap_ids_by_group_size = True
test_instead_of_data = False
dePickle = False
OpenGLLinesInsteadOfAgg = True
mayPlot = False # Used to control the importing of visualization packages; vispy doesn't run on arm :(
remoteExecution = False

  ### Variables ###
max_pixels_to_stitch = 150 # Max threshold for the number of pixels to use for Pairing from any Blob2d
debug_pixel_ops_y_depth = 500
overscan_coefficient = 1.1 # A number >= 1, which is the scaling for selecting edge pixels
min_val_threshold = 250
    # Recommended = 250
max_val_step = 5 # The maximum amount that two neighboring pixels can differ in val and be grouped by blob_id
    # Recommended = 5
minimal_nonzero_neighbors = 2 # The minimal amount of nzn a pixel must have to avoid being filtered; 0 = no filter
    # Recommended = 2
z_compression = 1
    # A value in the range of (0, oo), which multiplies the spacing between the slides.
    # A value of 2, double the separation between slides when plotted, .5 halves it
slide_portion = 1
    # (0,1]: The proportion of each slide to operate over, used to speed up test processing
hard_max_pixel_value = 255
max_stitch_cost = 90 # The max cost a stitch can be before it is ignored
max_distance = 7 # The max distance that two pixels can be apart and still be stitched together.
                 # If this threshold is breached, edge_pixels will not have any line to them, including a substitute
min_pixels_to_be_independent = 9 # The minimum number of pixels in a Blob2d of a Subblob3d to be considered as possibly another b3d
                        # This is currently experimental
min_pixels_to_split = 9 # The minimum number of pixels in a Blob2d of a Subblob3d
                        # This is currently experimental
#NOTE these would be good for a sliding scale?
assert 0 < z_compression <= 1
assert 0 < slide_portion <= 1

if remoteExecution: # Set up for execution on TS140
  ### Folders ###
  FIGURES_DIR = 'C:/Users/gio/Documents/Programming/serotonin/image-analysis/generated_figures/'
  DATA_DIR =  'H:/Dropbox/Serotonin/data/'
  TEST_DIR = 'H:/Dropbox/Serotonin/data/Tests/'
  IMAGEMAGICK_CONVERT_EXEC = 'C:/Program Files/ImageMagick-6.9.1-Q8/'
  pickledir = 'C:/Users/gio/Documents/Programming/serotonin/image-analysis/pickles/' # Can be relative

else:
  ### Folders ###
  FIGURES_DIR = ''
  DATA_DIR =  ''
  TEST_DIR = ''
  IMAGEMAGICK_CONVERT_EXEC = ''
  pickledir = '' # Can be relative




if OpenGLLinesInsteadOfAgg:
  linemethod = 'gl'
else:
  linemethod = 'agg'