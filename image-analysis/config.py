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


  ### Variables ###
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
slide_portion = .5
    # (0,1]: The proportion of each slide to operate over, used to speed up test processing

  ### Folders ###
FIGURES_DIR = ''
DATA_DIR =  ''
TEST_DIR = ''
IMAGEMAGICK_CONVERT_EXEC = ''



if OpenGLLinesInsteadOfAgg:
  linemethod = 'gl'
else:
  linemethod = 'agg'