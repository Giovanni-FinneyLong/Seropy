__author__ = 'gio'
# RENAME THIS FILE 'myconfig.py'; this file is a template
class Config:
    """
    All important global variables that are present at init are atrributes of this class
    Incl: Switches, Variables, Data-Folder Locations, Safety Checks
    """

    '''
      ____               _   _            _
     / ___|  __      __ (_) | |_    ___  | |__     ___   ___
     \___ \  \ \ /\ / / | | | __|  / __| | '_ \   / _ \ / __|
      ___) |  \ V  V /  | | | |_  | (__  | | | | |  __/ \__ \
     |____/    \_/\_/   |_|  \__|  \___| |_| |_|  \___| |___/
     '''  # All switches are either True or False
    mayPlot = True  # Used to control the importing of visualization packages; vispy doesn't run on arm :(
    test_instead_of_data = True  # Use the data at Test_Dir instead of Data_dir
    dePickle = False  # Load a preprocessed dataset. Run with False at least the first time
    process_internals = True  # Default True, Do blooming, set possible partners for the generated b2ds, then create b3ds from them
    base_b3ds_with_stitching = True  # Default True, This needs to be true to get accurate results. If speed is much more valueable that accuracy, make False
    # NOTE can allow this to control creation of b3ds, or allow a quick create method for b3ds (noting no stitching and much less accuracy)
    stitch_bloomed_b2ds = False  # Default False, this enables stitching between generated internal Blob2ds. This greatly increases processing time!
    do_logging = True # Default True, If true, logs all output to logs folder
    nervous_logging = True  # Default, If true, the logger saves after each output. Slightly increases processing time
    log_everything = True  # Enables the logging of debugging output, which may greatly increase log size

    OpenGLLinesInsteadOfAgg = True
    disable_warnings = False

    debug_blob_ids = False  # Enables extra output for debugging purposes
    debug_pixel_ops = False  # Enables extra output for debugging purposes
    debug_set_merge = False  # Enables extra output for debugging purposes
    debug_b3d_merge = False  # Enables extra output for debugging purposes
    debug_stitches = False  # Enables extra output for debugging purposes
    debug_bead_tagging = False  # Enables extra output for debugging purposes
    debug_blooming = False  # Enables extra output for debugging purposes
    debug_partners = False  # Enables extra output for debugging purposes

    '''
     __     __            _         _      _
     \ \   / /__ _  _ __ (_)  __ _ | |__  | |  ___  ___
      \ \ / // _` || '__|| | / _` || '_ \ | | / _ \/ __|
       \ V /| (_| || |   | || (_| || |_) || ||  __/\__ \
        \_/  \__,_||_|   |_| \__,_||_.__/ |_| \___||___/
    '''  # All switches are either True or False
    hard_max_pixel_value = 255 # The absolute maximal value that any pixel can have
    # Using 255 for 8-bit colors, there shouldn't be much need to adjust this

    recursion_limit = 5000 # The max recursive depth python can make calls to
    # This is set high to allow file saving/loading using pickle
    # Recommended 5000

    overscan_coefficient = 1.1  # A number [1,2], which is the scaling of the bounding area for selecting edge pixels
    # A value of 2 means that each blob2d will look in each of the 4 directions 2x as far
    # As the distance between the blob2d's midpoint and it's farthest pixel in that direction
    # The bounds are established to prevent unreasonable values
    # Recommended = 1.1

    min_val_threshold = 250  # The minimal value (out of 255) that a pixel must be to be considered 'on'
    # A higher value will have a tendency to preserve the general shape of a blob2d
    # A lower value will have a tendency to preserve the fringes of each blob2d, and also making them slightly larger
    # Recommended = 250

    max_val_step = 5  # The maximum amount that two neighboring pixels can differ in val and be grouped by blob_id
    # Recommended = 5, assuming using 8-bit colors (Values 0-255)

    minimal_nonzero_neighbors = 2  # The minimal amount of non-zero adjacent pixels a pixel must have to avoid being filtered
    # 0 = no filter
    # Recommended = 2

    z_compression = 1 # A value in the range of (0, 100), which multiplies the spacing between the slides.
    # A value of 2, halves the separation between slides when plotted, .5 doubles it
    # The upper bound is to prevent unreasonable values

    slide_portion = 1 # A value in the range of (0,1]
    # The proportion of each slide to operate over, used to speed up test processing
    # A value of .5 would mean scanning 1/2 the slide in the x & y directions, resulting in 1/4 the area

    image_channel_to_use = 0  # The slide channel to use (as tifs has multiple channels / subimages). Start at 0

    '''
      _____       _      _
     |  ___|___  | |  __| |  ___  _ __  ___
     | |_  / _ \ | | / _` | / _ \| '__|/ __|
     |  _|| (_) || || (_| ||  __/| |   \__ \
     |_|   \___/ |_| \__,_| \___||_|   |___/
    '''  # All folders are string of relative or complete paths (each ending with '\\' or '/'), or file patterns
    FIGURES_DIR = ''  # Directory where generated images are stored
    DATA_DIR = ''  # Directory where the input datasets are stored
    DATA_FILE_PATTERN = 'Swell*.tif'  # File name pattern of images used for input when test_instead_of_data is False
    # NOTE: This is currently configured for the swellshark dataset
    TEST_DIR = ''  # Directory where the 'test' files are stored
    TEST_FILE_PATTERN = '*.png'  # File type of images used for input when test_instead_of_data is True
    IMAGEMAGICK_CONVERT_EXEC = ''  # Full path to Image Magick .exe file (for generating gifs)
    PICKLEDIR = ''  # Folder to store processed datasets (as .pickle files)
    PICKLE_FILE_PREFIX = 'Swellshark_Adult_012615'  # Note that '.pickle' is appended
    #  Note been using 'Swellshark_Adult_012615' for Swellshark, 'C57BL6_Adult_CerebralCortex' for C57BL
    #  Note been using 'All_test_pre_b3d_tree' when running tests

    '''
      _____                           _                          _          _
     | ____|__  __ _ __    ___  _ __ (_) _ __ ___    ___  _ __  | |_  __ _ | |
     |  _|  \ \/ /| '_ \  / _ \| '__|| || '_ ` _ \  / _ \| '_ \ | __|/ _` || |
     | |___  >  < | |_) ||  __/| |   | || | | | | ||  __/| | | || |_| (_| || |
     |_____|/_/\_\| .__/  \___||_|   |_||_| |_| |_| \___||_| |_| \__|\__,_||_|
     __     __    |_|     _         _      _
     \ \   / /__ _  _ __ (_)  __ _ | |__  | |  ___  ___
      \ \ / // _` || '__|| | / _` || '_ \ | | / _ \/ __|
       \ V /| (_| || |   | || (_| || |_) || ||  __/\__ \
        \_/  \__,_||_|   |_| \__,_||_.__/ |_| \___||___/

    '''  # These variables are somewhat new, it is recommended that you do not modify them
    max_pixels_to_stitch = 50  # The max amount of pixels acceptable in EACH pair of slides to be stitched.
    # Increasing this can greatly increase the amount of time required to stitch large blobs
    # This is because the optimized Munkres algorithm is O(n^3)
    # Recommended 50-150, for experimental use lower is generally better
    # Semi-Experimental (trying to find a good range), may want to adjust to be a proportion of edge-pixels

    max_stitch_cost = 90  # The max cost a stitch can be before it is ignored
    # Experimental

    max_distance = 7  # The max distance that two pixels can be apart and still be stitched together.
    # If this threshold is breached, edge_pixels will not have any line to them, including a substitute
    # Experimental

    max_depth = 5 # Max recursive depth when blooming Note:(allows a total of n+2 depths, including the original one)
    # Experimental

    minimal_pixel_overlap_to_be_possible_partners = .10  # The minimal portion of area that one of a pair of blob2ds must overlap with the other to be partners
    # Experimental
    max_pixels_to_be_a_bead = 250  # Can be modified real time during visualization
    max_subbeads_to_be_a_bead = 4  # Can be modified real time during visualization
    child_bead_difference = 2  # Can be modified real time during visualization

    '''
      ____           __        _
     / ___|   __ _  / _|  ___ | |_  _   _
     \___ \  / _` || |_  / _ \| __|| | | |
      ___) || (_| ||  _||  __/| |_ | |_| |
     |____/  \__,_||_|   \___| \__| \__, |
                                    |___/
    '''  # Make sure configuration variables are within the correct ranges and well formatted
    assert 0 < z_compression <= 10
    assert 0 < slide_portion <= 1
    assert 1 <= overscan_coefficient <= 2
    assert 0 < min_val_threshold <= hard_max_pixel_value
    assert 1 <= max_pixels_to_stitch
    # Compare GL and Agg here: http://vispy.readthedocs.org/en/stable/examples/basics/visuals/line.html
    if OpenGLLinesInsteadOfAgg:
      linemethod = 'gl'
    else:
      linemethod = 'agg'
    # Checking folders terminate correctly with '/' or '\\'
    assert FIGURES_DIR[-1] in ['/', '\\']
    assert DATA_DIR[-1] in ['/', '\\']
    assert TEST_DIR[-1] in ['/', '\\']
    assert PICKLEDIR[-1] in ['/', '\\']
