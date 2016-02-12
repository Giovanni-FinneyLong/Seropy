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
     '''
    remap_ids_by_group_size = True
    test_instead_of_data = True
    swell_instead_of_c57bl6 = False # Allows swellshark files to be in the same folder as c57b16
    dePickle = False
    OpenGLLinesInsteadOfAgg = True
    mayPlot = True # Used to control the importing of visualization packages; vispy doesn't run on arm :(
    disable_warnings = False

    process_internals = True # Do blooming, set possible partners for the generated b2ds, then create b3ds from them
    base_b3ds_with_stitching = True # TODO TODO TODO this still needs to be true to get good results, abstractify for filtering b2ds in both cases
    # NOTE can allow this to control creation of b3ds, or allow a quick create method for b3ds (noting no stitching and much less accuracy)
    stitch_bloomed_b2ds = False # Default False

    debug_blob_ids = False
    debug_pixel_ops = False
    debug_set_merge = False
    '''
     __     __            _         _      _
     \ \   / /__ _  _ __ (_)  __ _ | |__  | |  ___  ___
      \ \ / // _` || '__|| | / _` || '_ \ | | / _ \/ __|
       \ V /| (_| || |   | || (_| || |_) || ||  __/\__ \
        \_/  \__,_||_|   |_| \__,_||_.__/ |_| \___||___/
    '''
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

    min_val_threshold = 250 # The minimal value (out of 255) that a pixel must be to be considered 'on'
    # A higher value will have a tendency to preserve the general shape of a blob2d
    # A lower value will have a tendency to preserve the fringes of each blob2d, and also making them slightly larger
    # Recommended = 250

    max_val_step = 5 # The maximum amount that two neighboring pixels can differ in val and be grouped by blob_id
    # Recommended = 5, assuming using 8-bit colors (Values 0-255)

    minimal_nonzero_neighbors = 2 # The minimal amount of non-zero adjacent pixels a pixel must have to avoid being filtered
    # 0 = no filter
    # Recommended = 2

    z_compression = 1 # A value in the range of (0, 10), which multiplies the spacing between the slides.
    # A value of 2, double the separation between slides when plotted, .5 halves it
    # The upper bound is to prevent unreasonable values

    slide_portion = 1 # A value in the range of (0,1]
    # The proportion of each slide to operate over, used to speed up test processing
    # A value of .5 would mean scanning 1/2 the slide in the x & y directions, resulting in 1/4 the area

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

    '''
    # These variables are somewhat new, it is recommended that you do not modify them
    max_pixels_to_stitch = 50 # The max amount of pixels acceptable in EACH pair of slides to be stitched.
    # Increasing this can greatly increase the amount of time required to stitch large blobs
    # This is because the optimized Munkres algorithm is O(n^3)
    # Recommended 50-150, for experimental use lower is generally better
    # Semi-Experimental (trying to find a good range)

    max_stitch_cost = 90 # The max cost a stitch can be before it is ignored
    # Experimental

    max_distance = 7  # The max distance that two pixels can be apart and still be stitched together.
    # If this threshold is breached, edge_pixels will not have any line to them, including a substitute
    # Experimental

    max_depth = 5 # Max recursive depth when blooming Note:(allows a total of n+2 depths, including the original one)
    # Experimental

    minimal_pixel_overlap_to_be_possible_partners = .10  # The minimal portion of area that one of a pair of blob2ds must overlap with the other to be partners
    # Experimental
    max_pixels_to_be_a_bead = 125
    max_subbeads_to_be_a_bead = 4
    child_bead_difference = 2

    '''
      _____       _      _
     |  ___|___  | |  __| |  ___  _ __  ___
     | |_  / _ \ | | / _` | / _ \| '__|/ __|
     |  _|| (_) || || (_| ||  __/| |   \__ \
     |_|   \___/ |_| \__,_| \___||_|   |___/
    '''
    FIGURES_DIR = ''
    DATA_DIR = ''
    TEST_DIR = ''
    IMAGEMAGICK_CONVERT_EXEC = ''
    PICKLEDIR = '' # Can be relative

    '''
      ____           __        _
     / ___|   __ _  / _|  ___ | |_  _   _
     \___ \  / _` || |_  / _ \| __|| | | |
      ___) || (_| ||  _||  __/| |_ | |_| |
     |____/  \__,_||_|   \___| \__| \__, |
                                    |___/
    '''
    assert 0 < z_compression <= 1
    assert 0 < slide_portion <= 1
    assert 1 <= overscan_coefficient <= 2
    assert 0 < min_val_threshold <= hard_max_pixel_value
    assert 1 <= max_pixels_to_stitch
    if OpenGLLinesInsteadOfAgg:
      linemethod = 'gl'
    else:
      linemethod = 'agg'