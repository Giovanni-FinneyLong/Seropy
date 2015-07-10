__author__ = 'gio'
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.cm as cm
import readline
import code
import rlcompleter
from sklearn.preprocessing import normalize
from PIL import ImageFilter
# from collections import OrderedDict

def plotHist(list, numBins):
    'Take a 1dimensional matrix or a list'
    plt.hist(list, bins=numBins)
    plt.show()

def plotMatrixBinary(mat):
    plt.spy(mat, markersize=1, aspect='auto', origin='lower')
    plt.show()

def plotMatrixColor(mat):
    plt.imshow(mat, vmin=55, vmax=99) # 0,99 are min,max defaults
    plt.colorbar()
    plt.show()

def plotMatrixPair(m1, m2):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    cmap = cm.jet
    # matplotlib.style.use('ggplot')
    plt.set_cmap(cmap)
    ax1.spy(m1, markersize=1, aspect='auto', origin='lower')
    ax2.spy(m2, markersize=1, aspect='auto', origin='lower')
    plt.show()

def plotMatrixTrio(m1, m2, m3):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, sharex=True)
    cmap = cm.jet
    matplotlib.style.use('ggplot')
    plt.set_cmap(cmap)
    ax1.spy(m1, markersize=1, aspect='auto', origin='lower')
    ax2.spy(m2, markersize=1, aspect='auto', origin='lower')
    ax3.spy(m3, markersize=1, aspect='auto', origin='lower')
    plt.show()

def runShell():
    vars = globals()
    vars.update(locals())
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(vars)
    shell.interact()

class Pixel:
    'This class is being used to hold the coordinates, base info and derived info of a single pixle of a single image\'s layer'

    def __init__(self, xin, yin, value):
        self.x = xin # The x coordinate
        self.y = yin # The y coordinate
        self.val = value
    def setNeighborValues(self, non_zero_neighbors, neighbor_sum):
        self.nz_neighbors = non_zero_neighbors # The number out of the 8 surrounding pixels that are non-zero
        self.neighbor_sum = neighbor_sum # The sum of the surrounding 8 pixels




imagein = Image.open('..\\data\\Swellshark_Adult_012615_TEL1s1_DorsalPallium_5-HT_CollagenIV_60X_C003Z001.tif')
#im.show()
imarray = np.array(imagein)
# numpy.set_printoptions(threshold='nan')
print(imarray.shape) # (1600, 1600, 3) => Means that there is one for each channel!!
                     # Can then store results etc into a 4th channel, and in theory save that back into the tiff
slices = []
(xdim, ydim, zdim) = imarray.shape
# np.set_printoptions(threshold=np.inf)
print('The are ' + str(zdim) + ' channels')
image_channels = imagein.split()
slices = []
norm_slices = []
non_zero_count = 0
pixels = []
sum_pixels = 0
sorted_pixels = []

for s in range(len(image_channels)): # Better to split image and use splits for arrays than to split an array
    buf = np.array(image_channels[s])
    slices.append(buf)
    norm_slices.append(255 * buf / np.linalg.norm(buf))
    if (np.amax(slices[s]) == 0):
        print('Slice #' + str(s) + ' is an empty slice')

plotMatrixColor(slices[0])

im = Image.fromarray(np.uint8(cm.jet(slices[0])*255))
out = im.filter(ImageFilter.MaxFilter)

# Can use im.load as needed to access Image pixels
# Opting to use numpy, I expect this will be faster to operate and easier to manipulate

(xdim, ydim) = slices[0].shape
pixels_for_hist = []
for pcol in range(xdim):
    for pix in range(ydim):
        pixel_value = slices[0][pcol][pix]
        # print('Pixel #' + str(pcol * xdim + pix) + ' = ' + str(pixel_value))
        if(pixel_value != 0): # Can use alternate min threshold and <=
            pixels.append((pixel_value, pcol, pix))
            pixels_for_hist.append(pixel_value) #Hack due to current lack of a good way to slice tuple list quickly
            sum_pixels += pixel_value
print('The are ' + str(len(pixels)) + ' non-zero pixels from the original ' + str(xdim * ydim) + ' pixels')
# Now to sort by 3rd element/2nd index = pixel value
sorted_pixels = sorted(pixels, key=lambda tuple: tuple[0])
# plotHist(pixels_for_hist, 500)
for (p_num, pixel) in enumerate(sorted_pixels):
    print(str(p_num) + ': ' + str(pixel))

# im.show()
# out.show()
# plotMatrixTrio(slices[0], slices[1], slices[2])

# runShell()

# plt.imsave for saving

