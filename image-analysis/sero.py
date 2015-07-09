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


def plotMatrixBinary(mat):
    plt.spy(mat, markersize=1, aspect='auto', origin='lower', marker='x')
    cmap2 = cm.BrBG
    plt.set_cmap(cmap2)
    plt.tight_layout()
    plt.show()
def plotMatrixColor(mat):
    plt.matshow(slice, vmin=10, vmax=100)
    # plt.tight_layout()
    plt.show()
def plotMatrixPair(m1, m2):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    #cmap2 = cm.BrBG
    matplotlib.style.use('ggplot')
    #plt.set_cmap(cmap2)
    plt.matshow(m1, vmin=10, vmax=100, markersize=1, aspect='auto', origin='lower')
    plt.matshow(m2, vmin=10, vmax=100, markersize=1, aspect='auto', origin='lower')
    plt.title('Before after filter')
    plt.tight_layout()
    plt.show()

def runShell():
    vars = globals()
    vars.update(locals())
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(vars)
    shell.interact()

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
for s in range(zdim):
    slice = imarray[:, :, s] # Creates array of shape (1600,)
    # slice = np.dsplit(imarray, 1)# Creates list of shape (1,1,1600)
    slices.append(slice)
    # plotMatrix(slice)
    # plotMatrixColor(slice)
# Now slices[0] holds a (1600,1600) nparray of the layer we want.
# Lets convert this one slice back to an 'Image', so that we can use PIL
#   See: http://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
raw_slice = slices[0]
normalized_slice = raw_slice / np.linalg.norm(raw_slice)
# plotMatrixBinary(raw_slice)
# plotMatrixColor(normalized_slice)


im = Image.fromarray(np.uint8(cm.jet(slices[0])*255))
out = im.filter(ImageFilter.MaxFilter)
im.show()
out.show()
# array_im = np.array(im)
# array_out = np.array(out)

runShell()
# plotMatrixPair(array_im, array_out)

# plt.imsave for saving

