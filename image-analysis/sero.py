__author__ = 'gio'
import matplotlib.pyplot as plt
from PIL import Image
import numpy
import matplotlib.pylab as plt
import matplotlib.cm as cm


def plotMatrixBinary(mat):
    plt.spy(mat, markersize=1, aspect='auto', origin='lower', marker='x')
    cmap2 = cm.BrBG
    plt.set_cmap(cmap2)
    plt.tight_layout()
    plt.show()
def plotMatrixColor(mat)
    plt.tight_layout()
    plt.matshow(slice, vmin=10, vmax=100)
    plt.show()


im = Image.open('..\\data\\Swellshark_Adult_012615_TEL1s1_DorsalPallium_5-HT_CollagenIV_60X_C003Z001.tif')
#im.show()
imarray = numpy.array(im)
# numpy.set_printoptions(threshold='nan')
print(imarray)
print(imarray.shape) # (1600, 1600, 3) => Means that there is one for each channel!!
                     # Can then store results etc into a 4th channel, and in theory save that back into the tiff
slices = []
(xdim, ydim, zdim) = imarray.shape
print('The are ' + str(zdim) + ' channels')
for s in range(zdim):
    slice = imarray[:, :, s]
    print(slice.shape)
    slices.append(slices)
    # plotMatrix(slice)
    plt.matshow(slice)
    plt.show()


# plt.imsave for saving

