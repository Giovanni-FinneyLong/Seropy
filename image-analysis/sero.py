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
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
from numpy import zeros
import math
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import matplotlib.colors as colortools



from mpl_toolkits.mplot3d import Axes3D
import pickle # Note uses cPickle automatically ONLY IF python 3

from sklearn.preprocessing import normalize
from PIL import ImageFilter
from collections import OrderedDict

def plotHist(list, numBins):
    'Take a 1dimensional matrix or a list'
    plt.hist(list, bins=numBins)
    plt.show()

def plotMatrixBinary(mat):
    plt.spy(mat, markersize=1, aspect='auto', origin='lower')
    plt.show()

def plotMatrixColor(mat):
    plotMatrixColor(mat, 0, 99)

def plotMatrixColor(mat, min_thresh, max_thresh):
    plt.imshow(mat, vmin=min_thresh, vmax=max_thresh) # 0,99 are min,max defaults
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

def findBestClusterCount(min, max, step):
    print('Attempting to find optimal number of clusters, range:(' + str(min) + ', ' + str(max))
    kVals = [] # The number of clusters
    distortionVSclusters = [] # The distortion per cluster
    for z in range(math.floor((max - min) / step)):
        num_clusters = (z * step) + min
        if(num_clusters == 0):
            num_clusters = 1
        print('Trying with ' + str(num_clusters) + ' clusters')
        (bookC, distortionC)  = kmeans(max_pixel_array_floats, num_clusters)
        # (centLabels, centroids) = vq(max_pixel_array_floats, bookC
        kVals.append(num_clusters)
        distortionVSclusters.append(distortionC)
    plt.plot(kVals, distortionVSclusters, marker='x')
    plt.grid(True)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Average cluster distortion')
    plt.title('Elbow method for K-Means Clustering on Pixels\nManually Record the desired K value\n')
    plt.show()

def KMeansClusterIntoLists(array_in, num_clusters):
    'Take an array of floats and returns a list of lists, each of which contains all the pixels of a cluster'
    cluster_lists = [[] for i in range(num_clusters)]
    # for c in range(num_clusters):
    #     cluster_arrays.append(zeros([xdim, ydim])) # (r,c)
    (bookC, distortionC)  = kmeans(array_in, num_clusters)
    (centLabels, centroids) = vq(array_in, bookC)
    for pixlabel in range(len(centLabels)):
        cluster = centLabels[pixlabel]
        # pix = max_pixel_array_floats[pixlabel]
        # cluster_arrays[cluster][pix[1]][pix[2]] = pix[0]
        cluster_lists[cluster].append(array_in[pixlabel])
    return cluster_lists

def MeanShiftCluster(array_in):
    'Does mean shift clustering on an array of max_points, does NOT take lists'
    #Trying another type of clustering
    #Largely copied from: http://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html
    bandwidth = estimate_bandwidth(array_in, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(array_in)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    #PLOTTING THE RESULT
    plt.figure(2)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        avoid_output = plt.plot(max_pixel_array_floats[my_members, 1], max_pixel_array_floats[my_members, 2], col + '.')
        avoid_output2 = plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def AffinityPropagationCluster(array_in):
    af = AffinityPropagation(preference=-50).fit(array_in)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(array_in, labels, metric='sqeuclidean'))
    #Plot results
    # plt.close('all')
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = array_in[cluster_centers_indices[k]]
        plt.plot(array_in[class_members, 1], array_in[class_members, 2], col + '.')
        plt.plot(cluster_center[1], cluster_center[2], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in array_in[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    # TODO NOTE that this process takes 9GB ram for 1M points, and that the demo was only on 300..
    # TODO Took 15 Minutes before estimating the # of clusters, unloading 5GB of RAM,  then encountering an invalid value by getting the mean of an empty slice
    #   IE, this method isn't scalable to use on an entire image of 2.56M pixels..
    #   Perhaps can use K-Means to reduce the points, and then do secondary clustering with above/else?
    #   If the k-means was done loosely, it would have the effect of grouping pixels into neighborhoods

def PlotListofClusterArraysColor(list_of_arrays):
    'Takes a list of 2D arrays, each of which is a populated cluster, and plots then in 3d.'
    # Try 3D plot
    colors2 = plt.get_cmap('gist_rainbow')
    num_clusters = len(list_of_arrays)
    cNorm  = colortools.Normalize(vmin=0, vmax=num_clusters-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(num_clusters)])
    for (c, color) in zip(range(len(list_of_arrays)), colors):
        (x,y) = list_of_arrays[c].nonzero()
        ax.scatter(x,y, -c, zdir='z', c=scalarMap.to_rgba(c))
    #plt.savefig("3D.png")
    plt.show()





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
# print(imarray.shape) # (1600, 1600, 3) => Means that there is one for each channel!!
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
# im = Image.fromarray(np.uint8(cm.jet(slices[0])*255))
# out = im.filter(ImageFilter.MaxFilter)
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
sorted_pixels = sorted(pixels, key=lambda tuple: tuple[0], reverse=True)
# Lets go further and grab the maximal pixels, which are at the front
endmax = 0
while(sorted_pixels[endmax][0] ==  255):
    endmax += 1
print('There are  ' + str(endmax) + ' maximal pixels')
# Time to pre-process the maximal pixels; try and create groups/clusters

max_pixel_list = sorted_pixels[0:endmax] # Pixels with value 255
max_pixel_array_floats = np.asarray([(float(i[0]), float(i[1]), float(i[2])) for i in max_pixel_list])

# findBestClusterCount(0, 100, 5)

# Now have labels in centLabels for each of the max_pixels
# For fun, going to make an array for each cluster

cluster_count = 20
cluster_lists = KMeansClusterIntoLists(max_pixel_array_floats, cluster_count)
for i in range(len(cluster_lists)):
    print('Index:' + str(i) + ', size:' + str(len(cluster_lists[i]))) # + ' pixels:' + str(cluster_lists[i]))


# MeanShiftCluster(max_pixel_array_floats)
# AffinityPropagationCluster(max_pixel_array_floats):




colors = cycle('bgrcmykbgrcmyk')
cmap=plt.get_cmap('gist_rainbow')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_color_cycle([cm(1.*i/cluster_count) for i in range(cluster_count)]) # Making a new color for each POINT not cluster

#colors = cm.rainbow(np.linspace(0, 1, cluster_count))


# cluster_arrays = [] # Each entry is an array, filled only with the maximal values from the corresponding
# for cluster in range(cluster_count):
#     cluster_arrays.append(zeros([xdim, ydim])) # (r,c)
#     cluster_color = cmap(float(cluster)/cluster_count)
#     cur_color = next(colors)
#     print('Color of cluster #' + str(cluster) + ' is ' + str(cur_color))
#     print('Color of clustermap #' + str(cluster) + ' is ' + str(cluster_color))
#     for pixel in cluster_lists[cluster]:
#         cluster_arrays[cluster][pixel[1]][pixel[2]] = pixel[0]
#         # avoid_output = plt.plot(pixel[1], pixel[2], str(cur_color + '.'))# Works..
#         a = plt.plot(pixel[1], pixel[2], c=cur_color)
# plt.show()


# cluster_array = zeros([zdim, xdim, ydim]) #Note that the format is z,x,y
cluster_arrays = [] # Each entry is an array, filled only with the maximal values from the corresponding
for (cluster, col) in zip(range(cluster_count), colors):
    cluster_arrays.append(zeros([xdim, ydim])) # (r,c)
    for pixel in cluster_lists[cluster]:
        cluster_arrays[cluster][pixel[1]][pixel[2]] = pixel[0]
        # avoid_output = plt.plot(pixel[1], pixel[2], col + '.')
    print('Color of cluster #' + str(cluster) + ' is ' + str(col + '.'))
# plt.show()


# colors2 = cm.rainbow(np.linspace(0, 1, cluster_count))

PlotListofClusterArraysColor(cluster_arrays)






sub_cluster_count = 10
# TODO time to switch to sparse matrices, it seems that there are indeed computational optimizations
# in addition to memory optimizations



# TODO look into DBSCAN from Sklearn as an alternate way to cluster
# TODO sklearn clustering techniques: http://scikit-learn.org/stable/modules/clustering.html
# for i in range(len(cluster_arrays)):
#     plotMatrixColor(cluster_arrays[i],0, 99)

# for (p_num, pixel) in enumerate(sorted_pixels):
#     print(str(p_num) + ': ' + str(pixel))

runShell()

# plt.imsave for saving
#TODO recover and use more of the data from kmeans, like the centroids, which can be used to relate blobs?

# Now to start working on each cluster, and see if we can generate some blobs!!! :D
'''
Rules:
    A max pixel (mp) has value 255
    Around a pixel means the 8 pixels that are touching it in the 2d plane
        123
        4.5
        678
    Any mp next to each other belong together
    Any mp that has no mp around it is removed as noise
    TODO: https://books.google.com/books?id=ROHaCQAAQBAJ&pg=PA287&lpg=PA287&dq=python+group+neighborhoods&source=bl&ots=f7Vuu9CQdg&sig=l6ASHdi27nvqbkyO_VvztpO9bRI&hl=en&sa=X&ei=4COgVbGFD8H1-QGTl7aABQ&ved=0CCUQ6AEwAQ#v=onepage&q=python%20group%20neighborhoods&f=false
        Info on neighborhoods
'''