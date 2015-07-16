__author__ = 'gio'
#import matplotlib.pyplot as plt
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
from matplotlib import animation
import time
import glob

from mpl_toolkits.mplot3d import Axes3D
import pickle # Note uses cPickle automatically ONLY IF python 3

from sklearn.preprocessing import normalize
from PIL import ImageFilter
from collections import OrderedDict

def plotHist(listin, numBins):
    'Take a 1dimensional matrix or a list'
    plt.hist(listin, bins=numBins)
    plt.show()

def plotMatrixBinary(mat):
    plt.spy(mat, markersize=1, aspect='auto', origin='lower')
    plt.show()

def plotMatrixColor(mat):
    plotMatrixColorThresholds(mat, 0, 99)

def plotMatrixColorThresholds(mat, min_thresh, max_thresh):
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
    gvars = globals()
    gvars.update(locals())
    readline.set_completer(rlcompleter.Completer(gvars).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(gvars)
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
    plt.figure.tight_layout()
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
    plt.figure()
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

def PlotListofClusterArraysColor(list_of_arrays, have_divides): #have_divides is 0 to not show, otherwise show
    'Takes a list of 2D arrays, each of which is a populated cluster, and plots then in 3d.'
    # Try 3D plot
    colors2 = plt.get_cmap('gist_rainbow')
    num_clusters = len(list_of_arrays)
    cNorm = colortools.Normalize(vmin=0, vmax=num_clusters-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
    fig = plt.figure(figsize=(25,15)) # figsize=(x_inches, y_inches), default 80-dpi
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(num_clusters)])

    #TODO Below are tests to see if there is performance improvement when visualizing plots.
    ax.set_xlim([0, 1600])
    ax.set_ylim([0, 1600])
    ax.set_zlim([0, num_clusters])
    ax.view_init(elev=10., azim=0) #There is also a dist which can be set
    ax.dist = 8 # Default is 10, 0 is too low..
    ###END TODO


    for c in range(num_clusters):
        (x,y) = list_of_arrays[c].nonzero()
        ax.scatter(x,y, c, zdir='z', c=scalarMap.to_rgba(c))
        #plt.savefig("3D.png")

    if have_divides > 0:
        #HACK TODO(change from static-1600)
        [xx, yy] = np.meshgrid([0,1600],[0,1600]) # Doing a grid with just the corners yields much better performance.
        for plane in range(len(list_of_arrays)-1):
            ax.plot_surface(xx,yy,plane+.5, alpha=.05)
    fig.tight_layout()
    plt.show()

def AnimateClusterArrays(list_of_arrays):
    'Takes a list of arrays, each of which is a populated cluster.'
    start_time = time.time()
    #Elev and azim are both in degrees
    colors2 = plt.get_cmap('gist_rainbow')
    num_clusters = len(list_of_arrays)
    cNorm = colortools.Normalize(vmin=0, vmax=num_clusters-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)


    fig = plt.figure(figsize=(32,18), dpi=100) # figsize=(x_inches, y_inches), default 80-dpi
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(num_clusters)])

    #HACK TODO
    total_frames = 1940
    t0 = time.time()
    def animate(i):
        if (i%20 == 0):
            curtime = time.time()
            temp = curtime - t0
            m = math.floor(temp / 60)
            print('Done with: ' + str(i) + '/' + str(total_frames) +
                  ' frames, = %.2f percent' % ((100 * i)/total_frames),end='')
            print('. Elapsed Time: ' + str(m) + ' minutes & %.0f seconds' % (temp % 60))
        if(i < 360): # Rotate 360 degrees around horizontal axis
            ax.view_init(elev=10., azim=i) #There is also a dist which can be set
        elif (i < 720):# 360 around vertical
            ax.view_init(elev=(10+i)%360., azim=0) #Going over
        elif (i < 1080):# 360 diagonal
            ax.view_init(elev=(10+i)%360., azim=i%360) #There is also a dist which can be set
        elif (i < 1100):# Quick rest
            #Sit for a sec to avoid sudden stop
            ax.view_init(elev=10., azim=0)
        elif (i < 1250): # zoom in(to)
            d = 13 - (i-1100)/15 # 13 because 0 is to zoomed, now has min zoom of 3
            ax.dist = d
        elif (i < 1790): #Spin from within, 540 degrees so reverse out backwards!
            ax.view_init(elev=(10+i-1250.), azim=0) #Going over
            ax.dist = 1
        else: # zoom back out(through non-penetrated side)
            d = 3 + (i-1790)/15
            ax.dist = d

    # Performance Increasers:
    ax.set_xlim([0, 1600])
    ax.set_ylim([0, 1600])
    ax.set_zlim([0, num_clusters])
    # ax.view_init(elev=10., azim=0) #There is also a dist which can be set
    # ax.dist = 3 # Default is 10

    for c in range(len(list_of_arrays)):
        (x,y) = list_of_arrays[c].nonzero()
        ax.scatter(x,y, c, zdir='z', c=scalarMap.to_rgba(c))

    [xx, yy] = np.meshgrid([0,1600],[0,1600]) # Doing a grid with just the corners yields much better performance.
    for plane in range(len(list_of_arrays)-1):
        ax.plot_surface(xx,yy,plane+.5, alpha=.05)

    plt.title('KMeans Clustering with 20 bins on ' + str(imagefile[8:-4] + '.mp4'))
    fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=20, blit=True) # 1100 = 360 + 360 + 360 + 30

    print('Saving, start_time: ' + str(time.ctime()))
    anim.save('Animation of ' + str(imagefile[8:-4]) + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    end_time = time.time()
    print('Time to save animation: ' + str(end_time - start_time))


class Pixel:
    'This class is being used to hold the coordinates, base info and derived info of a single pixle of a single image\'s layer'

    def __init__(self, xin, yin, value):
        self.x = xin # The x coordinate
        self.y = yin # The y coordinate
        self.val = value
    def setNeighborValues(self, non_zero_neighbors, neighbor_sum):
        self.nz_neighbors = non_zero_neighbors # The number out of the 8 surrounding pixels that are non-zero
        self.neighbor_sum = neighbor_sum # The sum of the surrounding 8 pixels


all_images = glob.glob('..\\data\\Swell*.tif')
print(all_images)

all_images = [all_images[0]] # HACK
print(all_images)

for imagefile in all_images:
    imagein = Image.open(imagefile)
    print('Starting on image: ' + imagefile)

    imarray = np.array(imagein)
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
    sorted_pixels = sorted(pixels, key=lambda tuplex: tuplex[0], reverse=True)
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

    cluster_arrays = [] # Each entry is an array, filled only with the maximal values from the corresponding
    for cluster in range(cluster_count):
        cluster_arrays.append(zeros([xdim, ydim])) # (r,c)
        for pixel in cluster_lists[cluster]:
            cluster_arrays[cluster][pixel[1]][pixel[2]] = pixel[0]
    # TODO may want to use numpy 3d array over a list of 2d arrays; remains to be checked for speed/memory



# TODO



PlotListofClusterArraysColor(cluster_arrays, 1)






sub_cluster_count = 10
# TODO time to switch to sparse matrices, it seems that there are indeed computational optimizations
# in addition to memory optimizations



# TODO look into DBSCAN from Sklearn as an alternate way to cluster
# TODO sklearn clustering techniques: http://scikit-learn.org/stable/modules/clustering.html
# for i in range(len(cluster_arrays)):
#     plotMatrixColorThresholds(cluster_arrays[i],0, 99)

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

####NOTE: intersting to note that sparse groups of pixels, including noise in a general area are often grouped together, probably due to their comparable lack of grouping improvements.
    # May be able to exploit this when removing sparse pixels.

#NOTE: What would happen if did iterative k-means until all pixels of all groups were touching, or there was a group consisting of a single pixel?