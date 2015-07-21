__author__ = 'gio'
# This file includes the various functions written to visualize data from within sero.py
# These functions have been separated for convenience; they are higher volume and lower maintenance.

from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import matplotlib.colors as colortools
from matplotlib import animation
import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np
import time
import math
from PIL import Image
from visvis.vvmovie.images2gif import writeGif
# from Scripts.images2gif import writeGif
import readline
import code
import rlcompleter
import glob
# import wand
# import cv2 # OpenCV version 2
import subprocess


def runShell():
    gvars = globals()
    gvars.update(locals())
    readline.set_completer(rlcompleter.Completer(gvars).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(gvars)
    shell.interact()

def PlotHist(listin, numBins):
    'Take a 1dimensional matrix or a list'
    plt.hist(listin, bins=numBins)
    plt.show()

def PlotCounterHist(counter, numBins):
    PlotHist(list(counter.values()), numBins)

def PlotCounter(counter):
    labels, values = zip(*counter.items())
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()

def PlotMatrixBinary(mat):
    plt.spy(mat, markersize=1, aspect='auto', origin='lower')
    plt.show()

def PlotMatrixColor(mat):
    PlotMatrixColorThresholds(mat, 0, 99)

def PlotMatrixColorThresholds(mat, min_thresh, max_thresh):
    plt.imshow(mat, vmin=min_thresh, vmax=max_thresh) # 0,99 are min,max defaults
    plt.colorbar()
    plt.show()

def plotMatrixPair(m1, m2):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(32,18))
    cmap = cm.jet
    # matplotlib.style.use('ggplot')
    plt.set_cmap(cmap)
    ax1.spy(m1, markersize=1, aspect='auto', origin='lower')
    ax2.spy(m2, markersize=1, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.show()

def PlotMatrixTrio(m1, m2, m3):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(32,18))
    cmap = cm.jet
    #plt.style.use('ggplot')
    plt.set_cmap(cmap)
    ax1.spy(m1, markersize=1, aspect='auto', origin='lower')
    ax2.spy(m2, markersize=1, aspect='auto', origin='lower')
    ax3.spy(m3, markersize=1, aspect='auto', origin='lower')
    plt.show()

def FindBestClusterCount(min, max, step):
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

def PlotListofClusterArraysColor2D(list_of_arrays, markersize):
    colors2 = plt.get_cmap('gist_rainbow')
    num_clusters = len(list_of_arrays)
    cNorm = colortools.Normalize(vmin=0, vmax=num_clusters-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
    fig = plt.figure(figsize=(32,18)) # figsize=(x_inches, y_inches), default 80-dpi
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 1600])
    ax.set_ylim([0, 1600])

    for c in range(num_clusters):
        (x,y) = list_of_arrays[c].nonzero()
        ax.scatter(x,y, s=markersize, c=scalarMap.to_rgba(c))
        #plt.savefig("3D.png")
    fig.tight_layout()
    plt.show()

def AnimateClusterArraysGif(list_of_arrays, imagefile, draw_divides):
    total_frames = 1940 #HACK

    speed_scale = 1. # Default is 1 (normal speed), 2 = 2x speed, .5 = .5x speed
    total_frames = math.floor(total_frames / speed_scale)

    colors2 = plt.get_cmap('gist_rainbow')
    num_clusters = len(list_of_arrays)
    cNorm = colortools.Normalize(vmin=0, vmax=num_clusters-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
    fig = plt.figure(figsize=(8.0,4.5), dpi=100) # figsize=(x_inches, y_inches), default 80-dpi
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    t0 = time.time()

    def animate(i):
        i = i * speed_scale
        if (i%20 == 0):
            curtime = time.time()
            temp = curtime - t0
            m = math.floor(temp / 60)
            print('Done with: ' + str(i) + '/' + str(total_frames * speed_scale) + ' frames, = %.2f percent' % ((100 * i)/(total_frames * speed_scale)), end='')
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

    def generateFrames():
        'Takes a list of arrays, each of which is a populated cluster.'

        #Elev and azim are both in degrees
        # Performance Increasers:
        ax.set_xlim([0, 1600])
        ax.set_ylim([0, 1600])
        ax.set_zlim([0, num_clusters])

        for c in range(len(list_of_arrays)):
            (x,y) = list_of_arrays[c].nonzero()
            ax.scatter(x, y, c, zdir='z', c=scalarMap.to_rgba(c))
        if draw_divides != 0:
            [xx, yy] = np.meshgrid([0, 1600],[0, 1600]) # Doing a grid with just the corners yields much better performance.
            for plane in range(len(list_of_arrays)-1):
                ax.plot_surface(xx, yy, plane+.5, alpha=.05)
        fig.tight_layout()
        print('Generating and saving frames, start_time: ' + str(time.ctime()))
        for i in range(total_frames):

            animate(i)
            #im = fig2img(fig)
            #im.show()
            plt.savefig('temp/gif_frame' + str(i) + '.png', bbox_inches='tight')
            #frames.append(im)
    def framesToGif(): # TODO convert to calling executable with: http://pastebin.com/JJ6ZuXdz
        # HACK
        imagemagick_convert_exec = 'C:\\Program Files\\ImageMagick-6.9.1-Q8\\convert.exe'
        # HACK
        frame_names = 'temp/*.png' # glob.glob('temp/*.png')
        #print('Frame names:' + str(frame_names))
        #frames = [Image.open(frame_name) for frame_name in frame_names]
        imagex = 800
        imagey = 450
        filename_out = str(imagefile[-12:-4] + '.gif')
        print('Now writing gif to:' + str(filename_out))

        command = [imagemagick_convert_exec, "-delay", "0", "-size", str(imagex)+'x'+str(imagey)] + [frame_names] + [filename_out]
        t1 = time.time()
        m = math.floor((t1-t0) / 60)
        s = (t1-t0) % 60
        print('It has been:' + str(m) + ' mins & ' + str(s) + ' seconds, now calling imagemagick_executable to generate gif')
        subprocess.call(command)
        t2 = time.time()
        m = math.floor((t2-t1) / 60)
        s = (t2-t1) % 60
        print('Done saving animated gif; took: ' + str(m) + ' mins & ' + str(s) + ' seconds.')
        runShell()

        # writeGif(filename, frames, duration=100, dither=0)



        print('Done writing gif')
    def framesToGifOpenCV():
        frame_names = glob.glob('temp/*.png')
        print('Frame names:' + str(frame_names))
        frames = [Image.open(frame_name) for frame_name in frame_names]
        print('Now writing gif')
        filename = imagefile[-12:-4] + '.gif'
        width = 800
        height = 450

        video = cv2.VideoWriter(filename,
                                -1, # Manual codec selection
                                10, #FPS
                                (width, height))
    def GifImageMagick():
        print('Generating image-magick anim')
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=20, blit=True) # 1100 = 360 + 360 + 360 + 30
        print('Now writing gif')
        filename = imagefile[-12:-4] + '.gif'
        print('Saving ImageMagick gif as:' + str(filename))
        anim.save("test.gif", writer='imagemagick', fps=10)
        # runShell()
        print('Done writing gif')

    #generateFrames()
    framesToGif()
    #framesToGifOpenCV()
    # GifImageMagick()

def AnimateClusterArrays(list_of_arrays, imagefile, draw_divides): #Image file just used for name
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
    if draw_divides != 0:
        [xx, yy] = np.meshgrid([0, 1600],[0, 1600]) # Doing a grid with just the corners yields much better performance.
        for plane in range(len(list_of_arrays)-1):
            ax.plot_surface(xx, yy, plane+.5, alpha=.05)

    plt.title('KMeans Clustering with 20 bins on ' + str(imagefile[8:-4] + '.mp4'))
    fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=20, blit=True) # 1100 = 360 + 360 + 360 + 30

    print('Saving, start_time: ' + str(time.ctime()))
    anim.save('Animation of ' + str(imagefile[8:-4]) + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    end_time = time.time()
    print('Time to save animation: ' + str(end_time - start_time))

# Note that these last methods are not exclusively meant for drawing, but for now fulfill only that function
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

