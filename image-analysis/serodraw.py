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
from numpy import zeros
from visvis.vvmovie.images2gif import writeGif
# from Scripts.images2gif import writeGif
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import glob
# import wand
# import cv2 # OpenCV version 2
import subprocess
import readline
import code
import rlcompleter
import pdb
import os

from config import *


# NOTE  ##########################
# NOTE  Setting up global vars:
current_path = os.getcwd()
xdim = -1
ydim = -1
zdim = -1

debug_blob_ids = False
debug_pixel_ops = False
debug_set_merge = False
remap_ids_by_group_size = True
test_instead_of_data = False
debug_pixel_ops_y_depth = 500

min_val_threshold = 250
    # Recommended = 250
max_val_step = 5 # The maximum amount that two neighboring pixels can differ in val and be grouped by blob_id
    # Recommended = 5
minimal_nonzero_neighbors = 2 # The minimal amount of nzn a pixel must have to avoid being filtered; 0 = no filter
    # Recommended = 2
# NOTE  ##########################



def setseerodrawdims(x,y,z):
    global xdim
    global ydim
    global zdim
    xdim = x
    ydim = y
    zdim = z


def runShell():
    gvars = globals()
    gvars.update(locals())
    readline.set_completer(rlcompleter.Completer(gvars).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(gvars)
    shell.interact()

def debug():
    pdb.set_trace()

def timeNoSpaces():
    return time.ctime().replace(' ', '_').replace(':', '-')

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

def PlotClusterLists(list_of_lists, **kwargs):
    '''
    Takes a list of lists, each list is a the pixels of the corresponding cluster
    '''
    dimensions = kwargs.get('dim', '2D').lower()
    if (dimensions != '2d' and dimensions != '3d'):
        print('ERROR, dimensions must be 2d or 3d!!!!!')
    else:
        cluster_count = len(list_of_lists)
        cluster_arrays = []  # Each entry is an array, filled only with the maximal values from the corresponding
        for cluster in range(cluster_count):
            cluster_arrays.append(zeros([xdim, ydim]))  # (r,c)
            for pixel in list_of_lists[cluster]:
                cluster_arrays[cluster][pixel.x][pixel.y] = int(pixel.val)
        if dimensions == '2d':
            PlotListofClusterArraysColor2D(cluster_arrays, **kwargs)
        else:
            PlotListofClusterArraysColor3D(cluster_arrays, 1)

def FindBestClusterCount(array_of_floats, min, max, step):
    print('Attempting to find optimal number of clusters, range:(' + str(min) + ', ' + str(max))
    kVals = [] # The number of clusters
    distortionVSclusters = [] # The distortion per cluster
    for z in range(math.floor((max - min) / step)):
        num_clusters = (z * step) + min
        if(num_clusters == 0):
            num_clusters = 1
        print('Trying with ' + str(num_clusters) + ' clusters')
        (bookC, distortionC)  = kmeans(array_of_floats, num_clusters)
        # (centLabels, centroids) = vq(max_pixel_array_floats, bookC
        kVals.append(num_clusters)
        distortionVSclusters.append(distortionC)
    plt.plot(kVals, distortionVSclusters, marker='x')
    plt.grid(True)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Average cluster distortion')
    plt.title('Elbow method for K-Means Clustering on Pixels\nManually Record the desired K value\n')
    plt.show()

def PlotListofClusterArraysColor3D(list_of_arrays, have_divides): #have_divides is 0 to not show, otherwise show
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

    ax.set_xlim([0, xdim])
    ax.set_ylim([ydim, 0])
    ax.set_zlim([0, num_clusters])
    ax.view_init(elev=10., azim=0) #There is also a dist which can be set
    ax.dist = 8 # Default is 10, 0 is too low..


    for c in range(num_clusters):
        (x,y) = list_of_arrays[c].nonzero()
        ax.scatter(x,y, c, zdir='z', c=scalarMap.to_rgba(c))
        #plt.savefig("3D.png")

    if have_divides > 0:
        [xx, yy] = np.meshgrid([0, xdim], [0, ydim]) # Doing a grid with just the corners yields much better performance.
        for plane in range(len(list_of_arrays)-1):
            ax.plot_surface(xx, yy, plane+.5, alpha=.05)
    fig.tight_layout()
    plt.show()

def PlotListofClusterArraysColor2D(list_of_arrays, **kwargs):
    numbered = kwargs.get('numbered', False) # Output the pixel's blob id and order in the id list
    # Note: Numbering greatly increase draw time
    label_start_finish = kwargs.get('marked', False) # X on the first element of a blob, + on the last
    figsize = kwargs.get('figsize', (32, 32))
    markersize = kwargs.get('markersize', 30)

    colors2 = plt.get_cmap('gist_rainbow')
    num_clusters = len(list_of_arrays)
    cNorm = colortools.Normalize(vmin=0, vmax=num_clusters-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
    fig = plt.figure(figsize=figsize) # figsize=(x_inches, y_inches), default 80-dpi
    plt.clf()
    ax = fig.add_subplot(111)

    ax.set_xlim([0, xdim])
    ax.set_ylim([ydim, 0])

    for c in range(num_clusters):
        (x,y) = list_of_arrays[c].nonzero()
        ax.scatter(x,y, s=markersize, c=scalarMap.to_rgba(c), edgecolor=scalarMap.to_rgba(c))
        if label_start_finish:
            ax.plot(x[0], y[0], marker='x', markersize=markersize)
            ax.plot(x[-1], y[-1], marker='+', markersize=markersize)
            ax.annotate(str(c), xy=(x[0], y[0]))
            ax.annotate('\\'+str(c), xy=(x[-1], y[-1]))
        if numbered:
            for lab in range(len(x)):
                ax.annotate(str(c) + '.' + str(lab), xy=(x[lab], y[lab]))
        #plt.savefig("3D.png")
    fig.tight_layout()
    plt.savefig('temp/2D_Plot_of_Cluster_Arrays__' + timeNoSpaces() + '.png')
    plt.show()

def AnimateClusterArraysGif(list_of_arrays, imagefile, **kwargs):
    start_time = time.time()
    draw_divides = kwargs.get('divides', False)
    video_format = kwargs.get('format', 'MP4') # Either Mp4 or gif
    video_format = video_format.lower()
    figsize = kwargs.get('figsize', (8,4.5))

    ok_to_run = True

    if video_format != 'mp4' and video_format != 'gif':
        print('INVALID VIDEO FORMAT PROVIDED:' + str(video_format))
        ok_to_run = False
    else:
        total_frames = 617 #1940 #HACK
        speed_scale = 1 # Default is 1 (normal speed), 2 = 2x speed, **must be int for now due to range()
        total_frames = math.floor(total_frames / speed_scale)
        frame_offset = 615

        colors2 = plt.get_cmap('gist_rainbow')
        num_clusters = len(list_of_arrays)
        cNorm = colortools.Normalize(vmin=0, vmax=num_clusters-1)
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colors2)
        fig = plt.figure(figsize=figsize, dpi=100) # figsize=(x_inches, y_inches), default 80-dpi
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        t0 = time.time()

        # DEBUG
        # Frame continuitiny issues between frames:
        # 620/621 # Changed to be % 270..

        # NOTE making new directory for animation for organization:
        animation_time_string = timeNoSpaces()
        animation_folder = current_path + '\\temp\\' + animation_time_string
        os.makedirs(animation_folder)



    def animate(i):
        # i = (i * speed_scale) + frame_offset
        if i%1 == 0:
            curtime = time.time()
            temp = curtime - t0
            m = math.floor(temp / 60)
            print('Done with: ' + str(i) + '/' + str(total_frames / speed_scale) + ' frames, = %.2f percent' % (100 * ( i - frame_offset)/(total_frames / speed_scale)), end='')
            print('. Elapsed Time: ' + str(m) + ' minutes & %.0f seconds' % (temp % 60))
        if i < 360: # Rotate 360 degrees around horizontal axis
            ax.view_init(elev=10., azim=i) #There is also a dist which can be set
        elif i < 720:# 360 around vertical
            ax.view_init(elev=(10+i)%360., azim=0) #Going over
        elif i < 1080:# 360 diagonal
            ax.view_init(elev=(ax.elev + 1), azim=i%360) #There is also a dist which can be set
        elif i < 1100:# Quick rest
            #Sit for a sec to avoid sudden stop
            ax.view_init(elev=10., azim=0)
        elif i < 1250: # zoom in(to)
            d = 13 - (i-1100)/15 # 13 because 0 is to zoomed, now has min zoom of 3
            ax.dist = d
        elif i < 1790: #Spin from within, 540 degrees so reverse out backwards!
            ax.view_init(elev=(ax.elev + 1), azim=0) #Going over
            ax.dist = 1
        else: # zoom back out(through non-penetrated side)
            d = 3 + (i-1790)/15
            ax.dist = d

    def generateFrames():
        'Takes a list of arrays, each of which is a populated cluster.'

        #Elev and azim are both in degrees
        # Performance Increasers:
        ax.set_xlim([0, xdim])
        ax.set_ylim([ydim, 0])
        ax.set_zlim([0, num_clusters])

        for c in range(len(list_of_arrays)):
            (x, y) = list_of_arrays[c].nonzero()
            ax.scatter(x, y, c, zdir='z', c=scalarMap.to_rgba(c))
        if draw_divides != 0:
            [xx, yy] = np.meshgrid([0, 1600],[0, 1600]) # Doing a grid with just the corners yields much better performance.
            for plane in range(len(list_of_arrays)-1):
                ax.plot_surface(xx, yy, plane+.5, alpha=.05)
        fig.tight_layout()
        print('Generating and saving frames, start_time: ' + str(time.ctime()) + ', saving to folder: ' + str(animation_folder))
        for i in range(frame_offset,total_frames, speed_scale):

            animate(i)
            #im = fig2img(fig)
            #im.show()
            buf = i
            padding = '00000' # Hack
            buf_digits = buf
            while buf_digits >= 10:
                padding = padding[1:]
                buf_digits = buf_digits / 10
            plt.savefig(animation_folder + '/gif_frame_' + padding + str(buf) + '.png', bbox_inches='tight')
            #frames.append(im)

    def framesToGif(): # TODO convert to calling executable with: http://pastebin.com/JJ6ZuXdz
        # HACK
        # IMAGEMAGICK_CONVERT_EXEC = 'C:\\Program Files\\ImageMagick-6.9.1-Q8\\convert.exe'
        # HACK
        frame_names = animation_folder + '/*.png' # glob.glob('temp/*.png')
        #print('Frame names:' + str(frame_names))
        #frames = [Image.open(frame_name) for frame_name in frame_names]
        imagex = 10 * figsize[0]
        imagey = 10 * figsize[1]
        filename_out = (imagefile[-12:-4] + '_' + animation_time_string + '.gif')
        print('Now writing gif to:' + str(filename_out))

        command = [IMAGEMAGICK_CONVERT_EXEC, "-delay", "0", "-size", str(imagex)+'x'+str(imagey)] + [frame_names] + [filename_out]
        t1 = time.time()
        m = math.floor((t1-t0) / 60)
        s = (t1-t0) % 60
        print('It has been:' + str(m) + ' mins & ' + str(s) + ' seconds, now calling imagemagick_executable to generate gif')
        subprocess.call(command)
        t2 = time.time()
        m = math.floor((t2-t1) / 60)
        s = (t2-t1) % 60
        print('Done saving animated gif; took: ' + str(m) + ' mins & ' + str(s) + ' seconds.')

        # writeGif(filename, frames, duration=100, dither=0)
        # TODO Change rotation over vertical 270 degrees
        # TODO Check that isnt also an issue horizontally
        # TODO Adjust the percentages output by animate(i)
        # TODO Check the ram issue's source; see if theres a way to view usage via debug somehow within pycharm
        # TODO Remove Anaconda3(Safely)
        print('Done writing gif')

    def GifImageMagick():
        print('Generating image-magick anim')
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=20, blit=True) # 1100 = 360 + 360 + 360 + 30
        print('Now writing gif')
        filename = imagefile[-12:-4] + '.gif'
        print('Saving ImageMagick gif as:' + str(filename))
        anim.save("test.gif", writer='imagemagick', fps=10)
        # runShell()
        print('Done writing gif')

    if ok_to_run:
        if video_format == 'mp4':
            for c in range(len(list_of_arrays)):
                (x,y) = list_of_arrays[c].nonzero()
                ax.scatter(x,y, c, zdir='z', c=scalarMap.to_rgba(c))
            if draw_divides:
                [xx, yy] = np.meshgrid([0, xdim],[0, ydim]) # Doing a grid with just the corners yields much better performance.
                for plane in range(len(list_of_arrays)-1):
                    ax.plot_surface(xx, yy, plane+.5, alpha=.05)

            plt.title('Animation_of_ ' + str(imagefile[8:-4] + '.mp4'))
            fig.tight_layout()
            anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=20, blit=True) # 1100 = 360 + 360 + 360 + 30
            print('Saving, start_time: ' + str(time.ctime()))
            anim.save('Animation_of ' + str(imagefile[8:-4]) + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            end_time = time.time()
            print('Time to save animation: ' + str(end_time - start_time))
        elif video_format == 'gif':
            generateFrames()
            framesToGif()

