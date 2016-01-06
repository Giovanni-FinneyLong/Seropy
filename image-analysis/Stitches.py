import math
import numpy as np
from munkres import Munkres
from myconfig import *
from serodraw import debug, progressBarUpdate
from Blob2d import Blob2d
import time

from Pixel import Pixel
class Pairing:
    """
    Only created when it is expected that two blobs from different slides belong to the same blob3d
    Contains the cost, and point information from stitching 2 blobs together.
    Contains 2 sets of mappings to their edge pixels
    As pairings are created before it is confirmed that two blobs may overlap or be suitable partners, the internal vars
        isConnected and isPartners indicate whether the Pairing are valid, in the sense of overlap and viable partners respectively
    """

    def edgepixelsinbounds(self, subsetblob, boundaryblob):
        """
        :param subsetblob: A blob2d from which we will find a subset of edge_pixels which are within an acceptable bound
        :param boundaryblob: A blob2d which is used to set the boundary constraints for pixels from subsetblob
        :param scale: Advised 1-1.2, the amount to expand the boundary of the boundary blob
            to contain the subset of the other blob
        :return: A list of pixels, chosen from subsetblob.edge_pixels,
            which are within the scaled boundary defined by boundaryblob
        """
        assert 0 < self.overscan_scale < 2
        left_bound = boundaryblob.avgx - ((boundaryblob.avgx - boundaryblob.minx) * self.overscan_scale)
        right_bound = boundaryblob.avgx + ((boundaryblob.maxx - boundaryblob.avgx) * self.overscan_scale)
        down_bound = boundaryblob.avgy - ((boundaryblob.avgy - boundaryblob.miny) * self.overscan_scale)
        up_bound = boundaryblob.avgy + ((boundaryblob.maxy - boundaryblob.avgy) * self.overscan_scale)
        boundedpixels = []
        for p_num, pixel in enumerate(subsetblob.edge_pixels): # TODO TODO TODO IMPORTANT, need two way setup, for blob and self. FIXME
            pixel = Pixel.get(pixel)
            if left_bound <= pixel.x <= right_bound and down_bound <= pixel.y <= up_bound:
                boundedpixels.append(pixel.id)
        return boundedpixels


    def setShapeContexts(self, num_bins):
        '''
        Uses the methods described here: https://www.cs.berkeley.edu/~malik/papers/BMP-shape.pdf
        to set a shape context histogram (with num_bins), for each edge pixel in the blob.
        Note that only the edge_pixels are used to determine the shape context,
        and that only edge points derive a context.

        num_bins is the number of bins in the histogram for each point
        '''
        # Note making the reference point for each pixel itself
        # Note that angles are NORMALLY measured COUNTER-clockwise from the +x axis,
        # Note  however the += 180, used to remove the negative values,
        # NOTE  makes it so that angles are counterclockwise from the NEGATIVE x-axis
        ledgep = len(self.lowerpixels)
        uedgep = len(self.upperpixels)

        self.lower_context_bins = np.zeros((ledgep , num_bins)) # Each edge pixel has rows of num_bins each
        self.upper_context_bins = np.zeros((uedgep , num_bins)) # Each edge pixel has rows of num_bins each
        # First bin is 0 - (360 / num_bins) degress
        for (pix_num, pixel) in enumerate(self.lowerpixels):
            pixel = Pixel.get(pixel)
            for (pix_num2, pixel2) in enumerate(self.lowerpixels):
                pixel2 = Pixel.get(pixel2)
                if pix_num != pix_num2: # Only check against other pixels.
                    distance = math.sqrt(math.pow(pixel.x - pixel2.x, 2) + math.pow(pixel.y - pixel2.y, 2))
                    angle = math.degrees(math.atan2(pixel2.y - pixel.y, pixel2.x - pixel.x)) # Note using atan2 handles the dy = 0 case
                    angle += 180
                    if not 0 <= angle <= 360:
                        print('\n\n\n--ERROR: Angle=' + str(angle))
                    # Now need bin # and magnitude for histogram
                    bin_num = math.floor((angle / 360.) * (num_bins - 1)) # HACK PSOE from -1
                    value = math.log(distance, 10)
                    # print('DB: Pixel:' + str(pixel) + ' Pixel2:' + str(pixel2) + ' distance:' + str(distance) + ' angle:' + str(angle) + ' bin_num:' + str(bin_num))
                    self.lower_context_bins[pix_num][bin_num] += value
        for (pix_num, pixel) in enumerate(self.upperpixels):

            # for pixel in Pixel.all.values():
            #     print(pixel)

            # print('DB pixel is originally:' + str(pixel))
            # print('Size of all pixels:' + str(len(Pixel.all)))

            pixel = Pixel.get(pixel)
            # print('Got pixel:' + str(pixel))
            # print('Upper pixels:' + str(self.upperpixels))

            for (pix_num2, pixel2) in enumerate(self.upperpixels):
                pixel2 = Pixel.get(pixel2)
                if pix_num != pix_num2: # Only check against other pixels.
                    distance = math.sqrt(math.pow(pixel.x - pixel2.x, 2) + math.pow(pixel.y - pixel2.y, 2))
                    angle = math.degrees(math.atan2(pixel2.y - pixel.y, pixel2.x - pixel.x)) # Note using atan2 handles the dy = 0 case
                    angle += 180
                    if not 0 <= angle <= 360:
                        print('\n\n\n--ERROR: Angle=' + str(angle))
                    # Now need bin # and magnitude for histogram
                    bin_num = math.floor((angle / 360.) * (num_bins - 1)) # HACK PSOE from -1
                    value = math.log(distance, 10)
                    # print('DB: Pixel:' + str(pixel) + ' Pixel2:' + str(pixel2) + ' distance:' + str(distance) + ' angle:' + str(angle) + ' bin_num:' + str(bin_num))
                    self.upper_context_bins[pix_num][bin_num] += value


    def munkresCost(self):
        """
        Finds the minimal cost of pairings between the chosen subsets of pixels and their generated context bins using their cost array
        Generates self.cost_array based on the subsets of pixels
        Then used the hungarian method (Munkres) to find the optimal subset of indeces & total cost from the given subset of edge_pixels
        :return:
        """
        def costBetweenPoints(bins1, bins2):
            assert len(bins1) == len(bins2)
            cost = 0
            for i in range(len(bins1)):
                if (bins1[i] + bins2[i]) != 0:
                    cost += math.pow(bins1[i] - bins2[i], 2) / (bins1[i] + bins2[i])
            return cost / 2

        def distanceCostBetweenPoints(pixel1, pixel2):
            pixel1 = Pixel.get(pixel1)
            pixel2 = Pixel.get(pixel2)

            buf = math.sqrt(math.pow(pixel1.x - pixel2.x, 2) + math.pow(pixel1.y - pixel2.y, 2))
            if buf > 0.0: # Because floats..
                try:
                    return math.log(buf, 2)
                except:
                    print('DB ERROR: buf = ' + str(buf))
                    import pdb
                    pdb.set_trace()
            else:
                return 0.0

        def makeCostArray():
            """
            Generates a cost array (self.cost_array) from the subset of pixels
            :return:
            """
            # Setting up cost array with the costs between respective points
            ndim = min(len(self.lowerpixels), len(self.upperpixels)) # Munkres can handle non-square matrices (by making them square)
            self.cost_array = np.zeros([ndim, ndim, 4])
            munkres_array = np.zeros([ndim,ndim]) #TODO FIXME replace this with slicing of cost array on third element
            for i in range(ndim):
                for j in range(ndim):
                    contourCost = costBetweenPoints(self.lower_context_bins[i], self.upper_context_bins[j])
                    distanceCost = distanceCostBetweenPoints(self.lowerpixels[i], self.upperpixels[j])
                    distance = math.sqrt(math.pow(Pixel.get(self.lowerpixels[i]).x - Pixel.get(self.upperpixels[j]).x, 2) + math.pow(Pixel.get(self.lowerpixels[i]).y - Pixel.get(self.upperpixels[j]).y, 2))
                    net_cost = contourCost * distanceCost
                    self.cost_array[i][j] = [contourCost, distanceCost, net_cost, distance] # TODO can reduce this later for efficiency
                    munkres_array[i][j] = net_cost
                    # self.cost_array[i][j] = costBetweenPoints(self.lower_context_bins[i], self.upper_context_bins[j]) \
                    # * distanceCostBetweenPoints(self.lowerpixels[i], self.upperpixels[j]) # TODO THIS IS NEW!!! WILL NEED ADJUSTING
            return munkres_array

        munkres_array = makeCostArray()
        munk = Munkres()

        self.indeces = munk.compute(munkres_array)
        self.cost = 0
        for row, col in self.indeces:
            # NOTE Sorting indeces = [contour_cost, dist_cost, total_cost, distance]
            if self.cost_array[row][col][3] < max_distance and self.cost_array[row][col][2] < max_stitch_cost :# HACK, may want to do this later, so that stitches can be manually removed via interactive interface (slide bar for max_value)
                self.stitches.append(Stitch(self.lowerpixels[row], self.upperpixels[col], self.lowerblob, self.upperblob, self.cost_array[row][col]))
                self.cost += self.cost_array[row][col][2] # 2 for total_cost
                # print('DB cost of (contour, distance, total):' + str(self.cost_array[row][col]))
            # else:
            #     print('Ignored cost:' + str(self.cost_array[row][col]))

    @staticmethod
    def stitchAllBlobs(slidelist, quiet=True, debug=False):
        def  printElapsedTime(t0, tf, pad=''): # HACK FIXME REMOVE THIS AND IMPORT CORRECTLY
            temp = tf - t0
            m = math.floor(temp / 60)
            plural_minutes = ''
            if m > 1:
                plural_minutes = 's'
            if m > 0:
                print(pad + 'Elapsed Time: ' + str(m) + ' minute' + str(plural_minutes) + ' & %.0f seconds' % (temp % 60))
            else:
                print(pad + 'Elapsed Time: %.2f seconds' % (temp % 60))
        pairlist = []
        if not quiet:
            print('Beginning to stitch together blobs')



        # total_edge_pixels = sum(len(blob2d.edge_pixels) for slide in slidelist for blob2d in slide.blob2dlist)
        # updateStatus = 0
        # pixels_processed = 0


        last_print = 0


        for slide_num, slide in enumerate(slidelist):
            #if not quiet or (debug and slide.debugFlag is True):
            print('\nStitching slide #' + str(slide_num) + '/' + str(len(slidelist)) + ', which contains ' + str(len(slide.blob2dlist)) + ' Blob2ds')

            for b_num, blob1 in enumerate(slide.blob2dlist):
                #Converting to static:
                blob1 = Blob2d.get(blob1)
                print('DB:' + str(((b_num - last_print) / len(slide.blob2dlist))))
                if ((b_num - last_print) / len(slide.blob2dlist)) >= .1:
                    print('.', end='')
                    last_print = b_num



                if len(blob1.possible_partners) > 0:
                    if debug and blob1.debugFlag is True:
                        print('  Starting on a new blob from bloblist:' + str(blob1) + ' which has:' + str(len(blob1.possible_partners)) + ' possible partners')
                # print('  Blob1 current parter_costs:' + str(blob1.partner_costs))

                sub_start_time = time.time()

                for b2_num, blob2 in enumerate(blob1.possible_partners):
                    #Converting to static:
                    blob2 = Blob2d.get(blob2)
                    if debug and (blob1.debugFlag is True or blob2.debugFlag is True):
                        print('   Comparing to blob2:' + str(blob2))
                    t0 = time.time()
                    bufStitch = Pairing(blob1, blob2, 1.1, 36, quiet=quiet)
                    if bufStitch.isConnected:
                        if (debug and (blob1.debugFlag is True or blob2.debugFlag is True)):
                            print('    +Blobs connected')
                        pairlist.append(bufStitch)
                        if not quiet:
                            tf = time.time()
                            printElapsedTime(t0, tf, pad='    ')
                    elif (debug and (blob1.debugFlag is True or blob2.debugFlag is True)):
                        print('    -Blobs not connected')
                # updateStatus = progressBarUpdate(pixels_processed, total_edge_pixels, last_update=updateStatus, steps=100)
                # pixels_processed += len(blob1.edge_pixels)




        return pairlist

    def __str__(self):
        if self.cost == -1:
            cost_str = 'Unset'
        else:
            cost_str = str(self.cost)

        return str('<Pairing between blob2ds at heights:(' + str(self.lowerheight) + ',' + str(self.upperheight) + ') with ids (' +
                   str(self.lowerblob.id) + ',' + str(self.upperblob.id) + '). Chose:' + str(len(self.lowerpixels)) +
                   '/' + str(len(self.lowerblob.edge_pixels)) + ' lower blob pixels and ' + str(len(self.upperpixels)) +
                   '/' + str(len(self.upperblob.edge_pixels)) + ' upper blob pixels. ' + 'Cost:' + cost_str + '>')
    __repr__ = __str__

    def __init__(self, lowerblob, upperblob, overscan_scale, num_bins, quiet=True):
        self.overscan_scale = overscan_scale
        self.num_bins = num_bins
        self.lowerheight = lowerblob.height # CHANGED
        self.upperheight = upperblob.height # CHANGED
        self.lowerblob = lowerblob
        self.upperblob = upperblob
        self.upperpixels = self.edgepixelsinbounds(upperblob, lowerblob)
        self.lowerpixels = self.edgepixelsinbounds(lowerblob, upperblob) # TODO psoe on the order of lower and upper
        self.isReduced = False # True when have chosen a subset of the edge pixels to reduce computation
        self.stitches = []
        self.cost = -1 # Just to indicate that it is unset


        if len(self.lowerpixels) != 0: # Optimization
            self.upperpixels = self.edgepixelsinbounds(upperblob, lowerblob)

        if self.upperpixels is not None and len(self.upperpixels) != 0 and len(self.lowerpixels) != 0:
            # HACK
            # NOTE planning to reduce to a subset
            # NOTE 1:28 for (203,301) pre-opt, :37 for (174, 178), 66mins for (640, 616) -> 4 mins after optimization (picking half of each) -> 59 seconds with selective[::3]
            # NOTE After ::2 opt, total time for [:3] data slides = 10 mins 19 seconds, instead of ~ 2 hours, after selective[::3], total time = 6mins 49 seconds
            # selective [::3] with 5 slides = 36 mins

            if len(self.upperpixels) > max_pixels_to_stitch or len(self.lowerpixels) > max_pixels_to_stitch:
                if not quiet:
                    print('-->Too many pixels in the below stitch, reducing to a subset, originally was: ' + str(len(self.lowerpixels)) +
                        '/' + str(len(self.lowerblob.edge_pixels)) + ' lower blob pixels and ' + str(len(self.upperpixels)) +
                        '/' + str(len(self.upperblob.edge_pixels)) + ' upper blob pixels.')
                pickoneovers = max(1, math.ceil(len(self.upperpixels) / max_pixels_to_stitch)), max(1, math.ceil(len(self.lowerpixels) / max_pixels_to_stitch)) # HACK TODO Modify these values to be more suitable dependent on computation time
                self.isReduced = True
                # if len(self.upperpixels) > 500 and len(self.lowerpixels) > 500:
                #     pickoneover = 5

                self.upperpixels = self.upperpixels[::pickoneovers[0]] # Every pickoneover'th element
                self.lowerpixels = self.lowerpixels[::pickoneovers[1]] # HACK this is a crude way of reducing the number of pixels

            self.isConnected = True
            self.setShapeContexts(num_bins) # Set lower and upper context bins
            if not quiet:
                print('   ' + str(self))
            self.munkresCost() # Now have set self.cost and self.indeces and self.connect
            lowerblob.updatePairings(self)
            upperblob.updatePairings(self)
        else:
            self.isConnected = False


class Stitch:
    '''
    A single instance of a stitch between two blob2ds.
    There may be many 'Stitch' object to one 'Pairing' between two blob2ds
    '''
    def __init__(self, lowerpixel, upperpixel, lowerblob, upperblob, cost):
        self.lowerpixel = lowerpixel
        self.upperpixel = upperpixel
        self.lowerblob = lowerblob
        self.upperblob = upperblob
        self.cost = cost
        self.distance = math.sqrt(math.pow(Pixel.get(lowerpixel).x - Pixel.get(upperpixel).x, 2) + math.pow(Pixel.get(lowerpixel).y - Pixel.get(upperpixel).y, 2)) # TODO replace with distancebetween static method from Pixel
            # math.sqrt(math.pow(lowerpixel.x - upperpixel.x, 2) + math.pow(lowerpixel.y - upperpixel.y, 2))


    def __str__(self):
        return str('<Stitch between blob2ds:(' + str(self.lowerblob.id) + ',' + str(self.upperblob.id) + '), between pixels:(' \
        + str(self.lowerpixel) + ',' + str(self.upperpixel) + '), has cost:' \
        + str(self.cost) + ')')

    def __repr__(self):
        return self.__str__()
