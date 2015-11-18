import math
import numpy as np
from munkres import Munkres
from myconfig import *
# from serodraw import debug

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
            if left_bound <= pixel.x <= right_bound and down_bound <= pixel.y <= up_bound:
                boundedpixels.append(pixel)
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
            for (pix_num2, pixel2) in enumerate(self.lowerpixels):
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
            for (pix_num2, pixel2) in enumerate(self.upperpixels):
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
            buf = math.sqrt(math.pow(pixel1.x - pixel2.x, 2) + math.pow(pixel1.y - pixel2.y, 2))
            if buf > 0.0: # Because floats..
                try:
                    return math.log(buf, 10)
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
            ndim = min(len(self.lowerpixels), len(self.upperpixels)) # HACK HACK min for now, in the hopes that munkres can handle non-square matrices.
            self.cost_array = np.zeros([ndim, ndim])
            for i in range(ndim):
                for j in range(ndim):
                    self.cost_array[i][j] = costBetweenPoints(self.lower_context_bins[i], self.upper_context_bins[j]) \
                    + distanceCostBetweenPoints(self.lowerpixels[i], self.upperpixels[j]) # TODO THIS IS NEW!!! WILL NEED ADJUSTING
            return self.cost_array

        makeCostArray()
        munk = Munkres()
        self.indeces = munk.compute(np.copy(self.cost_array))
        self.cost = 0
        for row, col in self.indeces:
            self.total_cost += self.cost_array[row][col]
            self.costs.append(self.cost_array[row][col])
            print('DB cost of:' + str(self.cost_array[row][col]) + ' from bin cost:' + str(costBetweenPoints(self.lower_context_bins[row], self.upper_context_bins[col])) + ' and cost from distance:' + str(distanceCostBetweenPoints(self.lowerpixels[row], self.upperpixels[col])))


    def __str__(self):
        return str('Stitch between slides:(' + str(self.lowerslidenum) + ',' + str(self.upperslidenum) + ') with blobs (' +
                   str(self.lowerblob.id) + ',' + str(self.upperblob.id) + '). Chose:' + str(len(self.lowerpixels)) +
                   '/' + str(len(self.lowerblob.edge_pixels)) + ' lower blob pixels and ' + str(len(self.upperpixels)) +
                   '/' + str(len(self.upperblob.edge_pixels)) + ' upper blob pixels. ' + 'TotalCost:' + str(self.total_cost))
    __repr__ = __str__

    def __init__(self, lowerblob, upperblob, overscan_scale, num_bins):
        self.overscan_scale = overscan_scale
        self.num_bins = num_bins
        self.lowerslidenum = lowerblob.slide.height # CHANGED
        self.upperslidenum = upperblob.slide.height # CHANGED
        self.lowerblob = lowerblob
        self.upperblob = upperblob
        self.upperpixels = self.edgepixelsinbounds(upperblob, lowerblob)
        self.lowerpixels = self.edgepixelsinbounds(lowerblob, upperblob) # TODO psoe on the order of lower and upper
        self.total_cost = -1
        self.costs = []
        self.isReduced = False # True when have chosen a subset of the edge pixels to reduce computation


        if len(self.lowerpixels) != 0: # Optimization
            self.upperpixels = self.edgepixelsinbounds(upperblob, lowerblob)

        if self.upperpixels is not None and len(self.upperpixels) != 0 and len(self.lowerpixels) != 0:
            # HACK
            # NOTE planning to reduce to a subset
            # NOTE 1:28 for (203,301) pre-opt, :37 for (174, 178), 66mins for (640, 616) -> 4 mins after optimization (picking half of each) -> 59 seconds with selective[::3]
            # NOTE After ::2 opt, total time for [:3] data slides = 10 mins 19 seconds, instead of ~ 2 hours, after selective[::3], total time = 6mins 49 seconds
            # selective [::3] with 5 slides = 36 mins

            if len(self.upperpixels) > max_pixels_to_stitch or len(self.lowerpixels) > max_pixels_to_stitch:
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
    def __init__(self, lowerpixel, upperpixel, lowerblob, upperblob,distance_cost, contour_cost):
        self.lowerpixel = lowerpixel
        self.upperpixel = upperpixel
        self.lowerblob = lowerblob
        self.upperblob = upperblob
        self.distance_cost = distance_cost
        self.contour_cost = contour_cost
        self.set_total_cost()

    def set_total_cost(self):#TODO find the best combo
        self.total_cost = self.distance_cost * self.contour_cost

    def __str__(self):
        print('Stitch between blob2ds:(' + str(self.lowerblob) + ',' + str(self.upperblob) + '), between pixels:(' \
        + str(self.lowerpixel) + ',' + str(self.upperpixel) + '), has (dist, contour, total) costs:(' \
        + str(self.distance_cost) + ',' str(self.contour_cost) + ',' + str(self.total_cost) + ')')
