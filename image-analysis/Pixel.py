import numpy as np
import math
class Pixel:
    '''
    This class is being used to hold the coordinates, base info and derived info of a pixel of a single image\'s layer
    '''

    id_num = 0
    total_pixels = 0
    all = dict() # A dictionary containing ALL Blob2ds. A blob2d's key is it's id
    @staticmethod
    def get(id):
        return Pixel.all.get(id)

    @staticmethod
    def getall():
        return Pixel.all.values()

    @staticmethod
    def getkeys():
        return Pixel.all.keys()


    def __init__(self, value, xin, yin, zin, validate=True):
        self.x = xin  # The x coordinate, int
        self.y = yin  # The y coordinate, int
        self.z = zin
        self.val = value  # float
        self.blob_id = -1 # 0 means that it is unset
        self.id = Pixel.total_pixels
        Pixel.total_pixels += 1

        if validate:
            self.validate();

    def validate(self):
        # self.id = Pixel.total_pixels
        Pixel.all[self.id] = self

    def setBlobID(self, new_val):
        self.blob_id = new_val

    def toTuple(self):
        return (self.val, self.x, self.y)

    def toArray(self):
        return np.array([self.val, self.x, self.y])

    def __str__(self):
        '''Method used to convert Pixel to string, generall for printing'''
        return str('P{ id:' + str(self.id) + ', [v:' + str(self.val) + ', x:' + str(self.x) + ', y:' + str(self.y) + ', z:' + str(self.z) + '], B2d_id:' + str(self.blob_id) + '}')
            # '[nzn:' + str(
            # self.nz_neighbors) + ', mn:' + str(self.maximal_neighbors) + ', ns:' + str(
            # self.neighbor_sum) + ', nc:' + str(self.neighbors_checked) + ']}')

    __repr__ = __str__

    def __lt__(self, other): # Used for sorting; 'less than'
        # Sort by y then x, so that (1,0) comes before (0,1) (x,y)

        if self.y == other.y:
            return self.x < other.x
        return self.y < other.y


    def getNeighbors(self, master_array): # TODO depreciated
        neighbors = []
        xpos = self.x
        ypos = self.y
        local_xdim, local_ydim = master_array.shape

        for horizontal_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1
            for vertical_offset in range(-1, 2, 1):  # NOTE CURRENTLY 1x1
                if (vertical_offset != 0 or horizontal_offset != 0):  # Don't measure the current pixel
                    if (xpos + horizontal_offset < local_xdim and xpos + horizontal_offset >= 0 and ypos + vertical_offset < local_ydim and ypos + vertical_offset >= 0):  # Boundary check.
                        neighbors.append(master_array[xpos + horizontal_offset][ypos + vertical_offset])
        return neighbors

    @staticmethod
    def midpointposition(pixel1, pixel2):
        return np.array([[(pixel1.x + pixel2.x) / 2, (pixel1.y + pixel2.y) / 2, (pixel1.z + pixel2.z) / 2]])

    @staticmethod
    def distancebetween(pixel1, pixel2):
        return math.sqrt(math.pow(pixel1.x - pixel2.x, 2) + math.pow(pixel1.y - pixel2.y, 2))
    @staticmethod
    def pixelidstodict(pixellist): # NOTE this takes ids, not pixels
        d = dict()

        # #DEBUG
        # print('DB allpixels:')
        # for pixel in Pixel.all:
        #     print(' ' + str(pixel))
        # #DEBUG
        # print('PRINTING PIXEL LIST')
        # for pixel in pixellist:
        #     print(pixel)
        all_pixels = all(type(pix) is Pixel for pix in pixellist)
        # print('\n\nDB all_pixel value is:' + str(all_pixels))
        # if len(pixellist):
        #     print('--Called pixels to dict with pixellist:' + str(pixellist))
        for pixel in pixellist:
            # print('DB pixel:' + str(pixel))
            # print('DB getting id:' + str(pixel))
            # buf = Pixel.get(pixel) # Converting from id to pixel
            # print('DB pixel before:' + str(pixel))
            #HACK
            if type(pixel) is int:

                pixel = Pixel.get(pixel) # Converting from id to pixel
            else:
                print('*************DB Didnt convert pixelid to pixel, original:' + str(pixel))
                # print('     Pixellist:' + str(pixellist))

            # HACK
            # print('Result is:' + str(pixel))
            d[pixel.x, pixel.y] = pixel
        return d

    def neighborsfromdict(self, dictin):
        found = []
        x=self.x
        y=self.y
        found.append(dictin.get((x+1, y)))
        found.append(dictin.get((x, y+1)))
        found.append(dictin.get((x+1, y+1)))
        found.append(dictin.get((x-1, y)))
        found.append(dictin.get((x, y-1)))
        found.append(dictin.get((x-1, y-1)))
        found.append(dictin.get((x-1, y+1)))
        found.append(dictin.get((x+1, y-1)))
        found = [val for val in found if val is not None]
        return found

    @staticmethod
    def pixelsToArray(pixels):
        '''
        :param pixels:
        :return: [The generated array, offsetx, offsety]
        '''
        # Note this does not group pixels that haven't been
        minx = min(pixel.x for pixel in pixels)
        maxx = max(pixel.x for pixel in pixels)
        miny = min(pixel.y for pixel in pixels)
        maxy = max(pixel.y for pixel in pixels)

        arr = np.zeros((maxx - minx + 1, maxy - miny + 1))
        for pixel in pixels:
            try:
                arr[pixel.x - minx][pixel.y - miny] = pixel.val
            except:
                print('Pixel:' + str(pixel))
        return [arr, minx, miny]

