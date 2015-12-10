import numpy as np
import math
class Pixel:
    '''
    This class is being used to hold the coordinates, base info and derived info of a pixel of a single image\'s layer
    '''

    id_num = 0
    def __init__(self, value, xin, yin, zin):
        self.x = xin  # The x coordinate, int
        self.y = yin  # The y coordinate, int
        self.z = zin
        self.val = value  # float
        self.nz_neighbors = 0
        self.maximal_neighbors = 0
        self.neighbor_sum = 0
        self.neighbors_checked = 0
        self.neighbors_set = False  # For keeping track later, in case things get nasty
        self.blob_id = -1 # 0 means that it is unset

    def setNeighborValues(self, non_zero_neighbors, max_neighbors, neighbor_sum, neighbors_checked):
        self.nz_neighbors = non_zero_neighbors  # The number out of the 8 surrounding pixels that are non-zero
        self.maximal_neighbors = max_neighbors
        self.neighbor_sum = neighbor_sum  # The sum of the surrounding 8 pixels
        self.neighbors_checked = neighbors_checked
        self.neighbors_set = True

    def setBlobID(self, new_val):
        self.blob_id = new_val

    def toTuple(self):
        return (self.val, self.x, self.y)

    def toArray(self):
        return np.array([self.val, self.x, self.y])

    def __str__(self):
        '''Method used to convert Pixel to string, generall for printing'''
        return str('P{[v:' + str(self.val) + ', x:' + str(self.x) + ', y:' + str(self.y) + ', z:' + str(self.z) + '], id:' + str(self.blob_id) + '}')
            # '[nzn:' + str(
            # self.nz_neighbors) + ', mn:' + str(self.maximal_neighbors) + ', ns:' + str(
            # self.neighbor_sum) + ', nc:' + str(self.neighbors_checked) + ']}')

    __repr__ = __str__

    def __lt__(self, other): # Used for sorting; 'less than'
        # Sort by y then x, so that (1,0) comes before (0,1) (x,y)
        if self.y < other.y:
            return True
        elif self.y == other.y:
            return self.x < other.x
        else:
            return False

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
    def pixelstodict(pixellist):
        d = dict()
        for pixel in pixellist:
            d[pixel.x, pixel.y] = pixel
        return d
    @staticmethod
    def neighborsfromdict(dictin, pixel):
        found = []
        x=pixel.x
        y=pixel.y
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