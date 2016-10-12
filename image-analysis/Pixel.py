import math

class Pixel:
    """
    This class is being used to hold the coordinates, base info and derived info of a pixel of a single image\'s layer
    """

    id_num = 0
    total_pixels = 0
    all = dict()  # A dictionary containing ALL Blob2ds. A blob2d's key is it's id

    @staticmethod
    def get(pixel_id):
        return Pixel.all.get(pixel_id)

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
        self.blob_id = -1  # 0 means that it is unset
        self.id = Pixel.total_pixels
        Pixel.total_pixels += 1

        if validate:
            self.validate()

    def validate(self):
        Pixel.all[self.id] = self

    def __str__(self):
        """Method used to convert Pixel to string, generall for printing"""
        return str('P{ id:' + str(self.id) + ', [v:' + str(self.val) + ', x:' + str(self.x) + ', y:' + str(
            self.y) + ', z:' + str(self.z) + '], B2d_id:' + str(self.blob_id) + '}')

    __repr__ = __str__

    def __lt__(self, other):  # Used for sorting; 'less than'
        # Sort by y then x, so that (1,0) comes before (0,1) (x,y)
        if self.y == other.y:
            return self.x < other.x
        return self.y < other.y

    @staticmethod
    def pixel_ids_to_dict(pixellist):  # NOTE this takes ids, not pixels
        d = dict()
        for pixel in pixellist:
            pixel = Pixel.get(pixel)  # Converting from id to pixel
            d[pixel.x, pixel.y] = pixel
        return d

    def get_neighbors_from_dict(self, dictin):
        found = []
        x = self.x
        y = self.y
        found.append(dictin.get((x + 1, y)))
        found.append(dictin.get((x, y + 1)))
        found.append(dictin.get((x + 1, y + 1)))
        found.append(dictin.get((x - 1, y)))
        found.append(dictin.get((x, y - 1)))
        found.append(dictin.get((x - 1, y - 1)))
        found.append(dictin.get((x - 1, y + 1)))
        found.append(dictin.get((x + 1, y - 1)))
        found = [val for val in found if val is not None]
        return found

    @staticmethod
    def cost_between_pixels(pixel1, pixel2):
        distance = Pixel.distance_between_pixels(pixel1, pixel2)
        if distance > 0.0:  # Because floats..
            return math.log(distance, 2)  # TODO adjust this?
        else:
            return 0.0

    @staticmethod
    def distance_between_pixels(pixel1, pixel2):
        """
        Returns the X,Y distance between pixels. DOES NOT take into account Z distance (between slides)
        :param pixel1: Pixel, ID's not accepted
        :param pixel2: Pixel, ID's not accepted
        :return: Pythagorean distance between two pixels on the same slide, along the X,Y axes only
        """
        return math.sqrt(math.pow(pixel1.x - pixel2.x, 2) + math.pow(pixel1.y - pixel2.y, 2))
