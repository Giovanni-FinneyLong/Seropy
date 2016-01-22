from myconfig import *
from Pixel import Pixel
from Blob2d import Blob2d

import time
import pickle
import math
import pdb

def warn(string):
    print('\n>\n->\n--> WARNING: ' + str(string) + ' <--\n->\n>')

def debug():
    pdb.set_trace()

# @profile
def save(blob3dlist, filename, directory=PICKLEDIR):
    if directory != '':
        if directory[-1] not in ['/', '\\']:
            slash = '/'
        else:
            slash = ''
    filename = directory + slash + filename
    print('Saving to pickle:'+ str(filename))
    done = False
    while not done:
        try:
            print('Pickling ' + str(len(blob3dlist)) + ' b3ds')
            t = time.time()
            pickle.dump({'b3ds' : blob3dlist}, open(filename + '_b3ds', "wb"), protocol=0)
            printElapsedTime(t,time.time())

            print('Pickling ' + str(len(Blob2d.all)) + ' b2ds')
            t = time.time()
            pickle.dump({'b2ds' : Blob2d.all, 'used_ids': Blob2d.used_ids}, open(filename + '_b2ds', "wb"), protocol=0)
            printElapsedTime(t,time.time())

            print('Pickling ' + str(len(Pixel.all)) + ' pixels from the total possible ' + str(Pixel.total_pixels))
            t = time.time()
            pickle.dump({'pixels' : Pixel.all, 'total_pixels' : Pixel.total_pixels}, open(filename + '_pixels', "wb"), protocol=0)
            printElapsedTime(t,time.time())
            done = True
        except RuntimeError:
            print('\nIf recursion depth has been exceeded, you may increase the maximal depth with: sys.setrecursionlimit(<newdepth>)')
            print('The current max recursion depth is: ' + str(sys.getrecursionlimit()))
            print('Opening up an interactive console, press \'n\' then \'enter\' to load variables before interacting, and enter \'exit\' to resume execution')
            debug()
            pass

# @profile
def load(filename, directory=PICKLEDIR):
        if directory[-1] not in ['/', '\\']:
            slash = '/'
        else:
            slash = ''
        filename = directory + slash + filename
        t_start = time.time()
        print('Loading from pickle:' + str(filename))
        print('Loading b3ds ', end='',flush=True)
        t = time.time()
        b3ds = pickle.load(open(filename + '_b3ds', "rb"))['b3ds']
        printElapsedTime(t, time.time())
        print('Loading b2ds ', end='',flush=True)
        t = time.time()

        buff = pickle.load(open(filename + '_b2ds', "rb"))
        Blob2d.all = buff['b2ds']
        Blob2d.used_ids = buff['used_ids']
        Blob2d.total_blobs = len(Blob2d.all)
        printElapsedTime(t, time.time())
        print('Loading pixels ', end='',flush=True)
        t = time.time()
        buff = pickle.load(open(filename + '_pixels', "rb"))
        Pixel.all = buff['pixels']
        Pixel.total_pixels = len(Pixel.all)
        printElapsedTime(t, time.time())

        print('There are a total of:' + str(len(b3ds)) + ' b3ds')
        print('There are a total of:' + str(len(Blob2d.all)) + ' b2ds')
        print('There are a total of:' + str(len(Pixel.all)) + ' pixels')
        print('Total to unpickle: ', end='')
        printElapsedTime(t_start, time.time())
        return b3ds


def printElapsedTime(t0, tf, pad='', prefix='Elapsed Time:', endLine=True):
    temp = tf - t0
    m = math.floor(temp / 60)
    plural_minutes = ''
    if endLine:
        end='\n'
    else:
        end=''

    if m > 1:
        plural_minutes = 's'
    if m > 0:
        print(pad + prefix + ' ' + str(m) + ' minute' + str(plural_minutes) + ' & %.0f seconds' % (temp % 60), end=end)
    else:
        print(pad + prefix + ' %.2f seconds' % (temp % 60), end=end)

def vispy_info():
    import vispy
    print(vispy.sys_info)

def vispy_tests():
    import vispy
    vispy.test()