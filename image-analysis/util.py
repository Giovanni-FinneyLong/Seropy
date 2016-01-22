from myconfig import *
import time
import pickle
import math
import pdb
import glob
import sys

def warn(string):
    print('\n>\n->\n--> WARNING: ' + str(string) + ' <--\n->\n>')

def debug():
    pdb.set_trace()


def getImages():
    if test_instead_of_data:
        dir = TEST_DIR
        extension = '*.png'
    else:
        dir = DATA_DIR
        extension = '*.tif'
    all_images = glob.glob(dir + extension)
    return all_images


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

def timeNoSpaces():
    return time.ctime().replace(' ', '_').replace(':', '-')

def progressBarUpdate(value, max, min=0, last_update=0, steps=10):
    ''' # TODO not functional
    Run like so:
    updateStatus = 0
    for num in range(100):
        updateStatus = progressBarUpdate(num, 100, last_update=updateStatus)
    :param value:
    :param max:
    :param min:
    :param last_update:
    :param steps:
    :return:
    '''
    if value == min:
        print('.', end='')
    else:
        # print('DB last_update=' + str(last_update) + ' val=' + str(value))
        # print(str((value - last_update)) + ' vs ' + str(((max-min) / steps)))
        if last_update < max:
            if (value - last_update) >= ((max-min) / steps):
                last_update = value;
                print('Diff=' + str((value - last_update)) + ' stepsize:' + str(((max-min) / steps)))
                print('Val' + str(value))
                # for i in range( math.ceil((value - last_update) / ((max-min) / steps))):
                print('.', end='')
        # if value >= max:
        #     print('', end='\n')
    return last_update



def vispy_info():
    import vispy
    print(vispy.sys_info)

def vispy_tests():
    import vispy
    vispy.test()