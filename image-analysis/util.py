from myconfig import config
import time
import pickle
import math
import pdb
import glob
import sys

class progressBar:
    def __init__(self, start_val=0, min_val=0, max_val=100, increments=10, symbol='.'):
        assert len(symbol) == 1
        assert start_val <= max_val
        self.symbol = symbol
        self.last_output = 0
        self.max = max_val
        self.min = min_val
        self.increments = increments
        self.cur_val =  0
        self.symbols_printed = 0
        # print('Db created progress bar, max_val = ' + str(max_val))
        # print('Start val = ' + str(start_val))

        while start_val - self.last_output and start_val - self.last_output >= ((self.max - self.min) /( 1. /  self.increments)):
            # print('Top of while, self.cur_val = ' + str(self.cur_val) + ', last output: ' + str(self.last_output))
            # print('Checked ' + str(start_val - self.last_output) + ' >= ' + str(((self.max - self.min) /( 1. /  self.increments))))
            self.cur_val += ((self.max - self.min) / self.increments)
            # print('Updated to ' + str(self.cur_val))
            self.tick()

    def update(self, new_val, set=True):
        '''
        Updates the internal counter of progressBar
        :param new_val: The value to update with
        :param set: If True, set the internal counter to this value
                    If False, add this value to the internal counter
        :return:
        '''
        if not set: # Then add
            new_val = new_val + self.cur_val
        while new_val - self.last_output >= ((self.max - self.min) / self.increments):
            self.cur_val += ((self.max - self.min) / self.increments)
            self.tick()
        self.cur_val = new_val

    def tick(self):
        '''
        Prints one more symbol to indicate passing another interval
        :return:
        '''
        print(self.symbol, end='', flush=True)
        self.symbols_printed += 1
        self.last_output = self.cur_val

    def finish(self, newline=False):
        '''
        All work is done, print any remaining symbols
        :return:
        '''
        symbols_before_finished = self.symbols_printed
        for i in range(self.increments - symbols_before_finished):
            self.tick()
        if newline:
            print('', flush=True)
        else:
            print(' ', end='', flush=True) # Add a space after the ticks


def warn(string):
    if not config.disable_warnings:
        print('\n>\n->\n--> WARNING: ' + str(string) + ' <--\n->\n>')

def debug():
    pdb.set_trace()


def getImages():
    if config.test_instead_of_data:
        dir = config.TEST_DIR
        extension = '*.png'
    else:
        dir = config.DATA_DIR
        if config.swell_instead_of_c57bl6:
            extension = 'Swell*.tif'
        else:
            extension = 'C57BL6*.tif'
    all_images = glob.glob(dir + extension)
    return all_images


def printElapsedTime(t0, tf, pad='', prefix='Elapsed Time:', endline=True):
    temp = tf - t0
    m = math.floor(temp / 60)
    plural_minutes = ''
    if endline:
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


def vispy_info():
    import vispy
    print(vispy.sys_info)

def vispy_tests():
    import vispy
    vispy.test()