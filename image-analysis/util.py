from myconfig import Config
import time
import math
import pdb
import glob
import os
from datetime import datetime


def fixpath(path):  # http://stackoverflow.com/questions/13162372/using-absolute-unix-paths-in-windows-with-python
    path = os.path.normpath(os.path.expanduser(path))
    if path.startswith("\\"):
        return "C:" + path
    return path


class Logger:
    def __init__(self, nervous=True):
        """
        Logs strings as requested
        :param nervous: Whether the logger should open and close the file before and after every write
                        A non-nervous logger must be closed after all writing is done! (To save the file)
        :return:
        """
        cur_dir = fixpath(os.getcwd())
        above_dir = cur_dir[:max(cur_dir.rfind('/'), cur_dir.rfind('\\'))]
        self.dir = fixpath(above_dir + '/logs')
        self.nervous = nervous
        if not os.path.exists(self.dir):
            print('Log Directory DNE, so making one at: ' + str(self.dir))
            os.makedirs(self.dir)
        self.generate_log_name()
        if not self.nervous:
            self.file = open(self.log_path, 'a+')

    def close(self):
        self.file.close()

    def flush(self):
        self.close()
        self.file = open(self.log_path, 'a+')

    def generate_log_name(self):
        date = datetime.now().strftime("(%m-%d-%Y)-(%H-%M-%S)")
        if Config.test_instead_of_data:
            data_type = 'Test'
        else:
            data_type = 'Dataset-'
            if Config.swell_instead_of_c57bl6:
                data_type += 'Swell'
            else:
                data_type += 'C57B16'
        if Config.base_b3ds_with_stitching:
            data_type += '-Stitched'
        else:
            data_type += '-Nonstitched'
        if Config.process_internals:
            bloom = 'Bloomed'
            if Config.stitch_bloomed_b2ds:
                bloom += '-Stitched'
            else:
                bloom += '-Nonstitched'
        else:
            bloom = 'Nonbloomed'
        if Config.dePickle:
            save_or_load = 'Load'
        else:
            save_or_load = 'Save'

        self.log_name = date + '_' + data_type + '_' + bloom + '_' + save_or_load + '.log'
        self.log_path = fixpath(self.dir + '/' + self.log_name)
        print('Log path: ' + str(self.log_path))

    def w(self, string, end='\n'):
        if self.nervous:
            self.file = open(self.log_path, 'a+')
            self.file.write(string + end)
            self.file.close()
        else:
            self.file.write(string + end)


log = Logger(nervous=Config.nervous_logging)  # TODO clean this up by moving it elsewhere or using Logger directly


def printl(string, end='\n', flush=False):
    """
    Prints to log and stdout
    :param string: The object to be written
    :param end: The suffix of the string
    :param flush: Whether to force-flush the print buffer
    :return:
    """
    if type(string) is not str:
        string = str(string)
    print(string, end=end, flush=flush)
    if Config.do_logging:
        log.w(string, end=end)


def printd(string, toggle, end='\n', flush=False):
    """
    Prints to stdout depending on the value of toggle, writes to log regardless
    :param toggle:
    :param string: The object to be written
    :param end: The suffix of the string
    :param flush: Whether to force-flush the print buffer
    :return:
    """
    if toggle:
        printl(string, end=end, flush=flush)
    else:
        if Config.log_everything:
            log.w(string, end=end)


class ProgressBar:
    def __init__(self, start_val=0, min_val=0, max_val=100, increments=10, symbol='.', log=False):
        assert len(symbol) == 1
        assert start_val <= max_val
        self.symbol = symbol
        self.last_output = 0
        self.max = max_val
        self.min = min_val
        self.increments = increments
        self.cur_val = 0
        self.symbols_printed = 0
        self.log = log

        while start_val - self.last_output and start_val - self.last_output >= (
                    (self.max - self.min) / (1. / self.increments)):
            self.cur_val += ((self.max - self.min) / self.increments)
            self.tick()

    def update(self, new_val, set_val=True):
        """
        Updates the internal counter of ProgressBar
        :param new_val: The value to update with
        :param set: If True, set the internal counter to this value
                    If False, add this value to the internal counter
        :return:
        """
        # printl(" DB updating progress bar, curval:" + str(self.cur_val) + ' new_val: ' + str(new_val) + ' set: ' + str(set) + ' maxval: ' + str(self.max))
        if not set_val:  # Then add
            new_val = new_val + self.cur_val
        while new_val - self.last_output >= ((self.max - self.min) / self.increments):
            self.cur_val += ((self.max - self.min) / self.increments)
            self.tick()
        self.cur_val = new_val

    def tick(self):
        """
        Prints one more symbol to indicate passing another interval
        :return:
        """
        if self.log:
            printl(self.symbol, end='', flush=True)
        else:
            print(self.symbol, end='', flush=True)
        self.symbols_printed += 1
        self.last_output = self.cur_val

    def finish(self, newline=False):
        """
        All work is done, print any remaining symbols
        :param newline:
        :return:
        """
        symbols_before_finished = self.symbols_printed
        for i in range(self.increments - symbols_before_finished):
            self.tick()
        if newline:
            printl('', flush=True)
        else:
            printl(' ', end='', flush=True)  # Add a space after the ticks


def warn(string):
    if not Config.disable_warnings:
        print('\n>\n->\n--> WARNING: ' + str(string) + ' <--\n->\n>')


def debug():
    pdb.set_trace()


def get_images():
    if Config.test_instead_of_data:
        dir_path = Config.TEST_DIR
        extension = '*.png'
    else:
        dir_path = Config.DATA_DIR
        if Config.swell_instead_of_c57bl6:
            extension = 'Swell*.tif'
        else:
            extension = 'C57BL6*.tif'
    all_images = glob.glob(dir_path + extension)
    return all_images


def print_elapsed_time(t0, tf, pad='', prefix='Elapsed Time:', endline=True, flush=False):
    temp = tf - t0
    m = math.floor(temp / 60)
    plural_minutes = ''
    if endline:
        end = '\n'
    else:
        end = ''

    if m > 1:
        plural_minutes = 's'
    if m > 0:
        printl(pad + prefix + ' ' + str(m) + ' minute' + str(plural_minutes) + ' & %.0f seconds' % (temp % 60), end=end, flush=flush)
    else:
        printl(pad + prefix + ' %.2f seconds' % (temp % 60), end=end, flush=flush)


def time_no_spaces():
    return time.ctime().replace(' ', '_').replace(':', '-')


def vispy_info():
    import vispy
    printl(vispy.sys_info)


def vispy_tests():
    import vispy
    vispy.test()
