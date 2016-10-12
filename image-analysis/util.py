from myconfig import Config
import time
import math
import pdb
import glob
import os
from datetime import datetime


def fixpath(path, drive_letter="C", network_drive=False):
    """
    Converts a path string to an absolute path format recognized by both Unix and Windows OS
    If a drive isn't specified, the drive letter is prepended unless the drive is a network drive
    See http://stackoverflow.com/questions/13162372/using-absolute-unix-paths-in-windows-with-python
    :param path: Path string
    :param drive_letter: The drive letter to prepend is a drive letter is missing
    and the location is not a network drive
    :param network_drive: True/False depending on whether location is a network drive
    :return: Formatted absolute path
    """
    assert type(path) is str
    assert type(drive_letter) is str and len(drive_letter) == 1
    assert type(network_drive) is bool
    path = os.path.normpath(os.path.expanduser(path))
    if path.startswith("\\") and not network_drive:
        return drive_letter + ":" + path
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
        """
        Saves the file that the logger has been writing to
        :return:
        """
        self.file.close()

    def flush(self):
        """
        Flushes the contents of the logger by closing and reopening the file it has been writing to
        :return:
        """
        self.close()
        self.file = open(self.log_path, 'a+')

    def generate_log_name(self):
        """
        Generates the file name to be written to by the logger, using the date and configuration from myconfig.py
        :return:
        """
        date = datetime.now().strftime("(%m-%d-%Y)-(%H-%M-%S)")
        if Config.test_instead_of_data:
            data_type = 'Test'
        else:
            data_type = 'Dataset-' + Config.DATA_FILE_PATTERN.replace('*', '(star)')  # Because '*' not allowed in filenames
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
        """
        Writes a str to the log's file
        :param string: The string to be written
        :param end: The termination of the string
        :return:
        """
        if self.nervous:
            self.file = open(self.log_path, 'a+')
            self.file.write(string + end)
            self.file.close()
        else:
            self.file.write(string + end)


def printl(string, end='\n', flush=False):
    """
    Prints to log and stdout
    :param string: The object to be written
    :param end: The suffix of the string
    :param flush: Whether to force-flush the print buffer
    :return:
    """
    assert type(end) is str
    assert type(flush) is bool
    if type(string) is not str:
        string = str(string)
    print(string, end=end, flush=flush)
    if Config.do_logging:
        log.w(string, end=end)


def printd(string, toggle, end='\n', flush=False):
    """
    Prints to stdout/log depending on the value of toggle, otherwise writes to log if
    config says to log everything
    :param toggle: Only prints if toggle is true, or if config says to log everything
    :param string: The object to be written
    :param end: The suffix of the string
    :param flush: Whether to force-flush the print buffer
    :return:
    """
    assert type(end) is str
    assert type(flush) is bool
    if toggle:
        printl(string, end=end, flush=flush)
    else:
        if Config.log_everything:
            log.w(string, end=end)


class ProgressBar:
    """
    A visual aid to keep the user satisfied that a job is still making progress
    This is done by using a counting system
    """
    def __init__(self, start_val=0, min_val=0, max_val=100, increments=10, symbol='.', write_to_log=False):
        """
        :param start_val: The inital progress of the job, relative to min_val & max_val
        :param min_val: The low value value of the counter, if start_val = min_val then no progress has been made
        :param max_val: The terminative value of the counter, progress is complete when the counter reaches this value
        :param increments: The total number of symbols to be printed / the length of the progress bar
        :param symbol: The symbol to be used to fill the progress bar, must be str of len 1
        :param write_to_log: A bool which says whether or not to write the progress bar to the logger
        """
        assert type(start_val) is int
        assert type(min_val) is int
        assert type(max_val) is int and max_val > min_val
        assert min_val <= start_val < max_val
        assert type(increments) is int
        assert type(symbol) is str and len(symbol) == 1
        assert type(write_to_log) is bool
        self.symbol = symbol
        self.last_output = 0
        self.max = max_val
        self.min = min_val
        self.increments = increments
        self.cur_val = 0
        self.symbols_printed = 0
        self.write_to_log = write_to_log
        while start_val - self.last_output and start_val - self.last_output >= (
                    (self.max - self.min) / (1. / self.increments)):
            self.cur_val += ((self.max - self.min) / self.increments)
            self.tick()

    def update(self, new_val, set_val=True):
        """
        Updates the internal counter of ProgressBar
        :param new_val: The value to update with
        :param set_val: If True, set the internal counter to this value
        If False, add this value to the internal counter
        :return:
        """
        assert type(new_val) is int and (not set_val or new_val >= self.cur_val)
        assert type(set_val) is bool
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
        if self.write_to_log:
            printl(self.symbol, end='', flush=True)
        else:
            print(self.symbol, end='', flush=True)
        self.symbols_printed += 1
        self.last_output = self.cur_val

    def finish(self, newline=False):
        """
        All work is done, print any remaining symbols
        :param newline:Whether to terminate string with newline char
        :return:
        """
        assert type(newline) is bool
        symbols_before_finished = self.symbols_printed
        for i in range(self.increments - symbols_before_finished):
            self.tick()
        if newline:
            printl('', flush=True)
        else:
            printl(' ', end='', flush=True)  # Add a space after the ticks


def warn(string):
    """
    Creats a large visual warnign to the user
    :param string: The text to warn the user with
    :return:
    """
    assert type(string) is str and len(string) > 0
    if not Config.disable_warnings:
        print('\n>\n->\n--> WARNING: ' + str(string) + ' <--\n->\n>')


def debug():
    """
    Opens an interactive debugging session for the user to examine what has gone wrong
    :return:
    """
    pdb.set_trace()


def get_images():
    """
    Gets filename of images which are to be processed, based on configuration in myconfig.py
    :return: A list of filenames which contain images
    """
    if Config.test_instead_of_data:
        dir_path = Config.TEST_DIR
        file_name_pattern = Config.TEST_FILE_PATTERN
    else:
        dir_path = Config.DATA_DIR
        file_name_pattern = Config.DATA_FILE_PATTERN
    all_images = glob.glob(dir_path + file_name_pattern)
    if len(all_images) == 0:
        raise Exception('No images found matching the pattern: ' + str(dir_path + file_name_pattern))
    return all_images


def print_elapsed_time(t0, tf, pad='', prefix='Elapsed Time:', endline=True, flush=False):
    """
    Prints how much time has passed between time t0 and tf, which are found using time.time()
    :param t0: Start of time interval
    :param tf: End of time interval (usually current time)
    :param pad: Padding used to offset the message from the left side of the console, usually string of spaces
    :param prefix: Prefix to time string, informs user that the output is an elapsed time
    :param endline: If true, the output terminates with a newline character
    :param flush: Flush the buffer being used to print
    :return:
    """
    assert type(t0) is float
    assert type(tf) is float
    assert type(pad) is str
    assert type(prefix) is str
    assert type(endline) is bool
    assert type(flush) is bool
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
    """
    Returns the current time without spaces, format: Weekday_Month(abbr)_Day_Hour(24h)_Minute_Second_Year
    :return: The current time without spaces
    """
    return time.ctime().replace(' ', '_').replace(':', '-')


def vispy_info():
    """
    Prints vispy's info, which is useful for checking vispy's / the system's setup
    :return:
    """
    import vispy
    printl(vispy.sys_info)


def vispy_tests():
    """
    Run vispy test's to make sure visualization backend is working well - is time consuming
    :return:
    """
    import vispy
    vispy.test()


log = Logger(nervous=Config.nervous_logging)  # TODO clean this up by moving it elsewhere or using Logger directly
