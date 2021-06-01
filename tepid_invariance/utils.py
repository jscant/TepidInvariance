"""
Some basic helper functions for formatting time and sticking dataframes
together.
"""

import math
import multiprocessing as mp
import shutil
import time
import types

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def truncate_float(x, precision=3):
    """Return input x truncated to <precision> dp."""
    str_x = str(x)
    decimal_pos = str_x.find('.')
    if decimal_pos == -1:
        return float(x)
    after_decimal_value = str_x[decimal_pos + 1:decimal_pos + precision + 1]
    return float(str_x[:decimal_pos] + '.' + after_decimal_value)


def coords_to_string(coords):
    """Return string representation of truncated coordinates."""
    return ' '.join([str(truncate_float(x)) for x in coords])


def print_df(df):
    """Print pandas dataframe in its entirity (with no truncation)."""
    with pd.option_context('display.max_colwidth', None):
        with pd.option_context('display.max_rows', None):
            with pd.option_context('display.max_columns', None):
                print(df)


def no_return_parallelise(func, *args, cpus=-1):
    cpus = mp.cpu_count() if cpus == -1 else cpus
    indices_to_multiply = []
    iterable_len = 1
    args = list(args)
    for idx in range(len(args)):
        if not isinstance(args[idx], (tuple, list, types.GeneratorType)):
            indices_to_multiply.append(idx)
        elif iterable_len == 1:
            iterable_len = len(args[idx])
        elif iterable_len != len(args[idx]):
            raise ValueError('Iterable args must have the same length')
    for idx in indices_to_multiply:
        args[idx] = [args[idx]] * iterable_len

    inputs = list(zip(*args))
    with mp.get_context('spawn').Pool(processes=cpus) as pool:
        pool.starmap(func, inputs)


def condense(arr, gap=100):
    """Condense large arrays into averages over a given window size.

    Arguments:
        arr: numpy array or list of numbers
        gap: size of window over which to average array

    Returns:
        Tuple with new condensed counts (x) and smaller array (y) which is the
        mean of every <gap> values.
    """
    arr = np.array(arr)
    x = np.arange(0, len(arr), step=gap)
    y = np.array([np.mean(arr[n:n + gap]) for n in range(0, len(arr), gap)])
    return x, y


def get_eta(start_time, iters_completed, total_iters):
    """Format time in seconds to hh:mm:ss."""
    time_elapsed = time.time() - start_time
    time_per_iter = time_elapsed / (iters_completed + 1)
    time_remaining = max(0, time_per_iter * (total_iters - iters_completed - 1))
    formatted_eta = format_time(time_remaining)
    return formatted_eta


def format_time(t):
    """Returns string continaing time in hh:mm:ss format.

    Arguments:
        t: time in seconds

    Raises:
        ValueError if t < 0
    """
    if t < 0:
        raise ValueError('Time must be positive.')

    t = int(math.floor(t))
    h = t // 3600
    m = (t - (h * 3600)) // 60
    s = t - ((h * 3600) + (m * 60))
    return '{0:02d}:{1:02d}:{2:02d}'.format(h, m, s)


class Timer:
    """Simple timer class.

    To time a block of code, wrap it like so:

        with Timer() as t:
            <some_code>
        total_time = t.interval

    The time taken for the code to execute is stored in t.interval.
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def print_with_overwrite(*s, spacer=' '):
    """Prints to console, but overwrites previous output, rather than creating
    a newline.

    Arguments:
        s: string (possibly with multiple lines) to print
        spacer: whitespace character to use between words on each line
    """
    s = '\n'.join(
        [spacer.join([str(word) for word in substring]) for substring in s])
    ERASE = '\x1b[2K'
    UP_ONE = '\x1b[1A'
    lines = s.split('\n')
    n_lines = len(lines)
    console_width = shutil.get_terminal_size((0, 20)).columns
    for idx in range(n_lines):
        lines[idx] += ' ' * max(0, console_width - len(lines[idx]))
    print((ERASE + UP_ONE) * (n_lines - 1) + s, end='\r', flush=True)


def plot_with_smoothing(y, gap=100, figsize=(12, 7.5), ax=None):
    """Plot averages with a window given by <gap>."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    plt.cla()
    x, y = condense(y, gap=gap)
    ax.plot(x, y, 'k-')
    return ax
