#cython: language_level=3
#cython: linetrace=True
# Eli (8/18/20) ... not sure why cython doesn't default to compiling with Python 3...but apparently this is explicitly needed https://github.com/cython/cython/issues/2299
"""Compressions arrays of Mantarray GMR data ."""
from typing import Any

from nptyping import NDArray
import numpy as np

R_SQUARE_CUTOFF = 0.95

def rsquared(int [:] x_values, int [:] y_values):
    """Return R^2 where x and y are array-like.

    Instead of doing a full linear regression, the compression process drops all points in between the first and the last, so R^2 residuals are calculated based off of the line between the first and last points.
    Typically the time and filtered GMR readings are supplied as the x and y axis values respectively.

    Args:
        x_values: 1D array of int values representing time
        y_values: 1D array of int values representing filtered GMR

    Returns:
        the R^2 value of the given dataset
    """
    cdef int x_0, x_1, y_0, y_1
    cdef float slope, intercept, ss_res, ss_tot, y_bar


    x_0 = x_values[0]
    x_1 = x_values[-1]
    y_0 = y_values[0]
    y_1 = y_values[-1]
    slope = (y_1 - y_0) / (x_1 - x_0)
    intercept = -slope * x_1 + y_1

    # based on https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    ss_res = 0
    ss_tot = 0
    cdef int y_sum,num_values,i
    y_sum=0
    num_values=y_values.shape[0]
    ss_res=0
    for i in range(num_values):
        y_sum += y_values[i]
        ss_res += (x_values[i] * slope + intercept - y_values[i]) ** 2
    y_bar = y_sum / num_values

    ss_tot = 0
    for i in range(num_values):
        ss_tot += (y_values[i] - y_bar) ** 2

    return 1 - ss_res / ss_tot


def compress_filtered_gmr(data: NDArray[(2, Any), int]) -> NDArray[(2, Any), int]:
    """Compress the data to allow for better plotting in the desktop app.

    Args:
        data: a 2D array of GMR data after noise filtering

    Returns:
        a 2D array containing slightly less points than the original data
    """
    # split time and GMR readings into individual arrays
    time: NDArray[int] = data[0, :]
    filtered_gmr: NDArray[int] = data[1, :]

    # create a boolean array of indicies that will be kept
    what_to_keep = [True] * len(time)

    # loop through values in time and filtered_gmr to determine what to compress
    left_idx = 0
    cdef int [:] subset_time
    cdef int [:] subset_filtered_gmr

    while left_idx < (len(time) - 2):
        right_idx = left_idx + 3  # create an initial subset of length 3

        subset_time = time[left_idx:(right_idx)]
        subset_filtered_gmr = filtered_gmr[left_idx:(right_idx)]

        # calculate r_squared value of the initial length 3 subset
        r_squared = rsquared(subset_time, subset_filtered_gmr)

        if r_squared > R_SQUARE_CUTOFF:
            # what_to_keep[left_idx + 1] = True
            while r_squared > R_SQUARE_CUTOFF and right_idx < (len(time) - 1):
                what_to_keep[right_idx - 2] = False
                # add another point into the subset
                right_idx += 1
                subset_time = time[left_idx:(right_idx)]
                subset_filtered_gmr = filtered_gmr[left_idx:(right_idx)]

                # re-calculate the new r_squared
                r_squared = rsquared(subset_time, subset_filtered_gmr)

            left_idx = right_idx - 1

        else:
            left_idx += 1

    # compress the arrays
    compressed_time = np.compress(what_to_keep, time)
    compressed_filtered_gmr = np.compress(what_to_keep, filtered_gmr)

    # combine the arrays back into proper format
    compressed_time = np.array(compressed_time, dtype=np.int32)
    compressed_filtered_gmr = np.array(compressed_filtered_gmr, dtype=np.int32)

    len_compress = len(compressed_time)

    compressed_data = np.zeros((2, len_compress), dtype=np.int32)
    for i in range(len_compress):
        compressed_data[0, i] = compressed_time[i]
        compressed_data[1, i] = compressed_filtered_gmr[i]

    return compressed_data
