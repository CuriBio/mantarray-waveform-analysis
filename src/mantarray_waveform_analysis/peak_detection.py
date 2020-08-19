# -*- coding: utf-8 -*-
"""Detecting peak and valleys of incoming Mantarray data."""

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from uuid import UUID

from nptyping import NDArray
import numpy as np
from scipy import signal

from .constants import AMPLITUDE_UUID
from .constants import AUC_UUID
from .constants import TWITCH_PERIOD_UUID

# from .constants import WIDTH_UUID

SAMPLING_RESOLUTION = 10000


def peak_detector(
    noise_free_data: NDArray[(2, Any), int],
    sampling_rate: int,
    data: NDArray[(2, Any), int],
) -> Tuple[List[int], List[int]]:
    """Locates peaks and valleys and returns the indices.

    Args:
        noise_free_data: a 2D array of the time and voltage data after it has gone through noise cancellation
        sampling_rate: an integer value of the sampling rate of the data in Hz
        data: a 2D array of the original time and voltage before noise cancellation

    Returns:
        A tuple of the indices of the peaks and valleys
    """
    clean_signal: NDArray[int] = noise_free_data[1, :]

    # find required height of peaks
    max_height = np.max(clean_signal)
    min_height = np.min(clean_signal)

    height = (max_height + min_height) / 2

    # find peaks and valleys
    peakmax, _ = signal.find_peaks(
        clean_signal, height=height, distance=sampling_rate * 0.45
    )
    peakmin, _ = signal.find_peaks(
        (-1 * clean_signal), height=(-1 * height), distance=sampling_rate * 0.45
    )

    # concatenate peaks and valleys
    peak_valley_tuple: Tuple[List[int], List[int]] = (peakmax, peakmin)

    # get exact peaks
    all_peak_valley_ind: NDArray[int] = np.concatenate(
        [peak_valley_tuple[0], peak_valley_tuple[1]]
    )
    all_peak_valley_ind.sort()

    peaks_times = []
    valleys_times = []

    for peak in peak_valley_tuple[0]:
        twitch_ind = list(all_peak_valley_ind).index(peak)
        peaks_times.append(find_x(twitch_ind, all_peak_valley_ind, data, "max"))

    # flip data for valleys
    for valley in peak_valley_tuple[1]:
        valley_ind = list(all_peak_valley_ind).index(valley)
        valleys_times.append(find_x(valley_ind, all_peak_valley_ind, data, "min"))

    peaks_ind = []
    valleys_ind = []

    for times in peaks_times:
        peaks_ind.append(list(noise_free_data[0, :]).index(times))
    for times in valleys_times:
        valleys_ind.append(list(noise_free_data[0, :]).index(times))

    peak_valley_tuple = (peaks_ind, valleys_ind)

    return peak_valley_tuple


def find_x(
    index: int,
    all_peak_valley_ind: NDArray[int],
    data: NDArray[(2, Any), int],
    peak_or_valley: str,
) -> int:
    """Return an int value of the time at which the peak or valley occurs.

    Args:
        index: an int representing the index in all_peak_valley_ind that the peak or valley is
        all_peak_valley_ind: all the indicies of the peaks and valleys
        data: a 2D array of the original time and voltage data
        peak_or_valley: a string to determine if we are finding a peak or valley

    Returns:
        an int that represents the time of the peak or valley
    """
    if index == 0:
        subset = data[:, 0 : all_peak_valley_ind[index + 1]]
        if peak_or_valley == "max":
            max_val = np.max(subset[1, :])
        else:
            max_val = np.min(subset[1, :])
        x_val = np.where(subset[1, :] == max_val)[0][0]
        x_val = subset[:, x_val][0]
    elif index == (len(all_peak_valley_ind) - 1):
        subset = data[:, all_peak_valley_ind[index - 1] : -1]
        if peak_or_valley == "max":
            max_val = np.max(subset[1, :])
        else:
            max_val = np.min(subset[1, :])
        x_val = np.where(subset[1, :] == max_val)[0][0]
        x_val = subset[:, x_val][0]
    else:
        subset = data[
            :, all_peak_valley_ind[index - 1] : all_peak_valley_ind[index + 1],
        ]
        if peak_or_valley == "max":
            max_val = np.max(subset[1, :])
        else:
            max_val = np.min(subset[1, :])
        x_val = np.where(subset[1, :] == max_val)[0][0]
        x_val = subset[:, x_val][0]

    return int(x_val)


def noise_filtering(
    noisy_data: NDArray[(2, Any), int], sampling_rate: int
) -> Tuple[NDArray[(2, Any), int], int]:
    """Remove noise from a signal using a butterworth filter.

    Args:
        noisy_data: a 2D array of the time and voltage data as it was recieved from file writer
        sampling_rate: an integer value of the sampling rate of the data in Hz

    Returns:
        noise_free_data: a 2D array of the time and voltage data after it has gone through noise cancellation
        sampling_rate: an integer value of the sampling rate of the data in Hz
    """
    time: NDArray[int] = noisy_data[0, :]
    cutoff = int(sampling_rate * 0.5)  # Cut-off frequency of the filter
    normalized_freq: float = cutoff / (sampling_rate * 2)  # nyquist criteria
    numer_poly, denom_poly = signal.butter(5, normalized_freq, "low")
    clean_signal: NDArray[int] = signal.filtfilt(
        numer_poly, denom_poly, noisy_data[1, :]
    )

    rounded_data = np.rint(clean_signal).astype(np.int32)
    noise_free_data: NDArray[(2, Any), int] = np.vstack(
        (time, rounded_data.astype(np.int32))
    )

    return noise_free_data, sampling_rate


def time_voltage_dict_creation(
    time: NDArray[1, int],
    clean_signal: NDArray[(2, Any), int],
    peak_valley_tuple: Tuple[int, int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Get time and voltage values using the index of peaks and valleys.

    Args:
        time: a 1D array of the time of the signal
        clean_signal: a 1D array of the voltage after noise filtering
        peak_valley_tuple: a tuple of the indices of the peaks and valleys

    Returns:
        time_voltage_dict_peaks: a dictionary of the time and voltage of each peak
        time_voltage_dict_valleys: a dictionary of the time and voltage of each valley
    """
    peaks: NDArray[int] = peak_valley_tuple[0]
    valleys: NDArray[int] = peak_valley_tuple[1]

    time_voltage_dict_peaks: Dict[int, int] = {}
    time_voltage_dict_valleys: Dict[int, int] = {}

    for i in peaks:
        time_voltage_dict_peaks.update({time[i]: clean_signal[i]})

    for i in valleys:
        time_voltage_dict_valleys.update({time[i]: clean_signal[i]})

    return time_voltage_dict_peaks, time_voltage_dict_valleys


def create_avg_dict(metric: NDArray[int]) -> Dict[str, int]:
    """Calculate the average values of a specific metric.

    Args:
        metric: a 1D array of integer values of a specific metric results

    Returns:
        a dictionary of the average statistics of that metric
    """
    dictionary: Dict[str, int] = {}

    dictionary["n"] = len(metric)
    dictionary["mean"] = int(round(np.mean(metric)))
    dictionary["std"] = int(round(np.std(metric)))
    dictionary["min"] = int(np.min(metric))
    dictionary["max"] = int(np.max(metric))

    return dictionary


def data_metrics(
    peakind: Tuple[int, int], noise_free_data: NDArray[(2, Any), int]
) -> Tuple[
    Dict[int, Dict[UUID, int]],
    Union[Dict[UUID, Dict[int, Dict[str, int]]], Dict[UUID, Dict[str, int]]],
]:
    """Find all data metrics for individual twitches and averages.

    Args:
        peakind: a tuple of integer values representing the indices of peaks and valleys within the data
        noise_free_data: a 2D array of the time and voltage data after it has gone through noise cancellation

    Returns:
        main_twitch_dict: a dictionary of individual peak metrics
        aggregate_dict: a dictionary of entire metric statistics
    """
    # create main dictionaries
    main_twitch_dict: Dict[int, Dict[UUID, int]] = {}
    aggregate_dict = {}

    # create dependent dicitonaries
    period_averages_dict: Dict[str, int] = {}
    amplitude_averages_dict: Dict[str, int] = {}
    auc_averages_dict: Dict[str, int] = {}

    peaks: NDArray[int] = peakind[0]
    valleys: NDArray[int] = peakind[1]

    # find twitch time points
    time_points: NDArray[int] = twitches(peaks, valleys, noise_free_data)
    length_tp: int = len(time_points)

    # find twitch periods
    combined_twitch_periods: NDArray[int] = twitch_period(time_points, length_tp)

    # find aggregate values of period data
    period_averages_dict = create_avg_dict(combined_twitch_periods)

    aggregate_dict[TWITCH_PERIOD_UUID] = period_averages_dict

    # find twitch amplitudes
    all_peak_valley_ind: NDArray[int] = np.concatenate([peaks, valleys])
    all_peak_valley_ind.sort()
    amplitudes: NDArray[int] = calculate_amplitudes(
        time_points, all_peak_valley_ind, noise_free_data
    )

    # find aggregate values of amplitude data
    amplitude_averages_dict = create_avg_dict(amplitudes)

    aggregate_dict[AMPLITUDE_UUID] = amplitude_averages_dict

    # calculate auc
    auc_per_twitch: NDArray[int] = area_under_curve(
        all_peak_valley_ind, time_points, noise_free_data
    )

    # find aggregate values of area under curve data
    auc_averages_dict = create_avg_dict(auc_per_twitch)

    aggregate_dict[AUC_UUID] = auc_averages_dict

    # add metrics to per peak dictionary
    for i in range(length_tp - 1):
        main_twitch_dict.update(
            {
                time_points[i]: {
                    TWITCH_PERIOD_UUID: combined_twitch_periods[i],
                    AMPLITUDE_UUID: amplitudes[i],
                    AUC_UUID: auc_per_twitch[i],
                }
            }
        )

    return main_twitch_dict, aggregate_dict


def twitch_period(time_points: NDArray[int], length_tp: int) -> NDArray[int]:
    """Find the distance between each twitch at its peak.

    Args:
        time_points: a 1D array of the time value of each twitch in centimilliseconds
        length_tp: an int the length of the time_points array

    Returns:
        an array of integers that are the peeriod of each twitch
    """
    period: List[int] = []
    for i in range(length_tp - 1):
        period.append(time_points[i + 1] - time_points[i])

    return period


def twitches(
    peaks: NDArray[int], valleys: NDArray[int], noise_free_data: NDArray[(2, Any), int],
) -> NDArray[int]:
    """Get time points of all the twitch peaks.

    Args:
        peaks: a 1D array of integers representing the indices of the peaks
        valleys: a 1D array of integers representing the indices of the valleys
        noise_free_data: a 2D array of the data after being noise cancelled

    Returns:
        a 1D array of integers representing the time points of all the twitches
    """
    time_points: NDArray[int] = []
    length_valley: int = len(valleys)
    length_peak: int = len(peaks)

    for i in range(length_valley - 1):
        for j in range(length_peak):
            if peaks[j] > valleys[i]:
                if peaks[j] < valleys[i + 1]:
                    time_points.append(int(noise_free_data[0, :][peaks[j]]))

    return time_points


def calculate_amplitudes(
    time_points: NDArray[int],
    all_peak_valley_ind: NDArray[int],
    noise_free_data: NDArray[(2, Any), int],
) -> NDArray[int]:
    """Get the amplitudes for all twitches.

    Args:
        time_points: a 1D array of all the time values of the peaks of interest
        all_peak_valley_ind: a 1D array of all the indices of the peaks and valleys
        noise_free_data: a 2D array of the time and voltage data after it has gone through noise cancellation

    Returns:
        a 1D array of integers representing the amplitude of each twitch
    """
    amplitudes: NDArray[int] = []

    peakind = []
    for times in time_points:
        peakind.append(list(noise_free_data[0, :]).index(times))

    for peak in peakind:
        twitch_ind = list(all_peak_valley_ind).index(peak)

        start_point = all_peak_valley_ind[(twitch_ind - 1)]
        mid_point = all_peak_valley_ind[twitch_ind]
        end_point = all_peak_valley_ind[(twitch_ind + 1)]

        pre_twitch_vol = noise_free_data[1, start_point]
        post_twitch_vol = noise_free_data[1, end_point]
        twitch_vol = noise_free_data[1, mid_point]

        amplitudes.append(
            int(((twitch_vol - pre_twitch_vol) + (twitch_vol - post_twitch_vol)) / 2)
        )

    return amplitudes


def area_under_curve(
    all_peak_valley_ind: NDArray[int],
    time_points: NDArray[int],
    noise_free_data: NDArray[(2, Any), int],
) -> NDArray[int]:
    """Calculate the area under the curve of each twitch.

    Args:
        all_peak_valley_ind: a 1D array of all the indices of the peaks and valleys
        time_points: a 1D array of the time value of each twitch in centimilliseconds
        noise_free_data: a 2D array of the time and voltage data after it has gone through noise cancellation

    Returns:
        a 1D array of integers representing the area under the curve of each twitch
    """
    auc_per_twitch: NDArray[int] = []

    peakind = []
    for times in time_points:
        peakind.append(list(noise_free_data[0, :]).index(times))

    for peak in peakind:
        twitch_ind = list(all_peak_valley_ind).index(peak)

        start_point = all_peak_valley_ind[(twitch_ind - 1)]
        end_point = all_peak_valley_ind[(twitch_ind + 1)]

        # equation of the line between start and end point (y = m * x + b)
        # slope (m) = y2 - y1 / x2 - x1
        slope: float = (
            noise_free_data[1, end_point] - noise_free_data[1, start_point]
        ) / (noise_free_data[0, end_point] - noise_free_data[0, start_point])
        # y-intercept (b) = y - m * x
        y_intercept: float = noise_free_data[1, end_point] - (
            slope * noise_free_data[0, end_point]
        )

        y_vals = []
        for i in range(start_point, end_point):
            y_curr = noise_free_data[1, i] - (
                (slope * noise_free_data[0, i]) + y_intercept
            )
            y_vals.append(y_curr)

        area = np.trapz(y_vals, x=noise_free_data[0, start_point:end_point])

        auc_per_twitch.append(int(area))

    return auc_per_twitch


def find_nearest(array: NDArray[float], value: float) -> float:
    """Find the value that is nearest to a desired value in an array.

    Args:
        array: a 1D array that will be paresed through the find nearest value
        value: the value that you are comparing to find one nearest to

    Returns:
        an float value of the nearest value in the array to the desired value
    """
    array = np.asarray(array)
    idx: int = (np.abs(array - value)).argmin()
    nearest_value: float = array[idx]
    return nearest_value


def twitch_widths(
    peakind: Tuple[NDArray[int], NDArray[int]], noise_free_data: NDArray[(2, Any), int],
) -> Dict[int, Dict[int, Any]]:
    """Determine twitch width between 10-90% down to the nearby valleys.

    Args:
        peakind: a Tuple of an Array of the ints for the peaks followed by an array of ints for the valleys
        noise_free_data: a 2D array of the time and voltage data after it has gone through noise cancellation

    Returns:
        a dictionary representing the width between the left and right side of a twitch a certain percentage of the way down to the nearby valleys
    """
    # initialize final dictionary
    widths_dict: Dict[int, Dict[int, Any]] = {}

    # set up data
    time_points: NDArray[int] = twitches(peakind[0], peakind[1], noise_free_data)
    all_peak_valley_ind: NDArray[int] = np.concatenate([peakind[0], peakind[1]])
    all_peak_valley_ind.sort()

    peak_indicies = []
    for times in time_points:
        peak_indicies.append(list(noise_free_data[0, :]).index(times))

    for peak in peak_indicies[:-1]:
        twitch_ind = list(all_peak_valley_ind).index(peak)

        # initialize percentage dict
        percent_dict: Dict[int, Any] = {}

        for i in range(10, 95, 5):
            # make subset of the data
            start_point = int(all_peak_valley_ind[(twitch_ind - 1)])
            mid_point = int(all_peak_valley_ind[twitch_ind])
            end_point = int(all_peak_valley_ind[(twitch_ind + 1)])

            # get amplitudes of this twitch
            amplitude_left: int = noise_free_data[1, mid_point] - noise_free_data[
                1, start_point
            ]
            amplitude_right: int = noise_free_data[1, mid_point] - noise_free_data[
                1, end_point
            ]

            subset_data_left_of_twitch = noise_free_data[:, start_point:mid_point]
            subset_data_right_of_twitch = noise_free_data[:, mid_point:end_point]

            # get y value of desired point
            y_val_left: float = noise_free_data[1, mid_point] - (
                amplitude_left * (i / 100)
            )
            y_val_right: float = noise_free_data[1, mid_point] - (
                amplitude_right * (i / 100)
            )

            # find nearest y point in subset to the desired y_val
            y_left: float = find_nearest(subset_data_left_of_twitch[1, :], y_val_left)
            y_right: float = find_nearest(
                subset_data_right_of_twitch[1, :], y_val_right
            )

            # find corresponding x values for those y values
            x_left = np.where(subset_data_left_of_twitch == y_left)[1][0]
            x_left = subset_data_left_of_twitch[:, x_left][0]
            x_right = np.where(subset_data_right_of_twitch == y_right)[1][0]
            x_right = subset_data_right_of_twitch[:, x_right][0]

            # calculate distance between the points as the width
            distance = int(x_right - x_left)

            # add values to dictionaries
            percent_dict[i] = [[x_left, y_left], [x_right, y_right], distance]

        widths_dict[int(noise_free_data[0, peak])] = percent_dict

    return widths_dict


def compressed_peak_detector(
    compressed_data: NDArray[(2, Any), int],
    peakind: Tuple[NDArray[int], NDArray[int]],
    noise_free_data: NDArray[(2, Any), int],
) -> Tuple[NDArray[int], NDArray[int]]:
    """Make the peaks/valleys of the compresssed data equal to the original.

    Args:
        compressed_data: a 2D array of the time and voltage daata after compression
        peakind: a tuple of peaks and valley indicies of the original data
        noise_free_data: the original data that the peakind was found on

    Returns:
        a tuple of peak and valley indicies of the new compressed data
    """
    peaks: NDArray[int] = []
    valleys: NDArray[int] = []

    for peak in peakind[0]:
        result = find_nearest(compressed_data[0, :], noise_free_data[0, peak])
        idx = np.where((compressed_data[0, :]) == result)
        peaks.append(idx[0][0])
    for valley in peakind[1]:
        result = find_nearest(compressed_data[0, :], noise_free_data[0, valley])
        idx = np.where((compressed_data[0, :]) == result)
        valleys.append(idx[0][0])

    peak_valley_tuple = (peaks, valleys)

    return peak_valley_tuple


def convert_voltage_to_displacement(
    data: NDArray[(2, Any), int]
) -> NDArray[(2, Any), int]:
    """Flip the data so the twitches are pointing upwards.

    Args:
        data: the original 2d array of time vs voltage data

    Returns:
        a 2d array of the time vs voltage data after voltage has been flipped
    """
    voltage = data[1, :]
    voltage_length = len(voltage)

    for i in range(voltage_length):
        voltage[i] = -voltage[i]

    data[1, :] = voltage

    return data
