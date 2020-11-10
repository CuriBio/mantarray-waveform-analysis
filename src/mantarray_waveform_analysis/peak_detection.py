# -*- coding: utf-8 -*-
"""Detecting peak and valleys of incoming Mantarray data."""

from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from uuid import UUID

from nptyping import NDArray
import numpy as np
from scipy import signal

from .constants import AMPLITUDE_UUID
from .constants import AUC_UUID
from .constants import CENTIMILLISECONDS_PER_SECOND
from .constants import MIN_NUMBER_PEAKS
from .constants import MIN_NUMBER_VALLEYS
from .constants import PRIOR_PEAK_INDEX_UUID
from .constants import PRIOR_VALLEY_INDEX_UUID
from .constants import SUBSEQUENT_PEAK_INDEX_UUID
from .constants import SUBSEQUENT_VALLEY_INDEX_UUID
from .constants import TWITCH_FREQUENCY_UUID
from .constants import TWITCH_PERIOD_UUID
from .constants import WIDTH_FALLING_COORDS_UUID
from .constants import WIDTH_RISING_COORDS_UUID
from .constants import WIDTH_UUID
from .constants import WIDTH_VALUE_UUID
from .exceptions import TooFewPeaksDetectedError
from .exceptions import TwoPeaksInARowError
from .exceptions import TwoValleysInARowError

TWITCH_WIDTH_PERCENTS = range(10, 95, 5)


def peak_detector(
    filtered_gmr: NDArray[(2, Any), int],
    twitches_point_up: bool = True,
) -> Tuple[List[int], List[int]]:
    """Locates peaks and valleys and returns the indices.

    Args:
        noise_free_data: a 2D array of the time and voltage data after it has gone through noise cancellation
        sampling_rate: an integer value of the sampling rate of the data in Hz
        twitches_point_up: whether in the incoming data stream the biological twitches are pointing up (in the positive direction) or down
        data: a 2D array of the original time and voltage before noise cancellation

    Returns:
        A tuple of the indices of the peaks and valleys
    """
    gmr_signal: NDArray[int] = filtered_gmr[1, :]
    peak_invertor_factor = 1
    valley_invertor_factor = -1
    if not twitches_point_up:
        peak_invertor_factor *= -1
        valley_invertor_factor *= -1
    sampling_period_cms = filtered_gmr[0, 1] - filtered_gmr[0, 0]
    maximum_possible_twitch_frequency = 7  # pylint:disable=invalid-name # (Eli 9/1/20): I can't think of a shorter name to describe this concept fully # Hz
    minimum_required_samples_between_twitches = int(  # pylint:disable=invalid-name # (Eli 9/1/20): I can't think of a shorter name to describe this concept fully
        round(
            (1 / maximum_possible_twitch_frequency)
            * CENTIMILLISECONDS_PER_SECOND
            / sampling_period_cms,
            0,
        )
    )

    # find required height of peaks
    max_height = np.max(gmr_signal)
    min_height = np.min(gmr_signal)
    max_prominence = abs(max_height - min_height)
    # find peaks and valleys
    peak_indices, _ = signal.find_peaks(
        gmr_signal * peak_invertor_factor,
        width=minimum_required_samples_between_twitches / 2,
        distance=minimum_required_samples_between_twitches,
        prominence=max_prominence / 4,
    )
    valley_indices, properties = signal.find_peaks(
        gmr_signal * valley_invertor_factor,
        width=minimum_required_samples_between_twitches / 2,
        distance=minimum_required_samples_between_twitches,
        prominence=max_prominence / 4,
    )
    left_valley_bases = properties["left_bases"]
    right_valley_bases = properties["right_bases"]
    # TODO Tanner (11/3/20): move this loop to find_twitch_indices
    for i in range(1, len(valley_indices)):
        if (
            left_valley_bases[i] == left_valley_bases[i - 1]
            and right_valley_bases[i] == right_valley_bases[i - 1]
        ):
            valley_indices = np.delete(valley_indices, i)
            i -= 1

    return peak_indices, valley_indices


def create_avg_dict(
    metric: NDArray[int], round_to_int: bool = True
) -> Dict[str, Union[float, int]]:
    """Calculate the average values of a specific metric.

    Args:
        metric: a 1D array of integer values of a specific metric results

    Returns:
        a dictionary of the average statistics of that metric
    """
    dictionary: Dict[str, Union[float, int]] = dict()

    dictionary["n"] = len(metric)
    dictionary["mean"] = np.mean(metric)
    dictionary["std"] = np.std(metric)
    dictionary["min"] = np.min(metric)
    dictionary["max"] = np.max(metric)
    if round_to_int:
        for iter_key in ("mean", "std", "min", "max"):
            dictionary[iter_key] = int(round(dictionary[iter_key]))

    return dictionary


def data_metrics(
    peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
    filtered_data: NDArray[(2, Any), int],
) -> Tuple[
    Dict[int, Dict[UUID, Union[float, int]]],
    Dict[
        UUID,
        Union[Dict[str, Union[float, int]], Dict[int, Dict[str, Union[float, int]]]],
    ],
]:  # pylint:disable=too-many-locals # Eli (9/8/20): there are a lot of metrics to calculate that need local variables
    """Find all data metrics for individual twitches and averages.

    Args:
        peakind: a tuple of integer values representing the time indices of peaks and valleys within the data
        filtered_data: a 2D array of the time and voltage data after it has gone through noise cancellation

    Returns:
        per_twitch_dict: a dictionary of individual peak metrics
        aggregate_dict: a dictionary of entire metric statistics. Most metrics have the stats underneath the UUID, but for twitch widths, there is an additional dictionary where the percent of repolarization is the key
    """
    # create main dictionaries
    main_twitch_dict: Dict[int, Dict[UUID, Union[float, int]]] = dict()
    aggregate_dict: Dict[
        UUID,
        Union[Dict[str, Union[float, int]], Dict[int, Dict[str, Union[float, int]]]],
    ] = dict()

    # create dependent dicitonaries
    period_averages_dict: Dict[str, Union[float, int]] = {}
    amplitude_averages_dict: Dict[str, Union[float, int]] = {}
    auc_averages_dict: Dict[str, Union[float, int]] = {}

    peak_indices, _ = peak_and_valley_indices

    # find twitch time points

    twitch_indices = find_twitch_indices(peak_and_valley_indices, filtered_data)
    num_twitches = len(twitch_indices)
    time_series = filtered_data[0, :]

    # find twitch periods
    combined_twitch_periods = calculate_twitch_period(
        twitch_indices, peak_indices, filtered_data
    )

    twitch_frequencies = 1 / (
        combined_twitch_periods.astype(np.float) / CENTIMILLISECONDS_PER_SECOND
    )
    frequency_averages_dict = create_avg_dict(twitch_frequencies, round_to_int=False)
    # find aggregate values of period data
    period_averages_dict = create_avg_dict(combined_twitch_periods)

    aggregate_dict[TWITCH_PERIOD_UUID] = period_averages_dict
    aggregate_dict[TWITCH_FREQUENCY_UUID] = frequency_averages_dict

    # find twitch amplitudes
    amplitudes: NDArray[int] = calculate_amplitudes(twitch_indices, filtered_data)

    # find aggregate values of amplitude data
    amplitude_averages_dict = create_avg_dict(amplitudes)

    aggregate_dict[AMPLITUDE_UUID] = amplitude_averages_dict

    # find twitch widths
    widths = calculate_twitch_widths(twitch_indices, filtered_data)
    width_stats_dict: Dict[int, Dict[str, Union[float, int]]] = dict()

    for iter_percent in TWITCH_WIDTH_PERCENTS:
        iter_list_of_width_values: List[Union[float, int]] = []
        for iter_twitch in widths:
            iter_width_value = iter_twitch[iter_percent][WIDTH_VALUE_UUID]
            if not isinstance(iter_width_value, (float, int)):  # making mypy happy
                raise NotImplementedError(
                    f"The width value under key {WIDTH_VALUE_UUID} must be a float or an int. It was: {iter_width_value}"
                )
            iter_list_of_width_values.append(iter_width_value)
        iter_stats_dict = create_avg_dict(iter_list_of_width_values)
        width_stats_dict[iter_percent] = iter_stats_dict

    aggregate_dict[WIDTH_UUID] = width_stats_dict
    # calculate auc
    auc_per_twitch: NDArray[int] = calculate_area_under_curve(
        twitch_indices, filtered_data, widths
    )
    # find aggregate values of area under curve data
    auc_averages_dict = create_avg_dict(auc_per_twitch)

    aggregate_dict[AUC_UUID] = auc_averages_dict

    # add metrics to per peak dictionary
    twitch_peak_indices = tuple(twitch_indices.keys())
    for i in range(num_twitches):
        main_twitch_dict.update(
            {
                time_series[twitch_peak_indices[i]]: {
                    TWITCH_PERIOD_UUID: combined_twitch_periods[i],
                    AMPLITUDE_UUID: amplitudes[i],
                    WIDTH_UUID: widths[i],
                    AUC_UUID: auc_per_twitch[i],
                    TWITCH_FREQUENCY_UUID: twitch_frequencies[i],
                }
            }
        )

    return main_twitch_dict, aggregate_dict


def calculate_twitch_period(
    twitch_indices: NDArray[int],
    all_peak_indices: NDArray[int],
    filtered_data: NDArray[(2, Any), int],
) -> NDArray[int]:
    """Find the distance between each twitch at its peak.

    Args:
        twitch_indices: a 1D array of the indices in the data array that the twitch peaks are at
        all_peak_indices: a 1D array of the indices in teh data array that all peaks are at
        filtered_data: a 2D array (time vs value) of the data

    Returns:
        an array of integers that are the peeriod of each twitch
    """
    list_of_twitch_indices = list(twitch_indices.keys())
    idx_of_first_twitch = np.where(all_peak_indices == list_of_twitch_indices[0])[0][0]
    period: List[int] = []
    time_series = filtered_data[0, :]
    for iter_twitch_idx in range(len(list_of_twitch_indices)):

        period.append(
            time_series[all_peak_indices[iter_twitch_idx + idx_of_first_twitch + 1]]
            - time_series[all_peak_indices[iter_twitch_idx + idx_of_first_twitch]]
        )

    return np.asarray(period, dtype=np.int32)


def find_twitch_indices(
    peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
    filtered_data: NDArray[(2, Any), int],
) -> Dict[int, Dict[UUID, Optional[int]]]:
    """Find twitches that can be analyzed.

    Sometimes the first and last peak in a trace can't be analyzed as a full twitch because not enough information is present.
    In order to be analyzable, a twitch needs to have a valley prior to it and another peak after it.

    Args:
        peak_and_valley_indices: a Tuple of 1D array of integers representing the indices of the peaks and valleys
        filtered_data: a 2D array of the data after being noise filtered

    Returns:
        a 1D array of integers representing the time points of all the twitches
    """
    peak_indices, valley_indices = peak_and_valley_indices
    if len(peak_indices) < MIN_NUMBER_PEAKS:
        raise TooFewPeaksDetectedError(
            f"A minimum of {MIN_NUMBER_PEAKS} peaks is required to extract twitch metrics, however only {len(peak_indices)} peak(s) were detected"
        )
    if len(valley_indices) < MIN_NUMBER_VALLEYS:
        raise TooFewPeaksDetectedError(
            f"A minimum of {MIN_NUMBER_VALLEYS} valleys is required to extract twitch metrics, however only {len(valley_indices)} valley(s) were detected"
        )
    twitches: Dict[int, Dict[UUID, Optional[int]]] = {}

    starts_with_peak = peak_indices[0] < valley_indices[0]
    prev_feature_is_peak = starts_with_peak
    peak_idx = 0
    valley_idx = 0
    if starts_with_peak:
        peak_idx += 1
    else:
        valley_idx += 1

    # check for two back-to-back features
    while peak_idx < len(peak_indices) and valley_idx < len(valley_indices):
        if prev_feature_is_peak:
            if valley_indices[valley_idx] > peak_indices[peak_idx]:
                raise TwoPeaksInARowError(
                    peak_and_valley_indices,
                    filtered_data,
                    (peak_indices[peak_idx - 1], peak_indices[peak_idx]),
                )
            valley_idx += 1
        else:
            if valley_indices[valley_idx] < peak_indices[peak_idx]:
                raise TwoValleysInARowError(
                    peak_and_valley_indices,
                    filtered_data,
                    (valley_indices[valley_idx - 1], valley_indices[valley_idx]),
                )
            peak_idx += 1
        prev_feature_is_peak = not prev_feature_is_peak
    if peak_idx < len(peak_indices) - 1:
        raise TwoPeaksInARowError(
            peak_and_valley_indices,
            filtered_data,
            (peak_indices[peak_idx], peak_indices[peak_idx + 1]),
        )
    if valley_idx < len(valley_indices) - 1:
        raise TwoValleysInARowError(
            peak_and_valley_indices,
            filtered_data,
            (valley_indices[valley_idx], valley_indices[valley_idx + 1]),
        )

    # compile dict of twitch information
    for itr_idx, itr_peak_index in enumerate(peak_indices):
        if itr_idx < peak_indices.shape[0] - 1:  # last peak
            if itr_idx == 0 and starts_with_peak:
                continue

            twitches[itr_peak_index] = {
                PRIOR_PEAK_INDEX_UUID: None
                if itr_idx == 0
                else peak_indices[itr_idx - 1],
                PRIOR_VALLEY_INDEX_UUID: valley_indices[
                    itr_idx - 1 if starts_with_peak else itr_idx
                ],
                SUBSEQUENT_PEAK_INDEX_UUID: peak_indices[itr_idx + 1],
                SUBSEQUENT_VALLEY_INDEX_UUID: valley_indices[
                    itr_idx if starts_with_peak else itr_idx + 1
                ],
            }

    # print(list(twitches.keys())[0])
    # print(twitches[list(twitches.keys())[0]])
    return twitches


def calculate_amplitudes(
    twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
    filtered_data: NDArray[(2, Any), int],
) -> NDArray[int]:
    """Get the amplitudes for all twitches.

    Args:
        twitch_indices: a 1D array of all the time values of the peaks of interest
        filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force...) data after it has gone through noise filtering

    Returns:
        a 1D array of integers representing the amplitude of each twitch
    """
    amplitudes: List[int] = list()
    amplitude_series = filtered_data[1, :]
    for iter_twitch_peak_idx, iter_twitch_indices_info in twitch_indices.items():
        peak_amplitude = amplitude_series[iter_twitch_peak_idx]
        prior_amplitude = amplitude_series[
            iter_twitch_indices_info[PRIOR_VALLEY_INDEX_UUID]
        ]
        subsequent_amplitude = amplitude_series[
            iter_twitch_indices_info[SUBSEQUENT_VALLEY_INDEX_UUID]
        ]
        amplitudes.append(
            abs(
                int(
                    round(
                        (
                            (peak_amplitude - prior_amplitude)
                            + (peak_amplitude - subsequent_amplitude)
                        )
                        / 2,
                        0,
                    )
                )
            )
        )

    return np.asarray(amplitudes, dtype=np.int32)


def interpolate_x_for_y_between_two_points(  # pylint:disable=invalid-name # (Eli 9/1/20: I can't think of a shorter name to describe this concept fully)
    desired_y: Union[int, float],
    x_1: Union[int, float],
    y_1: Union[int, float],
    x_2: Union[int, float],
    y_2: Union[int, float],
) -> Union[int, float]:
    """Find a value of x between two points that matches the desired y value.

    Uses linear interpolation, based on point-slope formula.
    """
    slope = (y_2 - y_1) / (x_2 - x_1)
    return (desired_y - y_1) / slope + x_1


def interpolate_y_for_x_between_two_points(  # pylint:disable=invalid-name # (Eli 9/1/20: I can't think of a shorter name to describe this concept fully)
    desired_x: Union[int, float],
    x_1: Union[int, float],
    y_1: Union[int, float],
    x_2: Union[int, float],
    y_2: Union[int, float],
) -> Union[int, float]:
    """Find a value of x between two points that matches the desired y value.

    Uses linear interpolation, based on point-slope formula.
    """
    slope = (y_2 - y_1) / (x_2 - x_1)
    return slope * (desired_x - x_1) + y_1


def calculate_twitch_widths(
    twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
    filtered_data: NDArray[(2, Any), int],
) -> List[Dict[int, Dict[UUID, Union[Tuple[int, int], int]]]]:
    """Determine twitch width between 10-90% down to the nearby valleys.

    Args:
        twitch_indices: a 1D array of all the time values of the peaks of interest
        filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force...) data after it has gone through noise filtering

    Returns:
        a list of dictionaries where the first key is the percentage of the way down to the nearby valleys, the second key is a UUID representing either the value of the width, or the rising or falling coordinates. The final value is either an int (for value) or a tuple of ints for the x/y coordinates
    """
    widths: List[Dict[int, Dict[UUID, Union[Tuple[int, int], int]]]] = list()
    value_series = filtered_data[1, :]
    time_series = filtered_data[0, :]
    for iter_twitch_peak_idx, iter_twitch_indices_info in twitch_indices.items():
        iter_width_dict: Dict[int, Dict[UUID, Union[Tuple[int, int], int]]] = dict()
        peak_value = value_series[iter_twitch_peak_idx]
        prior_valley_value = value_series[
            iter_twitch_indices_info[PRIOR_VALLEY_INDEX_UUID]
        ]
        subsequent_valley_value = value_series[
            iter_twitch_indices_info[SUBSEQUENT_VALLEY_INDEX_UUID]
        ]

        rising_amplitude = peak_value - prior_valley_value
        falling_amplitude = peak_value - subsequent_valley_value

        rising_idx = iter_twitch_peak_idx - 1
        falling_idx = iter_twitch_peak_idx + 1
        for iter_percent in TWITCH_WIDTH_PERCENTS:
            iter_percent_dict: Dict[UUID, Union[Tuple[int, int], int]] = dict()
            rising_threshold = peak_value - iter_percent / 100 * rising_amplitude
            falling_threshold = peak_value - iter_percent / 100 * falling_amplitude
            # move to the left from the twitch peak until the threshold is reached
            while abs(value_series[rising_idx] - prior_valley_value) > abs(
                rising_threshold - prior_valley_value
            ):
                rising_idx -= 1
            # move to the right from the twitch peak until the falling threshold is reached
            while abs(value_series[falling_idx] - subsequent_valley_value) > abs(
                falling_threshold - subsequent_valley_value
            ):
                falling_idx += 1
            interpolated_rising_timepoint = interpolate_x_for_y_between_two_points(
                rising_threshold,
                time_series[rising_idx],
                value_series[rising_idx],
                time_series[rising_idx + 1],
                value_series[rising_idx + 1],
            )
            interpolated_falling_timepoint = interpolate_x_for_y_between_two_points(
                falling_threshold,
                time_series[falling_idx],
                value_series[falling_idx],
                time_series[falling_idx - 1],
                value_series[falling_idx - 1],
            )
            iter_percent_dict[WIDTH_VALUE_UUID] = int(
                round(interpolated_falling_timepoint - interpolated_rising_timepoint, 0)
            )
            iter_percent_dict[WIDTH_RISING_COORDS_UUID] = (
                int(round(interpolated_rising_timepoint, 0)),
                int(round(rising_threshold, 0)),
            )
            iter_percent_dict[WIDTH_FALLING_COORDS_UUID] = (
                int(round(interpolated_falling_timepoint, 0)),
                int(round(falling_threshold, 0)),
            )
            iter_width_dict[iter_percent] = iter_percent_dict
        widths.append(iter_width_dict)
    return widths


def calculate_area_under_curve(  # pylint:disable=too-many-locals # Eli (9/1/20): may be able to refactor before pull request
    twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
    filtered_data: NDArray[(2, Any), int],
    per_twitch_widths: List[Dict[int, Dict[UUID, Union[Tuple[int, int], int]]]],
) -> NDArray[int]:
    """Calculate the area under the curve (AUC) for twitches."""
    width_percent = 90  # what percent of repolarization to use as the bottom limit for calculating AUC
    auc_per_twitch: List[int] = list()
    value_series = filtered_data[1, :]
    time_series = filtered_data[0, :]

    for iter_twitch_idx, (iter_twitch_peak_idx, iter_twitch_indices_info) in enumerate(
        twitch_indices.items()
    ):
        # iter_twitch_peak_timepoint = time_series[iter_twitch_peak_idx]
        width_info = per_twitch_widths[iter_twitch_idx]
        prior_valley_value = value_series[
            iter_twitch_indices_info[PRIOR_VALLEY_INDEX_UUID]
        ]
        subsequent_valley_value = value_series[
            iter_twitch_indices_info[SUBSEQUENT_VALLEY_INDEX_UUID]
        ]
        rising_coords = width_info[width_percent][WIDTH_RISING_COORDS_UUID]
        if not isinstance(
            rising_coords, tuple
        ):  # Eli (9/1/20): this appears needed to make mypy happy
            raise NotImplementedError(
                f"Rising coordinates under the key {WIDTH_RISING_COORDS_UUID} must be a tuple."
            )
        rising_x, rising_y = rising_coords
        falling_coords = width_info[width_percent][WIDTH_FALLING_COORDS_UUID]
        if not isinstance(
            falling_coords, tuple
        ):  # Eli (9/1/20): this appears needed to make mypy happy
            raise NotImplementedError(
                f"Falling coordinates under the key {WIDTH_FALLING_COORDS_UUID} must be a tuple."
            )
        falling_x, falling_y = falling_coords
        interp_y_for_lower_bound = partial(
            interpolate_y_for_x_between_two_points,
            x_1=rising_x,
            y_1=rising_y,
            x_2=falling_x,
            y_2=falling_y,
        )
        auc_total: Union[float, int] = 0

        # calculate area of rising side
        rising_idx = iter_twitch_peak_idx
        # move to the left from the twitch peak until the threshold is reached
        while abs(value_series[rising_idx - 1] - prior_valley_value) > abs(
            rising_y - prior_valley_value
        ):
            left_x = time_series[rising_idx - 1]
            right_x = time_series[rising_idx]
            left_y = value_series[rising_idx - 1]
            right_y = value_series[rising_idx]
            trapezoid_h = right_x - left_x
            trapezoid_left_side = abs(left_y - interp_y_for_lower_bound(left_x))
            trapezoid_right_side = abs(right_y - interp_y_for_lower_bound(right_x))

            auc_total += (trapezoid_left_side + trapezoid_right_side) / 2 * trapezoid_h
            rising_idx -= 1
        # final trapezoid at the boundary of the interpolated twitch width point
        left_x = rising_x
        right_x = time_series[rising_idx]
        left_y = rising_y
        right_y = value_series[rising_idx]
        trapezoid_h = right_x - left_x
        trapezoid_left_side = abs(left_y - interp_y_for_lower_bound(left_x))
        trapezoid_right_side = abs(right_y - interp_y_for_lower_bound(right_x))
        auc_total += (trapezoid_left_side + trapezoid_right_side) / 2 * trapezoid_h

        # calculate area of falling side
        falling_idx = iter_twitch_peak_idx
        # move to the left from the twitch peak until the threshold is reached
        while abs(value_series[falling_idx + 1] - subsequent_valley_value) > abs(
            falling_y - subsequent_valley_value
        ):
            left_x = time_series[falling_idx]
            right_x = time_series[falling_idx + 1]
            left_y = value_series[falling_idx]
            right_y = value_series[falling_idx + 1]
            trapezoid_h = right_x - left_x
            trapezoid_left_side = abs(left_y - interp_y_for_lower_bound(left_x))
            trapezoid_right_side = abs(right_y - interp_y_for_lower_bound(right_x))
            auc_total += (trapezoid_left_side + trapezoid_right_side) / 2 * trapezoid_h
            falling_idx += 1

        # final trapezoid at the boundary of the interpolated twitch width point
        left_x = time_series[falling_idx]
        right_x = falling_x
        left_y = value_series[rising_idx]
        right_y = falling_y
        trapezoid_h = right_x - left_x
        trapezoid_left_side = abs(left_y - interp_y_for_lower_bound(left_x))
        trapezoid_right_side = abs(right_y - interp_y_for_lower_bound(right_x))
        auc_total += (trapezoid_left_side + trapezoid_right_side) / 2 * trapezoid_h

        auc_per_twitch.append(int(round(auc_total, 0)))

    return np.asarray(auc_per_twitch, dtype=np.int64)
