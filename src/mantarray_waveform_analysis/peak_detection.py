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

from nptyping import Float64
from nptyping import NDArray
import numpy as np
from scipy import signal

from .constants import AMPLITUDE_UUID
from .constants import AUC_UUID
from .constants import CENTIMILLISECONDS_PER_SECOND
from .constants import CONTRACTION_VELOCITY_UUID
from .constants import MIN_NUMBER_PEAKS
from .constants import MIN_NUMBER_VALLEYS
from .constants import PRIOR_PEAK_INDEX_UUID
from .constants import PRIOR_VALLEY_INDEX_UUID
from .constants import RELAXATION_VELOCITY_UUID
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
TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_START = TWITCH_WIDTH_PERCENTS.index(10)
TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_END = TWITCH_WIDTH_PERCENTS.index(90)


def peak_detector(
    filtered_magnetic_signal: NDArray[(2, Any), int],
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
    magnetic_signal: NDArray[int] = filtered_magnetic_signal[1, :]
    peak_invertor_factor = 1
    valley_invertor_factor = -1
    if not twitches_point_up:
        peak_invertor_factor *= -1
        valley_invertor_factor *= -1
    sampling_period_cms = (
        filtered_magnetic_signal[0, 1] - filtered_magnetic_signal[0, 0]
    )
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
    max_height = np.max(magnetic_signal)
    min_height = np.min(magnetic_signal)
    max_prominence = abs(max_height - min_height)
    # find peaks and valleys
    peak_indices, _ = signal.find_peaks(
        magnetic_signal * peak_invertor_factor,
        width=minimum_required_samples_between_twitches / 2,
        distance=minimum_required_samples_between_twitches,
        prominence=max_prominence / 4,
    )
    valley_indices, properties = signal.find_peaks(
        magnetic_signal * valley_invertor_factor,
        width=minimum_required_samples_between_twitches / 2,
        distance=minimum_required_samples_between_twitches,
        prominence=max_prominence / 4,
    )
    left_ips = properties["left_ips"]
    right_ips = properties["right_ips"]

    # TODO Tanner (11/3/20): move this loop to find_twitch_indices
    # Patches error in B6 file for when two valleys are found in a single valley. If this is true left_bases, right_bases, prominences, and raw magnetic sensor data will also be equivalent to their previous value. This if statement indicates that the valley should be disregarded if the interpolated values on left and right intersection points of a horizontal line at the an evaluation height are equivalent. This would mean that the left and right sides of the peak and its neighbor peak align, indicating that it just one peak rather than two.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths
    for i in range(1, len(valley_indices)):
        if left_ips[i] == left_ips[i - 1] and right_ips[i] == right_ips[i - 1]:
            valley_idx = valley_indices[i]
            valley_idx_last = valley_indices[i - 1]
            if magnetic_signal[valley_idx_last] == magnetic_signal[valley_idx]:
                valley_indices = np.delete(valley_indices, i)
                i -= 1
            elif magnetic_signal[valley_idx] > magnetic_signal[valley_idx_last]:
                valley_indices = np.delete(valley_indices, i - 1)
            else:  # pragma: no cover  # (Anna 3/31/21): we don't have a case as of yet in which the first peak is higher than the second however know that it is possible and therefore aren't worried about code coverage in this case.
                valley_indices = np.delete(valley_indices, i)  # pragma: no cover
                i -= 1  # pragma: no cover

    return peak_indices, valley_indices


def create_avg_dict(
    metric: NDArray[int], round_to_int: bool = True
) -> Dict[str, Union[Float64, int]]:
    """Calculate the average values of a specific metric.

    Args:
        metric: a 1D array of integer values of a specific metric results

    Returns:
        a dictionary of the average statistics of that metric in which the metrics are the key and average statistics are the value
    """
    dictionary: Dict[str, Union[Float64, int]] = dict()

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
    rounded: bool = True,
) -> Tuple[
    Dict[
        int,
        Dict[
            UUID,  # pylint: disable=duplicate-code # Anna (1/7/21): long type definition causing failture
            Union[
                Dict[int, Dict[UUID, Union[Tuple[int, int], int]]],
                Union[float, int],
            ],
        ],  # pylint: disable=duplicate-code # Anna (1/7/21): long type definition causing failture
    ],
    Dict[
        UUID,
        Union[Dict[str, Union[float, int]], Dict[int, Dict[str, Union[float, int]]]],
    ],
]:  # pylint:disable=too-many-locals # Eli (9/8/20): there are a lot of metrics to calculate that need local variables
    """Find all data metrics for individual twitches and averages.

    Args:
        peak_and_valley_indices: a tuple of integer value arrays representing the time indices of peaks and valleys within the data
        filtered_data: a 2D array of the time and voltage data after it has gone through noise cancellation

    Returns:
        per_twitch_dict: a dictionary of individual peak metrics in which the twitch timepoint is accompanied by a dictionary in which the UUIDs for each twitch metric are the key and with its accompanying value as the value. For the Twitch Width metric UUID, another dictionary is stored in which the key is the percentage of the way down and the value is another dictionary in which the UUIDs for the rising coord, falling coord or width value are stored with the value as an int for the width value or a tuple of ints for the x/y coordinates
        aggregate_dict: a dictionary of entire metric statistics. Most metrics have the stats underneath the UUID, but for twitch widths, there is an additional dictionary where the percent of repolarization is the key
    """
    # create main dictionaries
    main_twitch_dict: Dict[
        int,
        Dict[
            UUID,
            Union[
                Dict[int, Dict[UUID, Union[Tuple[int, int], int]]], Union[float, int]
            ],
        ],
    ] = dict()
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
        combined_twitch_periods.astype(float) / CENTIMILLISECONDS_PER_SECOND
    )
    frequency_averages_dict = create_avg_dict(twitch_frequencies, round_to_int=False)

    # find aggregate values of period data
    period_averages_dict = create_avg_dict(combined_twitch_periods)

    aggregate_dict[TWITCH_PERIOD_UUID] = period_averages_dict
    aggregate_dict[TWITCH_FREQUENCY_UUID] = frequency_averages_dict

    # find twitch amplitudes
    amplitudes: NDArray[int] = calculate_amplitudes(
        twitch_indices, filtered_data, round_to_int=rounded
    )

    # find aggregate values of amplitude data
    amplitude_averages_dict = create_avg_dict(amplitudes, round_to_int=rounded)

    aggregate_dict[AMPLITUDE_UUID] = amplitude_averages_dict

    # find twitch widths
    widths = calculate_twitch_widths(
        twitch_indices, filtered_data, round_to_int=rounded
    )
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
        iter_stats_dict = create_avg_dict(
            iter_list_of_width_values, round_to_int=rounded
        )
        width_stats_dict[iter_percent] = iter_stats_dict

    aggregate_dict[WIDTH_UUID] = width_stats_dict

    # calculate auc
    auc_per_twitch: NDArray[int] = calculate_area_under_curve(
        twitch_indices, filtered_data, widths, round_to_int=rounded
    )

    # calculate twitch contraction/relaxation velocities
    contraction_velocity = calculate_twitch_velocity(twitch_indices, widths, True)
    contraction_velocity_averages = create_avg_dict(
        contraction_velocity, round_to_int=False
    )
    aggregate_dict[CONTRACTION_VELOCITY_UUID] = contraction_velocity_averages

    relaxation_velocity = calculate_twitch_velocity(twitch_indices, widths, False)
    relaxation_velocity_averages = create_avg_dict(
        relaxation_velocity, round_to_int=False
    )
    aggregate_dict[RELAXATION_VELOCITY_UUID] = relaxation_velocity_averages

    # find aggregate values of area under curve data
    auc_averages_dict = create_avg_dict(auc_per_twitch, round_to_int=rounded)

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
                    CONTRACTION_VELOCITY_UUID: contraction_velocity[i],
                    RELAXATION_VELOCITY_UUID: relaxation_velocity[i],
                }
            }
        )

    return main_twitch_dict, aggregate_dict


def calculate_twitch_velocity(
    twitch_indices: NDArray[int],
    widths: List[
        Dict[
            int,
            Dict[
                UUID,
                Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
            ],
        ],
    ],
    is_contraction: bool,
) -> NDArray[float]:
    """Find the velocity for each twitch.

    Args:
        twitch_indices: a dictionary in which the key is an integer representing the time points of all the peaks of interest and the value is an inner dictionary with various UUID of prior/subsequent peaks and valleys and their index values.
        widths: a list of dictionaries where the first key is the percentage of the way down to the nearby valleys, the second key is a UUID representing either the value of the width, or the rising or falling coordinates. The final value is either an int (for value) or a tuple of ints for the x/y coordinates
        is_contraction: a boolean indicating if twitch velocities to be calculating are for the twitch contraction or relaxation

    Returns:
        an array of floats that are the velocities of each twitch
    """
    list_of_twitch_indices = list(twitch_indices.keys())
    num_twitches = len(list_of_twitch_indices)
    coord_type = WIDTH_RISING_COORDS_UUID
    if not is_contraction:
        coord_type = WIDTH_FALLING_COORDS_UUID

    twitch_base = TWITCH_WIDTH_PERCENTS[TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_END]
    twitch_top = TWITCH_WIDTH_PERCENTS[TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_START]

    iter_list_of_velocities: List[Union[float, int]] = []
    for twitch in range(num_twitches):
        iter_coord_base = widths[twitch][twitch_base][coord_type]
        iter_coord_top = widths[twitch][twitch_top][coord_type]

        if not isinstance(iter_coord_base, tuple):  # making mypy happy
            raise NotImplementedError(
                f"The width value under twitch {twitch} must be a Tuple. It was: {iter_coord_base}"
            )
        if not isinstance(iter_coord_top, tuple):  # making mypy happy
            raise NotImplementedError(
                f"The width value under twitch {twitch} must be a Tuple. It was: {iter_coord_top}"
            )
        velocity = abs(
            (iter_coord_top[1] - iter_coord_base[1])
            / (iter_coord_top[0] - iter_coord_base[0])
        )
        iter_list_of_velocities.append(velocity)
    return np.asarray(iter_list_of_velocities, dtype=float)


def calculate_twitch_period(
    twitch_indices: NDArray[int],
    all_peak_indices: NDArray[int],
    filtered_data: NDArray[(2, Any), int],
) -> NDArray[int]:
    """Find the distance between each twitch at its peak.

    Args:
        twitch_indices:a dictionary in which the key is an integer representing the time points of all the peaks of interest and the value is an inner dictionary with various UUID of prior/subsequent peaks and valleys and their index values.
        all_peak_indices: a 1D array of the indices in teh data array that all peaks are at
        filtered_data: a 2D array (time vs value) of the data

    Returns:
        an array of integers that are the period of each twitch
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
        a dictionary in which the key is an integer representing the time points of all the peaks of interest and the value is an inner dictionary with various UUIDs of prior/subsequent peaks and valleys and their index values.
    """
    peak_indices, valley_indices = peak_and_valley_indices

    _too_few_peaks_or_valleys(peak_indices, valley_indices)

    twitches: Dict[int, Dict[UUID, Optional[int]]] = {}

    starts_with_peak = peak_indices[0] < valley_indices[0]
    prev_feature_is_peak = starts_with_peak
    peak_idx, valley_idx = _find_start_indices(starts_with_peak)

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


def _too_few_peaks_or_valleys(
    peak_indices: NDArray[int], valley_indices: NDArray[int]
) -> None:
    """Raise an error if there are too few peaks or valleys detected.

    Args:
        peak_indices: a 1D array of integers representing the indices of the peaks
        valley_indices: a 1D array of integeres representing the indices of the valleys
    """
    if len(peak_indices) < MIN_NUMBER_PEAKS:
        raise TooFewPeaksDetectedError(
            f"A minimum of {MIN_NUMBER_PEAKS} peaks is required to extract twitch metrics, however only {len(peak_indices)} peak(s) were detected"
        )
    if len(valley_indices) < MIN_NUMBER_VALLEYS:
        raise TooFewPeaksDetectedError(
            f"A minimum of {MIN_NUMBER_VALLEYS} valleys is required to extract twitch metrics, however only {len(valley_indices)} valley(s) were detected"
        )


def _find_start_indices(starts_with_peak: bool) -> Tuple[int, int]:
    """Find start indices for peaks and valleys.

    Args:
        starts_with_peak: bool indicating whether or not a peak rather than a valley comes first

    Returns:
        peak_idx: peak start index
        valley_idx: valley start index
    """
    peak_idx = 0
    valley_idx = 0
    if starts_with_peak:
        peak_idx += 1
    else:
        valley_idx += 1

    return peak_idx, valley_idx


def calculate_amplitudes(
    twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
    filtered_data: NDArray[(2, Any), int],
    round_to_int: bool = True,
) -> NDArray[float]:
    """Get the amplitudes for all twitches.

    Args:
        twitch_indices: a dictionary in which the key is an integer representing the time points of all the peaks of interest and the value is an inner dictionary with various UUID of prior/subsequent peaks and valleys and their index values.
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
        amplitude_value = (
            (peak_amplitude - prior_amplitude) + (peak_amplitude - subsequent_amplitude)
        ) / 2
        if round_to_int:
            amplitude_value = int(round(amplitude_value, 0))

        amplitudes.append(abs(amplitude_value))

    return np.asarray(amplitudes, dtype=float)


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
    """Find a value of y between two points that matches the desired x value.

    Uses linear interpolation, based on point-slope formula.
    """
    slope = (y_2 - y_1) / (x_2 - x_1)
    return slope * (desired_x - x_1) + y_1


def calculate_twitch_widths(
    twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
    filtered_data: NDArray[(2, Any), int],
    round_to_int: bool = True,
) -> List[
    Dict[
        int,
        Dict[
            UUID,
            Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
        ],
    ]
]:
    """Determine twitch width between 10-90% down to the nearby valleys.

    Args:
        twitch_indices: a dictionary in which the key is an integer representing the time points of all the peaks of interest and the value is an inner dictionary with various UUIDs of prior/subsequent peaks and valleys and their index values.
        filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force...) data after it has gone through noise filtering

    Returns:
        a list of dictionaries where the first key is the percentage of the way down to the nearby valleys, the second key is a UUID representing either the value of the width, or the rising or falling coordinates. The final value is either an int (for value) or a tuple of ints for the x/y coordinates
    """
    widths: List[
        Dict[
            int,
            Dict[
                UUID,
                Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
            ],
        ]
    ] = list()
    value_series = filtered_data[1, :]
    time_series = filtered_data[0, :]
    for iter_twitch_peak_idx, iter_twitch_indices_info in twitch_indices.items():
        iter_width_dict: Dict[
            int,
            Dict[
                UUID,
                Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
            ],
        ] = dict()
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
            iter_percent_dict: Dict[
                UUID,
                Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
            ] = dict()
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
            width_val = interpolated_falling_timepoint - interpolated_rising_timepoint
            if round_to_int:
                width_val = int(round(width_val, 0))
                interpolated_falling_timepoint = int(
                    round(interpolated_falling_timepoint, 0)
                )
                interpolated_rising_timepoint = int(
                    round(interpolated_rising_timepoint, 0)
                )
                rising_threshold = int(round(rising_threshold, 0))
                falling_threshold = int(round(falling_threshold, 0))

            iter_percent_dict[WIDTH_VALUE_UUID] = width_val
            iter_percent_dict[WIDTH_RISING_COORDS_UUID] = (
                interpolated_rising_timepoint,
                rising_threshold,
            )
            iter_percent_dict[WIDTH_FALLING_COORDS_UUID] = (
                interpolated_falling_timepoint,
                falling_threshold,
            )
            iter_width_dict[iter_percent] = iter_percent_dict
        widths.append(iter_width_dict)
    return widths


def calculate_area_under_curve(  # pylint:disable=too-many-locals # Eli (9/1/20): may be able to refactor before pull request
    twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
    filtered_data: NDArray[(2, Any), int],
    per_twitch_widths: List[
        Dict[
            int,
            Dict[
                UUID,
                Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
            ],
        ],
    ],
    round_to_int: bool = True,
) -> NDArray[float]:
    """Calculate the area under the curve (AUC) for twitches.

    Args:
        twitch_indices: a dictionary in which the key is an integer representing the time points of all the peaks of interest and the value is an inner dictionary with various UUIDs of prior/subsequent peaks and valleys and their index values.
        filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force...) data after it has gone through noise filtering
        per_twitch_widths: a list of dictionaries where the first key is the percentage of the way down to the nearby valleys, the second key is a UUID representing either the value of the width, or the rising or falling coordinates. The final value is either an int representing the width value or a tuple of ints for the x/y coordinates

    Returns:
        a 1D array of integers which represent the area under the curve for each twitch
    """
    width_percent = 90  # what percent of repolarization to use as the bottom limit for calculating AUC
    auc_per_twitch: List[float] = list()
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
        falling_coords = width_info[width_percent][WIDTH_FALLING_COORDS_UUID]

        if not isinstance(
            rising_coords, tuple
        ):  # Eli (9/1/20): this appears needed to make mypy happy
            raise NotImplementedError(
                f"Rising coordinates under the key {WIDTH_RISING_COORDS_UUID} must be a tuple."
            )

        if not isinstance(
            falling_coords, tuple
        ):  # Eli (9/1/20): this appears needed to make mypy happy
            raise NotImplementedError(
                f"Falling coordinates under the key {WIDTH_FALLING_COORDS_UUID} must be a tuple."
            )

        rising_x, rising_y = rising_coords
        falling_x, falling_y = falling_coords

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

            auc_total += _calculate_trapezoid_area(
                left_x,
                right_x,
                left_y,
                right_y,
                rising_coords,
                falling_coords,
            )
            rising_idx -= 1
        # final trapezoid at the boundary of the interpolated twitch width point
        left_x = rising_x
        right_x = time_series[rising_idx]
        left_y = rising_y
        right_y = value_series[rising_idx]

        auc_total += _calculate_trapezoid_area(
            left_x,
            right_x,
            left_y,
            right_y,
            rising_coords,
            falling_coords,
        )

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

            auc_total += _calculate_trapezoid_area(
                left_x,
                right_x,
                left_y,
                right_y,
                rising_coords,
                falling_coords,
            )
            falling_idx += 1

        # final trapezoid at the boundary of the interpolated twitch width point
        left_x = time_series[falling_idx]
        right_x = falling_x
        left_y = value_series[rising_idx]
        right_y = falling_y

        auc_total += _calculate_trapezoid_area(
            left_x,
            right_x,
            left_y,
            right_y,
            rising_coords,
            falling_coords,
        )
        if round_to_int:
            auc_total = int(round(auc_total, 0))
        auc_per_twitch.append(auc_total)

    return np.asarray(auc_per_twitch, dtype=float)


def _calculate_trapezoid_area(
    left_x: int,
    right_x: int,
    left_y: int,
    right_y: int,
    rising_coords: Tuple[Union[float, int], Union[float, int]],
    falling_coords: Tuple[Union[float, int], Union[float, int]],
) -> Union[int, float]:
    """Calculate the area under the trapezoid.

    Returns: area of the trapezoid
    """
    rising_x, rising_y = rising_coords
    falling_x, falling_y = falling_coords

    interp_y_for_lower_bound = partial(
        interpolate_y_for_x_between_two_points,
        x_1=rising_x,
        y_1=rising_y,
        x_2=falling_x,
        y_2=falling_y,
    )

    trapezoid_h = right_x - left_x
    trapezoid_left_side = abs(left_y - interp_y_for_lower_bound(left_x))
    trapezoid_right_side = abs(right_y - interp_y_for_lower_bound(right_x))
    auc_total = (trapezoid_left_side + trapezoid_right_side) / 2 * trapezoid_h

    return auc_total
