# -*- coding: utf-8 -*-
from mantarray_waveform_analysis import ALL_METRICS
from mantarray_waveform_analysis import AMPLITUDE_UUID
from mantarray_waveform_analysis import CONTRACTION_VELOCITY_UUID
from mantarray_waveform_analysis import find_twitch_indices
from mantarray_waveform_analysis import MICRO_TO_BASE_CONVERSION
from mantarray_waveform_analysis import MIN_NUMBER_PEAKS
from mantarray_waveform_analysis import MIN_NUMBER_VALLEYS
from mantarray_waveform_analysis import peak_detector
from mantarray_waveform_analysis import PRIOR_PEAK_INDEX_UUID
from mantarray_waveform_analysis import PRIOR_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import RELAXATION_VELOCITY_UUID
from mantarray_waveform_analysis import SUBSEQUENT_PEAK_INDEX_UUID
from mantarray_waveform_analysis import SUBSEQUENT_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import TooFewPeaksDetectedError
from mantarray_waveform_analysis import TWITCH_FREQUENCY_UUID
from mantarray_waveform_analysis import TWITCH_PERIOD_UUID
from mantarray_waveform_analysis import TwoPeaksInARowError
from mantarray_waveform_analysis import TwoValleysInARowError
from mantarray_waveform_analysis.peak_detection import create_statistics_dict
import numpy as np
import pytest

from .fixtures_compression import fixture_new_A1
from .fixtures_utils import _get_data_metrics

__fixtures__ = [fixture_new_A1]


def get_test_data_array():
    test_timepoints = np.arange(0, 10, 0.1) * MICRO_TO_BASE_CONVERSION
    return np.array([test_timepoints, np.sin(test_timepoints * np.pi * MICRO_TO_BASE_CONVERSION)])


def test_peak_detection__flips_data_if_it_is_magnetic():
    test_arr = get_test_data_array()

    non_magnetic_peaks, non_magnetic_valleys = peak_detector(test_arr, is_magnetic_data=False)
    magnetic_peaks, magnetic_valleys = peak_detector(test_arr, is_magnetic_data=True)

    num_peaks = len(non_magnetic_peaks)
    for idx in range(num_peaks):
        assert non_magnetic_peaks[idx] == magnetic_valleys[idx], idx
    num_valleys = len(non_magnetic_valleys)
    for idx in range(num_valleys):
        assert non_magnetic_valleys[idx] == magnetic_peaks[idx], idx


def test_peak_detection__analyzes_data_correctly_when_twitches_point_down():
    test_arr = get_test_data_array()

    non_flipped_peaks, non_flipped_valleys = peak_detector(test_arr, twitches_point_up=True)
    flipped_peaks, flipped_valleys = peak_detector(test_arr, twitches_point_up=False)

    num_peaks = len(non_flipped_peaks)
    for idx in range(num_peaks):
        assert non_flipped_peaks[idx] == flipped_peaks[idx], idx
    num_valleys = len(non_flipped_valleys)
    for idx in range(num_valleys):
        assert non_flipped_valleys[idx] == flipped_valleys[idx], idx


def test_create_statistics_dict__handles_empty_input_array():
    assert create_statistics_dict(np.array([])) == {
        "n": 0,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
    }


def test_find_twitch_indices__raises_error_if_less_than_3_peaks_given():
    with pytest.raises(
        TooFewPeaksDetectedError,
        match=rf"A minimum of {MIN_NUMBER_PEAKS} peaks is required to extract twitch metrics, however only 2 peak\(s\) were detected",
    ):
        find_twitch_indices((np.array([1, 2]), None))


def test_find_twitch_indices__raises_error_if_less_than_3_valleys_given():
    with pytest.raises(
        TooFewPeaksDetectedError,
        match=rf"A minimum of {MIN_NUMBER_VALLEYS} valleys is required to extract twitch metrics, however only 2 valley\(s\) were detected",
    ):
        find_twitch_indices((np.array([1, 3, 5]), np.array([2, 4])))


def test_find_twitch_indices__raises_error_if_no_valleys_given():
    with pytest.raises(
        TooFewPeaksDetectedError,
        match=rf"A minimum of {MIN_NUMBER_VALLEYS} valleys is required to extract twitch metrics, however only 0 valley\(s\) were detected",
    ):
        find_twitch_indices((np.array([1, 3, 5]), np.array([])))


def test_find_twitch_indices__excludes_first_and_last_peak_when_starts_and_ends_with_peaks():
    test_peak_indices = np.arange(0, 12, 2)
    test_valley_indices = np.arange(1, 11, 2)
    actual_twitch_indices = find_twitch_indices((test_peak_indices, test_valley_indices))

    actual_twitch_peak_indices = list(actual_twitch_indices.keys())
    assert actual_twitch_peak_indices == list(test_peak_indices[1:-1])

    assert actual_twitch_indices[test_peak_indices[1]] == {
        PRIOR_PEAK_INDEX_UUID: test_peak_indices[0],
        PRIOR_VALLEY_INDEX_UUID: test_valley_indices[0],
        SUBSEQUENT_PEAK_INDEX_UUID: test_peak_indices[2],
        SUBSEQUENT_VALLEY_INDEX_UUID: test_valley_indices[1],
    }


def test_find_twitch_indices__excludes_only_last_peak_when_no_outer_peak_at_beginning_and_no_outer_valley_at_end():
    test_peak_indices = np.arange(3, 12, 2)
    test_valley_indices = np.arange(1, 11, 2)
    actual_twitch_indices = find_twitch_indices((test_peak_indices, test_valley_indices))

    actual_twitch_peak_indices = list(actual_twitch_indices.keys())
    assert actual_twitch_peak_indices == list(test_peak_indices[:-1])

    assert actual_twitch_indices[test_peak_indices[0]] == {
        PRIOR_PEAK_INDEX_UUID: None,
        PRIOR_VALLEY_INDEX_UUID: test_valley_indices[0],
        SUBSEQUENT_PEAK_INDEX_UUID: test_peak_indices[1],
        SUBSEQUENT_VALLEY_INDEX_UUID: test_valley_indices[1],
    }


@pytest.mark.parametrize(
    "test_peaks,expected_match,test_description",
    [
        (
            [0, 1, 4, 8, 12],
            "0 and 1",
            "raises error when two peaks in a row at beginning",
        ),
        (
            [0, 5, 7, 8, 12],
            "7 and 8",
            "raises error when two peaks in a row in middle",
        ),
        (
            [0, 4, 8, 11, 12],
            "11 and 12",
            "raises error when two peaks in a row at end",
        ),
    ],
)
def test_find_twitch_indices__raises_error_if_two_peaks_in_a_row__and_start_with_peak(
    test_peaks, expected_match, test_description
):
    test_valleys = [2, 6, 10]
    with pytest.raises(TwoPeaksInARowError, match=expected_match):
        find_twitch_indices((test_peaks, test_valleys))


@pytest.mark.parametrize(
    "test_peaks,expected_match,test_description",
    [
        (
            [0, 1, 4, 8, 12],
            "0 and 1",
            "raises error when two peaks in a row at beginning",
        ),
        (
            [0, 5, 7, 8, 12],
            "7 and 8",
            "raises error when two peaks in a row in middle",
        ),
        (
            [0, 4, 8, 11, 12],
            "11 and 12",
            "raises error when two peaks in a row at end",
        ),
    ],
)
def test_find_twitch_indices__raises_error_if_two_peaks_in_a_row__and_does_not_start_with_peak(
    test_peaks, expected_match, test_description
):
    test_valleys = [-2, 2, 6, 10]
    with pytest.raises(TwoPeaksInARowError, match=expected_match):
        find_twitch_indices((test_peaks, test_valleys))


@pytest.mark.parametrize(
    "test_valleys,expected_match,test_description",
    [
        (
            [0, 1, 4, 8, 12],
            "0 and 1",
            "raises error when two valleys in a row at beginning",
        ),
        (
            [0, 5, 7, 8, 12],
            "7 and 8",
            "raises error when two valleys in a row in middle",
        ),
        (
            [0, 4, 8, 11, 12],
            "11 and 12",
            "raises error when two valleys in a row at end",
        ),
    ],
)
def test_find_twitch_indices__raises_error_if_two_valleys_in_a_row__and_start_with_valley(
    test_valleys, expected_match, test_description
):
    test_peaks = [2, 6, 10]
    with pytest.raises(TwoValleysInARowError, match=expected_match):
        find_twitch_indices((test_peaks, test_valleys))


@pytest.mark.parametrize(
    "test_valleys,expected_match,test_description",
    [
        (
            [0, 1, 4, 8, 12],
            "0 and 1",
            "raises error when two valleys in a row at beginning",
        ),
        (
            [0, 5, 7, 8, 12],
            "7 and 8",
            "raises error when two valleys in a row in middle",
        ),
        (
            [0, 4, 8, 11, 12],
            "11 and 12",
            "raises error when two valleys in a row at end",
        ),
    ],
)
def test_find_twitch_indices__raises_error_if_two_valleys_in_a_row__and_does_not_start_with_valley(
    test_valleys, expected_match, test_description
):
    test_peaks = [-2, 2, 6, 10]
    with pytest.raises(TwoValleysInARowError, match=expected_match):
        find_twitch_indices((test_peaks, test_valleys))


def test_find_twitch_indices__returns_correct_values_with_data_that_ends_in_peak():
    peak_indices = np.array([1, 3, 5], dtype=np.int32)
    valley_indices = np.array([0, 2, 4], dtype=np.int32)
    actual = find_twitch_indices((peak_indices, valley_indices))

    assert actual[1][PRIOR_PEAK_INDEX_UUID] is None
    assert actual[3][PRIOR_PEAK_INDEX_UUID] == 1

    assert actual[1][PRIOR_VALLEY_INDEX_UUID] == 0
    assert actual[3][PRIOR_VALLEY_INDEX_UUID] == 2

    assert actual[1][SUBSEQUENT_PEAK_INDEX_UUID] == 3
    assert actual[3][SUBSEQUENT_PEAK_INDEX_UUID] == 5

    assert actual[1][SUBSEQUENT_VALLEY_INDEX_UUID] == 2
    assert actual[3][SUBSEQUENT_VALLEY_INDEX_UUID] == 4


def test_find_twitch_indices__returns_correct_values_with_data_that_ends_in_valley():
    peak_indices = np.array([1, 3, 5], dtype=np.int32)
    valley_indices = np.array([2, 4, 6], dtype=np.int32)
    actual = find_twitch_indices((peak_indices, valley_indices))

    assert actual[3][PRIOR_PEAK_INDEX_UUID] == 1
    assert actual[3][PRIOR_VALLEY_INDEX_UUID] == 2
    assert actual[3][SUBSEQUENT_PEAK_INDEX_UUID] == 5
    assert actual[3][SUBSEQUENT_VALLEY_INDEX_UUID] == 4


@pytest.mark.parametrize(
    "expected_metrics,test_description",
    [
        ([TWITCH_FREQUENCY_UUID, AMPLITUDE_UUID], "only creates freq and amp metrics"),
        ([TWITCH_PERIOD_UUID, CONTRACTION_VELOCITY_UUID], "only creates period and contraction velocity"),
        ([RELAXATION_VELOCITY_UUID], "only creates relaxation velocity"),
    ],
)
def test_data_metrics__accepts_kwarg_dict_of_which_metrics_to_create(
    expected_metrics, test_description, new_A1
):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A1, metrics_to_create=expected_metrics)
    for metric_id in ALL_METRICS:
        # make sure metric handled correctly in aggregate dict
        assert (metric_id in aggregate_metrics_dict) is (metric_id in expected_metrics), metric_id
        # make sure metric handled correctly in per twitch dict
        for twitch_timepoint, twitch_dict in per_twitch_dict.items():
            assert (metric_id in twitch_dict) is (
                metric_id in expected_metrics
            ), f"{twitch_timepoint}, {metric_id}"
