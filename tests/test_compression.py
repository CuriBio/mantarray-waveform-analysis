# -*- coding: utf-8 -*-
import os
import time

from mantarray_waveform_analysis import AMPLITUDE_UUID
from mantarray_waveform_analysis import AUC_UUID
from mantarray_waveform_analysis import compress_filtered_magnetic_data
from mantarray_waveform_analysis import MICRO_TO_BASE_CONVERSION
from mantarray_waveform_analysis import peak_detection
from mantarray_waveform_analysis import peak_detector
from mantarray_waveform_analysis import TWITCH_PERIOD_UUID
from mantarray_waveform_analysis import WIDTH_UUID
from mantarray_waveform_analysis import WIDTH_VALUE_UUID
import matplotlib
import numpy as np
from stdlib_utils import get_current_file_abs_directory

from .fixtures_compression import fixture_new_A1
from .fixtures_compression import fixture_new_A2
from .fixtures_compression import fixture_new_A3
from .fixtures_compression import fixture_new_A4
from .fixtures_compression import fixture_new_A5
from .fixtures_compression import fixture_new_A6
from .fixtures_pipelines import fixture_generic_pipeline_template
from .fixtures_utils import _get_data_metrics
from .fixtures_utils import _plot_data
from .fixtures_utils import assert_percent_diff
from .fixtures_utils import PATH_TO_PNGS

__fixtures__ = (
    fixture_new_A4,
    fixture_new_A1,
    fixture_new_A2,
    fixture_new_A3,
    fixture_new_A5,
    fixture_new_A6,
    fixture_generic_pipeline_template,
)

matplotlib.use("Agg")
PATH_OF_CURRENT_FILE = get_current_file_abs_directory()

COMPRESSION_ACCURACY = 0.10
COMPRESSION_ACCURACY_AMPLITUDE = 0.01  # amplitude is the key metric, so it must remain more accurate

COMPRESSION_FACTOR = 0.30


def _get_info_for_compression(well_fixture, file_prefix, pipeline_template_with_filter):
    pipeline, _ = well_fixture
    pipeline_with_filter = pipeline_template_with_filter.create_pipeline()
    unfiltered_data = pipeline.get_noise_filtered_gmr()
    pipeline_with_filter.load_raw_gmr_data(unfiltered_data, unfiltered_data)

    filtered_data = pipeline_with_filter.get_noise_filtered_gmr()
    original_per_twitch_dict, original_aggregate_metrics_dict = _get_data_metrics(well_fixture)

    original_num_samples = filtered_data.shape[1]

    # compress the data
    compressed_data = compress_filtered_magnetic_data(filtered_data)
    new_num_samples = compressed_data.shape[1]
    original_peak_and_valley_indices = peak_detector(filtered_data, twitches_point_up=False)
    compressed_peak_and_valley_indices = peak_detector(compressed_data, twitches_point_up=False)
    _plot_data(
        compressed_peak_and_valley_indices,
        compressed_data,
        os.path.join(PATH_TO_PNGS, f"{file_prefix}_compressed.png"),
    )
    _plot_data(
        original_peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, f"{file_prefix}_uncompressed.png"),
    )
    (
        original_per_twitch_dict,
        original_aggregate_metrics_dict,
    ) = peak_detection.data_metrics(original_peak_and_valley_indices, filtered_data)
    (
        compressed_per_twitch_dict,
        compressed_aggregate_metrics_dict,
    ) = peak_detection.data_metrics(compressed_peak_and_valley_indices, compressed_data)

    return (
        new_num_samples,
        original_num_samples,
        compressed_per_twitch_dict,
        compressed_aggregate_metrics_dict,
        original_per_twitch_dict,
        original_aggregate_metrics_dict,
    )


def test_compression__removes_all_except_first_and_last_points_of_flat_horizontal_line():
    expected = np.array([[0, 99], [10, 10]], dtype=np.int32)
    flat_data = np.array([list(range(100)), [10 for _ in range(100)]], dtype=np.int32)
    actual = compress_filtered_magnetic_data(flat_data)
    np.testing.assert_equal(actual, expected)


def test_compression_performance(new_A1):
    # data creation, noise cancellation, peak detection
    #  expected time:                        10416666
    # started at                            663155597
    # after stopping linear regression:      82693642
    #                                        39733980
    # after switching to numpy sum:          45458017
    # cythonize original rsquared code:      75517946
    #                                        35390130
    # fully converting rsquared to cython:    2518059
    # adding cpdef to rsquared:               1723097
    # better C typing:                         190731
    # linetrace=False:                         181949
    # switch dtype to int64                    197158
    # better numpy                             166147

    pipeline, _ = new_A1
    filtered_data = pipeline.get_noise_filtered_gmr()
    starting_time = time.perf_counter_ns()
    num_iters = 15
    for _ in range(num_iters):
        compress_filtered_magnetic_data(filtered_data)

    ending_time = time.perf_counter_ns()
    ns_per_iter = (ending_time - starting_time) / num_iters
    seconds_of_data = (filtered_data[0, -1] - filtered_data[0, 0]) * 10 / MICRO_TO_BASE_CONVERSION

    expected_time_per_compression = seconds_of_data / 24 / 4 / 10 * 10 ** 9
    # print(ns_per_iter)
    assert ns_per_iter < expected_time_per_compression


def test_new_A1_compression(new_A1, generic_pipeline_template):
    (
        new_num_samples,
        original_num_samples,
        compressed_per_twitch_dict,
        compressed_aggregate_metrics_dict,
        original_per_twitch_dict,
        original_aggregate_metrics_dict,
    ) = _get_info_for_compression(new_A1, "new_A1", generic_pipeline_template)

    # make sure sampling rate has been reduced by appropriate amount
    assert new_num_samples <= (original_num_samples * COMPRESSION_FACTOR)

    # make sure data metrics have not been altered heavily
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AUC_UUID]["mean"],
        original_aggregate_metrics_dict[AUC_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        original_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY_AMPLITUDE,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        original_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )

    iter_twitch_timepoint = list(original_per_twitch_dict.keys())[0]
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        threshold=COMPRESSION_ACCURACY,
    )


def test_new_A2_compression(new_A2, generic_pipeline_template):
    (
        new_num_samples,
        original_num_samples,
        compressed_per_twitch_dict,
        compressed_aggregate_metrics_dict,
        original_per_twitch_dict,
        original_aggregate_metrics_dict,
    ) = _get_info_for_compression(new_A2, "new_A2", generic_pipeline_template)

    # make sure sampling rate has been reduced by appropriate amount
    assert new_num_samples <= (original_num_samples * COMPRESSION_FACTOR)

    # make sure data metrics have not been altered heavily
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AUC_UUID]["mean"],
        original_aggregate_metrics_dict[AUC_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        original_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY_AMPLITUDE,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        original_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )

    iter_twitch_timepoint = list(original_per_twitch_dict.keys())[0]
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        threshold=COMPRESSION_ACCURACY,
    )


def test_new_A3_compression(new_A3, generic_pipeline_template):
    (
        new_num_samples,
        original_num_samples,
        compressed_per_twitch_dict,
        compressed_aggregate_metrics_dict,
        original_per_twitch_dict,
        original_aggregate_metrics_dict,
    ) = _get_info_for_compression(new_A3, "new_A3", generic_pipeline_template)

    # make sure sampling rate has been reduced by appropriate amount
    assert new_num_samples <= (original_num_samples * COMPRESSION_FACTOR)

    # make sure data metrics have not been altered heavily
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AUC_UUID]["mean"],
        original_aggregate_metrics_dict[AUC_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        original_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY_AMPLITUDE,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        original_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )

    iter_twitch_timepoint = list(original_per_twitch_dict.keys())[0]
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        threshold=COMPRESSION_ACCURACY,
    )


def test_new_A4_compression(new_A4, generic_pipeline_template):
    (
        new_num_samples,
        original_num_samples,
        compressed_per_twitch_dict,
        compressed_aggregate_metrics_dict,
        original_per_twitch_dict,
        original_aggregate_metrics_dict,
    ) = _get_info_for_compression(new_A4, "new_A4", generic_pipeline_template)

    # make sure sampling rate has been reduced by appropriate amount
    assert new_num_samples <= (original_num_samples * COMPRESSION_FACTOR)
    # make sure data metrics have not been altered heavily
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AUC_UUID]["mean"],
        original_aggregate_metrics_dict[AUC_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        original_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY_AMPLITUDE,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        original_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )

    iter_twitch_timepoint = list(original_per_twitch_dict.keys())[0]
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        threshold=COMPRESSION_ACCURACY,
    )


def test_new_A5_compression(new_A5, generic_pipeline_template):
    (
        new_num_samples,
        original_num_samples,
        compressed_per_twitch_dict,
        compressed_aggregate_metrics_dict,
        original_per_twitch_dict,
        original_aggregate_metrics_dict,
    ) = _get_info_for_compression(new_A5, "new_A5", generic_pipeline_template)

    # make sure sampling rate has been reduced by appropriate amount
    assert new_num_samples <= (original_num_samples * COMPRESSION_FACTOR)
    # make sure data metrics have not been altered heavily
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AUC_UUID]["mean"],
        original_aggregate_metrics_dict[AUC_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        original_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY_AMPLITUDE,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        original_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )

    iter_twitch_timepoint = list(original_per_twitch_dict.keys())[0]
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        threshold=COMPRESSION_ACCURACY,
    )


def test_new_A6_compression(new_A6, generic_pipeline_template):
    (
        new_num_samples,
        original_num_samples,
        compressed_per_twitch_dict,
        compressed_aggregate_metrics_dict,
        original_per_twitch_dict,
        original_aggregate_metrics_dict,
    ) = _get_info_for_compression(new_A6, "new_A6", generic_pipeline_template)

    # make sure sampling rate has been reduced by appropriate amount
    assert new_num_samples <= (original_num_samples * COMPRESSION_FACTOR)
    # make sure data metrics have not been altered heavily
    assert len(compressed_per_twitch_dict.keys()) == len(original_per_twitch_dict.keys())
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AUC_UUID]["mean"],
        original_aggregate_metrics_dict[AUC_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        original_aggregate_metrics_dict[AMPLITUDE_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY_AMPLITUDE,
    )
    assert_percent_diff(
        compressed_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        original_aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"],
        threshold=COMPRESSION_ACCURACY,
    )

    iter_twitch_timepoint = list(original_per_twitch_dict.keys())[0]
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][AUC_UUID],
        threshold=COMPRESSION_ACCURACY,
    )
    assert_percent_diff(
        compressed_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        original_per_twitch_dict[iter_twitch_timepoint][WIDTH_UUID][90][WIDTH_VALUE_UUID],
        threshold=COMPRESSION_ACCURACY,
    )
