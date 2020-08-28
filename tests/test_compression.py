# -*- coding: utf-8 -*-
import inspect
import os
import time

from mantarray_waveform_analysis import AUC_UUID
from mantarray_waveform_analysis import compress_filtered_gmr
from mantarray_waveform_analysis import peak_detection
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .fixtures_compression import fixture_new_A1
from .fixtures_compression import fixture_new_A2
from .fixtures_compression import fixture_new_A3
from .fixtures_compression import fixture_new_A4
from .fixtures_compression import fixture_new_A5
from .fixtures_compression import fixture_new_A6
from .fixtures_utils import PATH_TO_PNGS

matplotlib.use("Agg")

PATH_OF_CURRENT_FILE = os.path.dirname((inspect.stack()[0][1]))

COMPRESSION_ACCURACY = 0.30

COMPRESSION_FACTOR = 0.40

__fixtures__ = [
    fixture_new_A4,
    fixture_new_A1,
    fixture_new_A2,
    fixture_new_A3,
    fixture_new_A5,
    fixture_new_A6,
]


def test_compression__does_not_raise_error_with_horizontal_line():
    flat_data = np.array([list(range(100)), [10 for _ in range(100)]], dtype=np.int32)
    compress_filtered_gmr(flat_data)


def test_compression_performance(new_A1):
    # data creation, noise cancellation, peak detection
    #  expected time:                        10416666.666666666
    # started at                            663155597.9
    # after stopping linear regression:      82693642.5
    #                                        39733980.4
    # after switching to numpy sum:          45458017.3
    # cythonize original rsquared code:      75517946
    #                                        35390130
    # fully converting rsquared to cython:    2518059
    # adding cpdef to rsquared:               1723097

    _, _, _, _, noise_free_data = new_A1
    starting_time = time.perf_counter_ns()
    num_iters = 15
    for _ in range(num_iters):
        compress_filtered_gmr(noise_free_data)

    ending_time = time.perf_counter_ns()
    ns_per_iter = (ending_time - starting_time) / num_iters
    centimilliseconds_per_second = 100000
    seconds_of_data = (
        noise_free_data[0, -1] - noise_free_data[0, 0]
    ) / centimilliseconds_per_second

    expected_time_per_compression = seconds_of_data / 24 / 4 * 10 ** 9
    assert ns_per_iter < expected_time_per_compression


def test_new_A1_compression(new_A1):
    # data creation, noise cancellation, peak detection
    _, _, peakind, original_sampling_rate, noise_free_data = new_A1

    # determine data metrics of original data
    per_beat_dict_original, window_dict_original = peak_detection.data_metrics(
        peakind, noise_free_data
    )
    widths_dict_original = peak_detection.twitch_widths(peakind, noise_free_data)

    # compress the data
    compressed_data = compress_filtered_gmr(noise_free_data)
    new_sample_rate = int(len(compressed_data[0, :]) / 10)

    # run compressed peak detection
    peak_valley_tuple = peak_detection.compressed_peak_detector(
        compressed_data, peakind, noise_free_data
    )

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(
        peak_valley_tuple, compressed_data
    )

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peak_valley_tuple, compressed_data)

    compress_peakind = np.concatenate([peak_valley_tuple[0], peak_valley_tuple[1]])
    compress_peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A1_compressed.png")

    # plot compressed data for visual inspection
    plt.figure()
    plt.plot(compressed_data[0], compressed_data[1])
    plt.plot(
        compressed_data[0][compress_peakind], compressed_data[1][compress_peakind], "ro"
    )
    plt.xlabel("Time (centimilliseconds)")
    plt.ylabel("Voltage (V)")
    plt.savefig(my_local_path_graphs)

    # make sure sampling rate has been reduced by at least 75%
    assert new_sample_rate <= (original_sampling_rate * COMPRESSION_FACTOR)

    # make sure data metrics have not been altered heavily
    assert (
        np.absolute(
            window_dict_original[AUC_UUID]["mean"] - window_dict[AUC_UUID]["mean"]
        )
        / window_dict_original[AUC_UUID]["mean"]
    ) <= COMPRESSION_ACCURACY  # currently 6.7%
    this_twitch_idx = 105000
    assert (
        np.absolute(
            per_beat_dict_original[this_twitch_idx][AUC_UUID]
            - per_beat_dict[this_twitch_idx][AUC_UUID]
        )
        / per_beat_dict_original[this_twitch_idx][AUC_UUID]
    ) <= COMPRESSION_ACCURACY  # currently 1.2%

    assert (
        np.absolute(
            widths_dict_original[this_twitch_idx][10][2] - widths_dict[105000][10][2]
        )
        / widths_dict_original[this_twitch_idx][10][2]
    ) <= COMPRESSION_ACCURACY  # currently 5.6%


def test_new_A2_compression(new_A2):
    # data creation, noise cancellation, peak detection
    _, _, peakind, original_sampling_rate, noise_free_data = new_A2

    # determine data metrics of original data
    per_beat_dict_original, window_dict_original = peak_detection.data_metrics(
        peakind, noise_free_data
    )
    widths_dict_original = peak_detection.twitch_widths(peakind, noise_free_data)

    # compress the data
    compressed_data = compress_filtered_gmr(noise_free_data)
    new_sample_rate = int(len(compressed_data[0, :]) / 10)

    # run compressed peak detection
    peak_valley_tuple = peak_detection.compressed_peak_detector(
        compressed_data, peakind, noise_free_data
    )

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(
        peak_valley_tuple, compressed_data
    )

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peak_valley_tuple, compressed_data)

    compress_peakind = np.concatenate([peak_valley_tuple[0], peak_valley_tuple[1]])
    compress_peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A2_compressed.png")

    # plot compressed data for visual inspection
    plt.figure()
    plt.plot(compressed_data[0], compressed_data[1], "o-b")
    plt.plot(
        compressed_data[0][compress_peakind], compressed_data[1][compress_peakind], "ro"
    )
    plt.xlabel("Time (centimilliseconds)")
    plt.ylabel("Voltage (V)")
    plt.savefig(my_local_path_graphs)

    # make sure sampling rate has been reduced by at least 75%
    assert new_sample_rate <= (original_sampling_rate * COMPRESSION_FACTOR)

    # make sure data metrics have not been altered heavily
    assert (
        np.absolute(
            window_dict_original[AUC_UUID]["mean"] - window_dict[AUC_UUID]["mean"]
        )
        / window_dict_original[AUC_UUID]["mean"]
    ) < COMPRESSION_ACCURACY  # currently < 5%
    this_twitch_idx = 104000
    assert (
        np.absolute(
            per_beat_dict_original[this_twitch_idx][AUC_UUID]
            - per_beat_dict[this_twitch_idx][AUC_UUID]
        )
        / per_beat_dict_original[this_twitch_idx][AUC_UUID]
    ) < COMPRESSION_ACCURACY  # currently 12.5%

    assert (
        np.absolute(
            widths_dict_original[this_twitch_idx][10][2] - widths_dict[104000][10][2]
        )
        / widths_dict_original[this_twitch_idx][10][2]
    ) < COMPRESSION_ACCURACY  # currently 3.12%


def test_new_A3_compression(new_A3):
    # data creation, noise cancellation, peak detection
    _, _, peakind, original_sampling_rate, noise_free_data = new_A3

    # determine data metrics of original data
    per_beat_dict_original, window_dict_original = peak_detection.data_metrics(
        peakind, noise_free_data
    )
    widths_dict_original = peak_detection.twitch_widths(peakind, noise_free_data)

    # compress the data
    compressed_data = compress_filtered_gmr(noise_free_data)
    new_sample_rate = int(len(compressed_data[0, :]) / 10)

    # run compressed peak detection
    peak_valley_tuple = peak_detection.compressed_peak_detector(
        compressed_data, peakind, noise_free_data
    )

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(
        peak_valley_tuple, compressed_data
    )

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peak_valley_tuple, compressed_data)

    compress_peakind = np.concatenate([peak_valley_tuple[0], peak_valley_tuple[1]])
    compress_peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A3_compressed.png")

    # plot compressed data for visual inspection
    plt.figure()
    plt.plot(compressed_data[0], compressed_data[1], "o-b")
    plt.plot(
        compressed_data[0][compress_peakind], compressed_data[1][compress_peakind], "ro"
    )
    plt.xlabel("Time (centimilliseconds)")
    plt.ylabel("Voltage (V)")
    plt.savefig(my_local_path_graphs)

    # make sure sampling rate has been reduced by at least 75%
    assert new_sample_rate <= (original_sampling_rate * COMPRESSION_FACTOR)

    # make sure data metrics have not been altered heavily
    assert (
        np.absolute(
            window_dict_original[AUC_UUID]["mean"] - window_dict[AUC_UUID]["mean"]
        )
        / window_dict_original[AUC_UUID]["mean"]
    ) < COMPRESSION_ACCURACY
    this_twitch_idx = 108000
    assert (
        np.absolute(
            per_beat_dict_original[this_twitch_idx][AUC_UUID]
            - per_beat_dict[109000][AUC_UUID]
        )
        / per_beat_dict_original[this_twitch_idx][AUC_UUID]
    ) < COMPRESSION_ACCURACY

    this_twitch_idx = 266000
    assert (
        np.absolute(
            widths_dict_original[this_twitch_idx][10][2] - widths_dict[267000][10][2]
        )
        / widths_dict_original[this_twitch_idx][10][2]
    ) < COMPRESSION_ACCURACY


def test_new_A4_compression(new_A4):
    # data creation, noise cancellation, peak detection
    _, _, peakind, original_sampling_rate, noise_free_data = new_A4

    # determine data metrics of original data
    per_beat_dict_original, window_dict_original = peak_detection.data_metrics(
        peakind, noise_free_data
    )

    # compress the data
    compressed_data = compress_filtered_gmr(noise_free_data)
    new_sample_rate = int(len(compressed_data[0, :]) / 10)

    # run compressed peak detection
    peak_valley_tuple = peak_detection.compressed_peak_detector(
        compressed_data, peakind, noise_free_data
    )

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(
        peak_valley_tuple, compressed_data
    )

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peak_valley_tuple, compressed_data)

    compress_peakind = np.concatenate([peak_valley_tuple[0], peak_valley_tuple[1]])
    compress_peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A4_compressed.png")

    # plot compressed data for visual inspection
    plt.figure()
    plt.plot(compressed_data[0], compressed_data[1])
    plt.plot(
        compressed_data[0][compress_peakind], compressed_data[1][compress_peakind], "ro"
    )
    plt.xlabel("Time (centimilliseconds)")
    plt.ylabel("Voltage (V)")
    plt.savefig(my_local_path_graphs)

    # make sure sampling rate has been reduced by at least 75%
    assert new_sample_rate <= (original_sampling_rate * COMPRESSION_FACTOR)

    # make sure data metrics have not been altered heavily
    assert (
        np.absolute(
            window_dict_original[AUC_UUID]["mean"] - window_dict[AUC_UUID]["mean"]
        )
        / window_dict_original[AUC_UUID]["mean"]
    ) <= COMPRESSION_ACCURACY
    this_twitch_idx = 81000
    assert (
        np.absolute(
            per_beat_dict_original[this_twitch_idx][AUC_UUID]
            - per_beat_dict[this_twitch_idx][AUC_UUID]
        )
        / per_beat_dict_original[this_twitch_idx][AUC_UUID]
    ) <= COMPRESSION_ACCURACY
    assert (
        np.absolute(9000 - widths_dict[81000][10][2]) / 30000
    ) <= COMPRESSION_ACCURACY


def test_new_A5_compression(new_A5):
    # data creation, noise cancellation, peak detection
    _, _, peakind, original_sampling_rate, noise_free_data = new_A5

    # determine data metrics of original data
    per_beat_dict_original, window_dict_original = peak_detection.data_metrics(
        peakind, noise_free_data
    )

    widths_dict_original = peak_detection.twitch_widths(peakind, noise_free_data)

    # compress the data
    compressed_data = compress_filtered_gmr(noise_free_data)
    # print (noise_free_data.shape)
    # print (compressed_data.shape)
    new_sample_rate = int(len(compressed_data[0, :]) / 10)

    # run compressed peak detection
    peak_valley_tuple = peak_detection.compressed_peak_detector(
        compressed_data, peakind, noise_free_data
    )

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(
        peak_valley_tuple, compressed_data
    )

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peak_valley_tuple, compressed_data)

    compress_peakind = np.concatenate([peak_valley_tuple[0], peak_valley_tuple[1]])
    compress_peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A5_compressed.png")

    # plot compressed data for visual inspection
    plt.figure()
    plt.plot(compressed_data[0], compressed_data[1])
    plt.plot(
        compressed_data[0][compress_peakind], compressed_data[1][compress_peakind], "ro"
    )
    plt.xlabel("Time (centimilliseconds)")
    plt.ylabel("Voltage (V)")
    plt.savefig(my_local_path_graphs)

    # make sure sampling rate has been reduced by at least 75%
    assert new_sample_rate <= (original_sampling_rate * COMPRESSION_FACTOR)

    # make sure data metrics have not been altered heavily
    assert (
        np.absolute(
            window_dict_original[AUC_UUID]["mean"] - window_dict[AUC_UUID]["mean"]
        )
        / window_dict_original[AUC_UUID]["mean"]
    ) < COMPRESSION_ACCURACY
    this_twitch_idx = 80000
    assert (
        np.absolute(
            per_beat_dict_original[this_twitch_idx][AUC_UUID]
            - per_beat_dict[this_twitch_idx][AUC_UUID]
        )
        / per_beat_dict_original[this_twitch_idx][AUC_UUID]
    ) < COMPRESSION_ACCURACY
    assert (
        np.absolute(
            widths_dict_original[this_twitch_idx][10][2]
            - widths_dict[this_twitch_idx][10][2]
        )
        / widths_dict_original[this_twitch_idx][10][2]
    ) < COMPRESSION_ACCURACY


def test_new_A6_compression(new_A6):
    # data creation, noise cancellation, peak detection
    _, _, peakind, original_sampling_rate, noise_free_data = new_A6

    # determine data metrics of original data
    per_beat_dict_original, window_dict_original = peak_detection.data_metrics(
        peakind, noise_free_data
    )
    widths_dict_original = peak_detection.twitch_widths(peakind, noise_free_data)

    # compress the data
    compressed_data = compress_filtered_gmr(noise_free_data)
    new_sample_rate = int(len(compressed_data[0, :]) / 10)

    # run compressed peak detection
    peak_valley_tuple = peak_detection.compressed_peak_detector(
        compressed_data, peakind, noise_free_data
    )

    # determine data metrics of compressed data
    per_beat_dict, window_dict = peak_detection.data_metrics(
        peak_valley_tuple, compressed_data
    )

    # determine twitch widths of compressed data
    widths_dict = peak_detection.twitch_widths(peak_valley_tuple, compressed_data)

    compress_peakind = np.concatenate([peak_valley_tuple[0], peak_valley_tuple[1]])
    compress_peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A6_compressed.png")

    # plot compressed data for visual inspection
    plt.figure()
    plt.plot(compressed_data[0], compressed_data[1])
    plt.plot(
        compressed_data[0][compress_peakind], compressed_data[1][compress_peakind], "ro"
    )
    plt.xlabel("Time (centimilliseconds)")
    plt.ylabel("Voltage (V)")
    plt.savefig(my_local_path_graphs)

    # make sure sampling rate has been reduced by at least 75%
    assert new_sample_rate <= (original_sampling_rate * COMPRESSION_FACTOR)

    # make sure data metrics have not been altered heavily
    assert (
        np.absolute(
            window_dict_original[AUC_UUID]["mean"] - window_dict[AUC_UUID]["mean"]
        )
        / window_dict_original[AUC_UUID]["mean"]
    ) < COMPRESSION_ACCURACY
    this_twitch_idx = 88000
    assert (
        np.absolute(
            per_beat_dict_original[this_twitch_idx][AUC_UUID]
            - per_beat_dict[this_twitch_idx][AUC_UUID]
        )
        / per_beat_dict_original[this_twitch_idx][AUC_UUID]
    ) < COMPRESSION_ACCURACY

    this_twitch_idx = 201000
    assert (
        np.absolute(
            widths_dict_original[this_twitch_idx][90][2]
            - widths_dict[this_twitch_idx][90][2]
        )
        / widths_dict_original[this_twitch_idx][90][2]
    ) < COMPRESSION_ACCURACY
