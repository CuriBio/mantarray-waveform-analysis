# -*- coding: utf-8 -*-
import os

from mantarray_waveform_analysis import AMPLITUDE_UUID
from mantarray_waveform_analysis import AUC_UUID
from mantarray_waveform_analysis import peak_detection
from mantarray_waveform_analysis import TWITCH_PERIOD_UUID
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .fixtures_compression import fixture_new_A1
from .fixtures_compression import fixture_new_A2
from .fixtures_compression import fixture_new_A3
from .fixtures_compression import fixture_new_A4
from .fixtures_compression import fixture_new_A5
from .fixtures_compression import fixture_new_A6
from .fixtures_peak_detection import fixture_maiden_voyage_data
from .fixtures_utils import PATH_TO_PNGS

matplotlib.use("Agg")


__fixtures__ = [
    fixture_maiden_voyage_data,
    fixture_new_A1,
    fixture_new_A2,
    fixture_new_A3,
    fixture_new_A4,
    fixture_new_A5,
    fixture_new_A6,
]


def assert_percent_diff(actual, expected):
    percent_diff = abs(actual - expected) / expected
    assert percent_diff < 0.0006


def _plot_data(time_series, noise_free_data, data, peakind, my_local_path_graphs):
    # plot and save results
    plt.figure()
    plt.plot(time_series, data[1])
    plt.plot(time_series[peakind], data[1][peakind], "ro")
    plt.xlabel("Time (centimilliseconds)")
    plt.ylabel("Voltage (V)")
    plt.savefig(my_local_path_graphs)


def _plot_twitch_widths(time_series, data, widths_dict, my_local_path_graphs):
    # plot and save results
    plt.figure()
    plt.plot(time_series, data[1, :])
    label_made = False
    for twitch in widths_dict:
        percent_dict = widths_dict[twitch]
        clrs = [
            "r",
            "g",
            "b",
            "c",
            "m",
            "y",
            "k",
            "darkred",
            "darkgreen",
            "darkblue",
            "darkcyan",
            "darkmagenta",
            "orange",
            "gray",
            "lime",
            "crimson",
            "yellow",
        ]
        count = 0
        for percent in percent_dict:
            if not label_made:
                plt.plot(
                    percent_dict[percent][0][0],
                    percent_dict[percent][0][1],
                    "o",
                    color=clrs[count],
                    label=percent,
                )
                plt.plot(
                    percent_dict[percent][1][0],
                    percent_dict[percent][1][1],
                    "o",
                    color=clrs[count],
                )
            else:
                plt.plot(
                    percent_dict[percent][0][0],
                    percent_dict[percent][0][1],
                    "o",
                    color=clrs[count],
                )
                plt.plot(
                    percent_dict[percent][1][0],
                    percent_dict[percent][1][1],
                    "o",
                    color=clrs[count],
                )
            count += 1
            if count > 16:
                label_made = True
    plt.legend(loc="best")
    plt.xlabel("Time (centimilliseconds)")
    plt.ylabel("Voltage (V)")
    plt.savefig(my_local_path_graphs)


def test_maiden_voyage_data_period(maiden_voyage_data):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = maiden_voyage_data

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["n"], 10)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["mean"], 80750)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["std"], 23902)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["min"], 45500)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["max"], 119500)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[77000][TWITCH_PERIOD_UUID], 93000)
    assert_percent_diff(per_beat_dict[170000][TWITCH_PERIOD_UUID], 93000)
    assert_percent_diff(per_beat_dict[263000][TWITCH_PERIOD_UUID], 80000)


def test_maiden_voyage_data_amplitude(maiden_voyage_data):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = maiden_voyage_data

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["n"], 11)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["mean"], 471835)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["std"], 36008)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["min"], 394534)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["max"], 516590)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[77000][AMPLITUDE_UUID], 492263)
    assert_percent_diff(per_beat_dict[170000][AMPLITUDE_UUID], 516590)
    assert_percent_diff(per_beat_dict[263000][AMPLITUDE_UUID], 451850)


def test_maiden_voyage_data_auc(maiden_voyage_data):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = maiden_voyage_data

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AUC_UUID]["n"], 11)
    assert_percent_diff(window_dict[AUC_UUID]["mean"], 26042829121)
    assert_percent_diff(window_dict[AUC_UUID]["std"], 1890944885)
    assert_percent_diff(window_dict[AUC_UUID]["min"], 22778633628)
    assert_percent_diff(window_dict[AUC_UUID]["max"], 28953333906)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[77000][AUC_UUID], 27209375212)
    assert_percent_diff(per_beat_dict[170000][AUC_UUID], 28953333906)
    assert_percent_diff(per_beat_dict[263000][AUC_UUID], 24541063774)


def test_maiden_voyage_data_twitch_widths(maiden_voyage_data):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, _ = maiden_voyage_data

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peakind, data)

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "maiden_voyage_data_widths.png")

    # plot and save results
    _plot_twitch_widths(time_series, data, widths_dict, my_local_path_graphs)

    assert_percent_diff(widths_dict[77000][10][2], 40500)
    assert_percent_diff(widths_dict[170000][50][2], 57500)
    assert_percent_diff(widths_dict[263000][90][2], 68500)


def test_maiden_voyage_data_peak_detection(maiden_voyage_data):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, noise_free_data = maiden_voyage_data

    # find time and voltages
    peaks_dict, valleys_dict = peak_detection.time_voltage_dict_creation(
        data[0, :], noise_free_data[1, :], peakind
    )

    peakind = np.concatenate([peakind[0], peakind[1]])
    peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "maiden_voyage_peaks.png")

    # plot and save results
    _plot_data(time_series, noise_free_data, data, peakind, my_local_path_graphs)

    actual_values = [
        30,
        84,
        154,
        247,
        340,
        413,
        526,
        576,
        686,
        739,
        862,
        899,
        960,
        1059,
        1185,
        1226,
        1309,
        1383,
        1439,
        1549,
        1678,
        1712,
        1769,
        1875,
        1942,
    ]

    # tests peak_detector & noise_cancellation
    assert np.array_equal(peakind, actual_values)

    # tests time_voltage_dict_creation
    assert (len(peaks_dict) + len(valleys_dict)) == len(peakind)


def test_new_A1_period(new_A1):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A1

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["n"], 11)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["mean"], 80182)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["std"], 1696)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["min"], 78000)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["max"], 84000)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[105000][TWITCH_PERIOD_UUID], 81000)
    assert_percent_diff(per_beat_dict[186000][TWITCH_PERIOD_UUID], 80000)
    assert_percent_diff(per_beat_dict[266000][TWITCH_PERIOD_UUID], 78000)


def test_new_A1_amplitude(new_A1):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A1

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["n"], 12)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["mean"], 97887)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["std"], 1993)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["min"], 94961)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["max"], 102159)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[105000][AMPLITUDE_UUID], 102159)
    assert_percent_diff(per_beat_dict[186000][AMPLITUDE_UUID], 98768)
    assert_percent_diff(per_beat_dict[266000][AMPLITUDE_UUID], 97879)


def test_new_A1_auc(new_A1):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A1

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AUC_UUID]["n"], 12)
    assert_percent_diff(window_dict[AUC_UUID]["mean"], 2565504261)
    assert_percent_diff(window_dict[AUC_UUID]["std"], 65869124)
    assert_percent_diff(window_dict[AUC_UUID]["min"], 2465789418)
    assert_percent_diff(window_dict[AUC_UUID]["max"], 2695170612)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[105000][AUC_UUID], 2695170612)
    assert_percent_diff(per_beat_dict[186000][AUC_UUID], 2577932134)
    assert_percent_diff(per_beat_dict[266000][AUC_UUID], 2566184048)


def test_new_A1_twitch_widths(new_A1):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, _ = new_A1

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peakind, data)

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A1_widths.png")

    # plot and save results
    _plot_twitch_widths(time_series, data, widths_dict, my_local_path_graphs)

    assert_percent_diff(widths_dict[105000][10][2], 10000)
    assert_percent_diff(widths_dict[186000][50][2], 26000)
    assert_percent_diff(widths_dict[266000][90][2], 45000)


def test_new_A1(new_A1):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, noise_free_data = new_A1

    # find time and voltages
    peaks_dict, valleys_dict = peak_detection.time_voltage_dict_creation(
        data[0, :], noise_free_data[1, :], peakind
    )

    peakind = np.concatenate([peakind[0], peakind[1]])
    peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A1_peaks.png")

    # plot and save results
    _plot_data(time_series, noise_free_data, data, peakind, my_local_path_graphs)

    actual_values = [
        3,
        24,
        70,
        105,
        147,
        186,
        220,
        266,
        305,
        344,
        397,
        424,
        463,
        502,
        555,
        586,
        628,
        667,
        713,
        745,
        779,
        825,
        871,
        906,
        963,
        987,
    ]

    # tests peak_detector & noise_cancellation
    assert np.array_equal(peakind, actual_values)

    # tests time_voltage_dict_creation
    assert (len(peaks_dict) + len(valleys_dict)) == len(peakind)


def test_new_A2_period(new_A2):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A2

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["n"], 10)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["mean"], 80200)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["std"], 2400)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["min"], 77000)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["max"], 85000)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[104000][TWITCH_PERIOD_UUID], 81000)
    assert_percent_diff(per_beat_dict[185000][TWITCH_PERIOD_UUID], 77000)
    assert_percent_diff(per_beat_dict[262000][TWITCH_PERIOD_UUID], 85000)


def test_new_A2_amplitude(new_A2):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A2

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["n"], 11)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["mean"], 91576)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["std"], 1872)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["min"], 88541)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["max"], 94583)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[104000][AMPLITUDE_UUID], 90034)
    assert_percent_diff(per_beat_dict[185000][AMPLITUDE_UUID], 93547)
    assert_percent_diff(per_beat_dict[262000][AMPLITUDE_UUID], 94583)


def test_new_A2_auc(new_A2):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A2

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AUC_UUID]["n"], 11)
    assert_percent_diff(window_dict[AUC_UUID]["mean"], 2381135373)
    assert_percent_diff(window_dict[AUC_UUID]["std"], 57692350)
    assert_percent_diff(window_dict[AUC_UUID]["min"], 2278734037)
    assert_percent_diff(window_dict[AUC_UUID]["max"], 2491134876)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[104000][AUC_UUID], 2372614155)
    assert_percent_diff(per_beat_dict[185000][AUC_UUID], 2393152678)
    assert_percent_diff(per_beat_dict[262000][AUC_UUID], 2491134876)


def test_new_A2_twitch_widths(new_A2):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, _ = new_A2

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peakind, data)

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A2_widths.png")

    # plot and save results
    _plot_twitch_widths(time_series, data, widths_dict, my_local_path_graphs)

    assert_percent_diff(widths_dict[104000][10][2], 10000)
    assert_percent_diff(widths_dict[185000][50][2], 25000)
    assert_percent_diff(widths_dict[262000][90][2], 43000)


def test_new_A2(new_A2):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, noise_free_data = new_A2

    # find time and voltages
    peaks_dict, valleys_dict = peak_detection.time_voltage_dict_creation(
        data[0, :], noise_free_data[1, :], peakind
    )

    peakind = np.concatenate([peakind[0], peakind[1]])
    peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A2_peaks.png")

    # plot and save results
    _plot_data(time_series, noise_free_data, data, peakind, my_local_path_graphs)

    actual_values = [
        24,
        58,
        104,
        164,
        185,
        235,
        262,
        308,
        347,
        393,
        424,
        466,
        505,
        565,
        586,
        643,
        663,
        716,
        744,
        797,
        825,
        867,
        906,
        947,
        986,
    ]

    # tests peak_detector & noise_cancellation
    assert np.array_equal(peakind, actual_values)

    # tests time_voltage_dict_creation
    assert (len(peaks_dict) + len(valleys_dict)) == len(peakind)


def test_new_A3_period(new_A3):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A3

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["n"], 11)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["mean"], 80182)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["std"], 4386)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["min"], 73000)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["max"], 85000)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[108000][TWITCH_PERIOD_UUID], 85000)
    assert_percent_diff(per_beat_dict[193000][TWITCH_PERIOD_UUID], 73000)
    assert_percent_diff(per_beat_dict[266000][TWITCH_PERIOD_UUID], 85000)


def test_new_A3_amplitude(new_A3):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A3

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["n"], 12)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["mean"], 65018)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["std"], 2055)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["min"], 62540)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["max"], 68058)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[108000][AMPLITUDE_UUID], 63705)
    assert_percent_diff(per_beat_dict[193000][AMPLITUDE_UUID], 67594)
    assert_percent_diff(per_beat_dict[266000][AMPLITUDE_UUID], 62540)


def test_new_A3_auc(new_A3):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A3

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AUC_UUID]["n"], 12)
    assert_percent_diff(window_dict[AUC_UUID]["mean"], 2023742808)
    assert_percent_diff(window_dict[AUC_UUID]["std"], 47641340)
    assert_percent_diff(window_dict[AUC_UUID]["min"], 1969429673)
    assert_percent_diff(window_dict[AUC_UUID]["max"], 2127972146)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[108000][AUC_UUID], 2127972146)
    assert_percent_diff(per_beat_dict[193000][AUC_UUID], 2006425706)
    assert_percent_diff(per_beat_dict[266000][AUC_UUID], 2037515623)


def test_new_A3_twitch_widths(new_A3):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, _ = new_A3

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peakind, data)

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A3_widths.png")

    # plot and save results
    _plot_twitch_widths(time_series, data, widths_dict, my_local_path_graphs)

    assert_percent_diff(widths_dict[108000][10][2], 13000)
    assert_percent_diff(widths_dict[193000][50][2], 29000)
    assert_percent_diff(widths_dict[266000][90][2], 65000)


def test_new_A3(new_A3):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, noise_free_data = new_A3

    # find time and voltages
    peaks_dict, valleys_dict = peak_detection.time_voltage_dict_creation(
        data[0, :], noise_free_data[1, :], peakind
    )

    peakind = np.concatenate([peakind[0], peakind[1]])
    peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A3_peaks.png")

    # plot and save results
    _plot_data(time_series, noise_free_data, data, peakind, my_local_path_graphs)

    actual_values = [
        3,
        28,
        62,
        108,
        165,
        193,
        239,
        266,
        324,
        351,
        404,
        428,
        470,
        509,
        562,
        594,
        628,
        667,
        727,
        752,
        805,
        832,
        871,
        910,
        963,
        995,
    ]

    # tests peak_detector & noise_cancellation
    assert np.array_equal(peakind, actual_values)

    # tests time_voltage_dict_creation
    assert (len(peaks_dict) + len(valleys_dict)) == len(peakind)


def test_new_A4_period(new_A4):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A4

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["n"], 15)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["mean"], 57667)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["std"], 1247)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["min"], 56000)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["max"], 59000)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[81000][TWITCH_PERIOD_UUID], 56000)
    assert_percent_diff(per_beat_dict[137000][TWITCH_PERIOD_UUID], 59000)
    assert_percent_diff(per_beat_dict[196000][TWITCH_PERIOD_UUID], 59000)


def test_new_A4_amplitude(new_A4):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A4

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["n"], 16)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["mean"], 127583)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["std"], 3405)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["min"], 120823)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["max"], 132344)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[81000][AMPLITUDE_UUID], 128961)
    assert_percent_diff(per_beat_dict[137000][AMPLITUDE_UUID], 132298)
    assert_percent_diff(per_beat_dict[196000][AMPLITUDE_UUID], 125759)


def test_new_A4_auc(new_A4):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A4

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AUC_UUID]["n"], 16)
    assert_percent_diff(window_dict[AUC_UUID]["mean"], 2834631743)
    assert_percent_diff(window_dict[AUC_UUID]["std"], 120717341)
    assert_percent_diff(window_dict[AUC_UUID]["min"], 2639460824)
    assert_percent_diff(window_dict[AUC_UUID]["max"], 3032386646)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[81000][AUC_UUID], 2798486844)
    assert_percent_diff(per_beat_dict[137000][AUC_UUID], 3032386646)
    assert_percent_diff(per_beat_dict[196000][AUC_UUID], 2851895189)


def test_new_A4_twitch_widths(new_A4):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, _ = new_A4

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peakind, data)

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A4_widths.png")

    # plot and save results
    _plot_twitch_widths(time_series, data, widths_dict, my_local_path_graphs)

    assert_percent_diff(widths_dict[81000][10][2], 9000)
    assert_percent_diff(widths_dict[137000][50][2], 21000)
    assert_percent_diff(widths_dict[196000][90][2], 36000)


def test_new_A4(new_A4):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, noise_free_data = new_A4

    # find time and voltages
    peaks_dict, valleys_dict = peak_detection.time_voltage_dict_creation(
        data[0, :], noise_free_data[1, :], peakind
    )

    peakind = np.concatenate([peakind[0], peakind[1]])
    peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A4_peaks.png")

    # plot and save results
    _plot_data(time_series, noise_free_data, data, peakind, my_local_path_graphs)

    actual_values = [
        23,
        58,
        81,
        117,
        137,
        176,
        196,
        235,
        255,
        290,
        311,
        349,
        369,
        407,
        427,
        466,
        486,
        518,
        545,
        584,
        601,
        639,
        659,
        697,
        715,
        756,
        774,
        815,
        832,
        867,
        890,
        926,
        946,
        985,
    ]

    # tests peak_detector & noise_cancellation
    assert np.array_equal(peakind, actual_values)

    # tests time_voltage_dict_creation
    assert (len(peaks_dict) + len(valleys_dict)) == len(peakind)


def test_new_A5_period(new_A5):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A5

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["n"], 16)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["mean"], 57812)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["std"], 1424)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["min"], 55000)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["max"], 59000)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[80000][TWITCH_PERIOD_UUID], 58000)
    assert_percent_diff(per_beat_dict[138000][TWITCH_PERIOD_UUID], 59000)
    assert_percent_diff(per_beat_dict[197000][TWITCH_PERIOD_UUID], 58000)


def test_new_A5_amplitude(new_A5):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A5

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["n"], 17)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["mean"], 53213)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["std"], 1189)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["min"], 51582)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["max"], 56040)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[80000][AMPLITUDE_UUID], 51879)
    assert_percent_diff(per_beat_dict[138000][AMPLITUDE_UUID], 51582)
    assert_percent_diff(per_beat_dict[197000][AMPLITUDE_UUID], 54132)


def test_new_A5_auc(new_A5):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A5

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AUC_UUID]["n"], 17)
    assert_percent_diff(window_dict[AUC_UUID]["mean"], 1152075794)
    assert_percent_diff(window_dict[AUC_UUID]["std"], 42595033)
    assert_percent_diff(window_dict[AUC_UUID]["min"], 1078044570)
    assert_percent_diff(window_dict[AUC_UUID]["max"], 1275633877)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[80000][AUC_UUID], 1148297870)
    assert_percent_diff(per_beat_dict[138000][AUC_UUID], 1133384235)
    assert_percent_diff(per_beat_dict[197000][AUC_UUID], 1152377663)


def test_new_A5_twitch_widths(new_A5):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, _ = new_A5

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peakind, data)

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A5_widths.png")

    # plot and save results
    _plot_twitch_widths(time_series, data, widths_dict, my_local_path_graphs)

    assert_percent_diff(widths_dict[80000][10][2], 8000)
    assert_percent_diff(widths_dict[138000][50][2], 21000)
    assert_percent_diff(widths_dict[197000][90][2], 38000)


def test_new_A5(new_A5):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, noise_free_data = new_A5

    # find time and voltages
    peaks_dict, valleys_dict = peak_detection.time_voltage_dict_creation(
        data[0, :], noise_free_data[1, :], peakind
    )

    peakind = np.concatenate([peakind[0], peakind[1]])
    peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A5_peaks.png")

    # plot and save results
    _plot_data(time_series, noise_free_data, data, peakind, my_local_path_graphs)

    actual_values = [
        4,
        25,
        62,
        80,
        121,
        138,
        180,
        197,
        232,
        255,
        294,
        311,
        345,
        370,
        411,
        428,
        463,
        487,
        522,
        545,
        584,
        601,
        640,
        660,
        694,
        718,
        753,
        777,
        805,
        836,
        871,
        891,
        930,
        950,
        989,
    ]

    # tests peak_detector & noise_cancellation
    assert np.array_equal(peakind, actual_values)

    # tests time_voltage_dict_creation
    assert (len(peaks_dict) + len(valleys_dict)) == len(peakind)


def test_new_A6_period(new_A6):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A6

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["n"], 16)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["mean"], 57625)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["std"], 4768)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["min"], 48000)
    assert_percent_diff(window_dict[TWITCH_PERIOD_UUID]["max"], 66000)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[88000][TWITCH_PERIOD_UUID], 60000)
    assert_percent_diff(per_beat_dict[148000][TWITCH_PERIOD_UUID], 53000)
    assert_percent_diff(per_beat_dict[201000][TWITCH_PERIOD_UUID], 54000)


def test_new_A6_amplitude(new_A6):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A6

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["n"], 17)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["mean"], 8462)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["std"], 607)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["min"], 7387)
    assert_percent_diff(window_dict[AMPLITUDE_UUID]["max"], 9520)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[88000][AMPLITUDE_UUID], 9284)
    assert_percent_diff(per_beat_dict[148000][AMPLITUDE_UUID], 8548)
    assert_percent_diff(per_beat_dict[201000][AMPLITUDE_UUID], 8570)


def test_new_A6_auc(new_A6):
    # data creation, noise cancellation, peak detection
    _, _, peakind, _, noise_free_data = new_A6

    # determine data metrics
    per_beat_dict, window_dict = peak_detection.data_metrics(peakind, noise_free_data)

    # test data_metrics aggregate dictionary
    assert_percent_diff(window_dict[AUC_UUID]["n"], 17)
    assert_percent_diff(window_dict[AUC_UUID]["mean"], 222102904)
    assert_percent_diff(window_dict[AUC_UUID]["std"], 25056302)
    assert_percent_diff(window_dict[AUC_UUID]["min"], 173375434)
    assert_percent_diff(window_dict[AUC_UUID]["max"], 270066889)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_beat_dict[88000][AUC_UUID], 270066889)
    assert_percent_diff(per_beat_dict[148000][AUC_UUID], 246401757)
    assert_percent_diff(per_beat_dict[201000][AUC_UUID], 216833502)


def test_new_A6_twitch_widths(new_A6):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, _ = new_A6

    # determine twitch widths
    widths_dict = peak_detection.twitch_widths(peakind, data)

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A6_widths.png")

    # plot and save results
    _plot_twitch_widths(time_series, data, widths_dict, my_local_path_graphs)

    assert_percent_diff(widths_dict[88000][10][2], 12000)
    assert_percent_diff(widths_dict[148000][50][2], 32000)
    assert_percent_diff(widths_dict[201000][90][2], 55000)


def test_new_A6(new_A6):
    # data creation, noise cancellation, peak detection
    data, time_series, peakind, _, noise_free_data = new_A6

    # find time and voltages
    peaks_dict, valleys_dict = peak_detection.time_voltage_dict_creation(
        data[0, :], noise_free_data[1, :], peakind
    )

    peakind = np.concatenate([peakind[0], peakind[1]])
    peakind.sort()

    # creating local path for plotting
    my_local_path_graphs = os.path.join(PATH_TO_PNGS, "new_A6_peaks.png")

    # plot and save results
    _plot_data(time_series, noise_free_data, data, peakind, my_local_path_graphs)

    actual_values = [
        6,
        31,
        51,
        88,
        117,
        148,
        176,
        201,
        238,
        255,
        294,
        318,
        341,
        376,
        400,
        428,
        459,
        490,
        525,
        552,
        580,
        604,
        625,
        663,
        698,
        722,
        757,
        784,
        808,
        832,
        874,
        898,
        926,
        953,
        981,
    ]

    # tests peak_detector & noise_cancellation
    assert np.array_equal(peakind, actual_values)

    # tests time_voltage_dict_creation
    assert (len(peaks_dict) + len(valleys_dict)) == len(peakind)
