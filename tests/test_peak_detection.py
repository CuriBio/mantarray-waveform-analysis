# -*- coding: utf-8 -*-
import os

from mantarray_waveform_analysis import AMPLITUDE_UUID
from mantarray_waveform_analysis import AUC_UUID
from mantarray_waveform_analysis import find_twitch_indices
from mantarray_waveform_analysis import MIN_NUMBER_PEAKS
from mantarray_waveform_analysis import peak_detector
from mantarray_waveform_analysis import PRIOR_PEAK_INDEX_UUID
from mantarray_waveform_analysis import PRIOR_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import SUBSEQUENT_PEAK_INDEX_UUID
from mantarray_waveform_analysis import SUBSEQUENT_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import TooFewPeaksDetectedError
from mantarray_waveform_analysis import TWITCH_FREQUENCY_UUID
from mantarray_waveform_analysis import TWITCH_PERIOD_UUID
from mantarray_waveform_analysis import TwoPeaksInARowError
from mantarray_waveform_analysis import TwoValleysInARowError
from mantarray_waveform_analysis import WIDTH_FALLING_COORDS_UUID
from mantarray_waveform_analysis import WIDTH_RISING_COORDS_UUID
from mantarray_waveform_analysis import WIDTH_UUID
from mantarray_waveform_analysis import WIDTH_VALUE_UUID
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import approx

from .fixtures_compression import fixture_new_A1
from .fixtures_compression import fixture_new_A2
from .fixtures_compression import fixture_new_A3
from .fixtures_compression import fixture_new_A4
from .fixtures_compression import fixture_new_A5
from .fixtures_compression import fixture_new_A6
from .fixtures_peak_detection import fixture_maiden_voyage_data
from .fixtures_peak_detection import fixture_noisy_data_A1
from .fixtures_peak_detection import fixture_noisy_data_B1
from .fixtures_utils import _get_data_metrics
from .fixtures_utils import _load_file_tsv
from .fixtures_utils import _plot_data
from .fixtures_utils import assert_percent_diff
from .fixtures_utils import create_numpy_array_of_raw_gmr_from_python_arrays
from .fixtures_utils import PATH_TO_DATASETS
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
    fixture_noisy_data_A1,
    fixture_noisy_data_B1,
]


def _plot_twitch_widths(filtered_data, per_twitch_dict, my_local_path_graphs):
    # plot and save results
    plt.figure()
    plt.plot(filtered_data[0, :], filtered_data[1, :])
    label_made = False
    for twitch in per_twitch_dict:
        percent_dict = per_twitch_dict[twitch][WIDTH_UUID]
        clrs = [
            "k",
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
        for percent in reversed(list(percent_dict.keys())):
            rising_x = percent_dict[percent][WIDTH_RISING_COORDS_UUID][0]
            falling_x = percent_dict[percent][WIDTH_FALLING_COORDS_UUID][0]
            if not label_made:
                plt.plot(
                    rising_x,
                    percent_dict[percent][WIDTH_RISING_COORDS_UUID][1],
                    "o",
                    color=clrs[count],
                    label=percent,
                )
                plt.plot(
                    falling_x,
                    percent_dict[percent][WIDTH_FALLING_COORDS_UUID][1],
                    "o",
                    color=clrs[count],
                )
            else:
                plt.plot(
                    rising_x,
                    percent_dict[percent][WIDTH_RISING_COORDS_UUID][1],
                    "o",
                    color=clrs[count],
                )
                plt.plot(
                    falling_x,
                    percent_dict[percent][WIDTH_FALLING_COORDS_UUID][1],
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


def test_new_A1_period(new_A1):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A1)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["n"] == 11
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"] == approx(80182, rel=1e-5)
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["std"], 1696)
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["min"] == 78000
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["max"] == 84000
    # test data_metrics per beat dictionary
    assert per_twitch_dict[105000][TWITCH_PERIOD_UUID] == 81000
    assert per_twitch_dict[186000][TWITCH_PERIOD_UUID] == 80000
    assert per_twitch_dict[266000][TWITCH_PERIOD_UUID] == 78000


def test_new_A1_frequency(new_A1):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A1)

    assert aggregate_metrics_dict[TWITCH_FREQUENCY_UUID]["mean"] == approx(
        1.2477183310516644
    )
    assert aggregate_metrics_dict[TWITCH_FREQUENCY_UUID]["std"] == approx(
        0.026146910973044845
    )
    assert per_twitch_dict[105000][TWITCH_FREQUENCY_UUID] == approx(1.2345679)


def test_new_A2_period(new_A2):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A2)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["n"] == 11
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"], 80182)
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["std"], 2289)
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["min"] == 77000
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["max"] == 85000
    # test data_metrics per beat dictionary
    assert per_twitch_dict[104000][TWITCH_PERIOD_UUID] == 81000
    assert per_twitch_dict[185000][TWITCH_PERIOD_UUID] == 77000
    assert per_twitch_dict[262000][TWITCH_PERIOD_UUID] == 85000


def test_new_A3_period(new_A3):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A3)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["n"] == 10
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"], 80182)
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["std"], 4600)
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["min"] == 73000
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["max"] == 85000
    # test data_metrics per beat dictionary
    assert per_twitch_dict[108000][TWITCH_PERIOD_UUID] == 85000
    assert per_twitch_dict[193000][TWITCH_PERIOD_UUID] == 73000
    assert per_twitch_dict[266000][TWITCH_PERIOD_UUID] == 85000


def test_new_A4_period(new_A4):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A4)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["n"] == 15
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"], 57667)
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["std"], 1247)
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["min"] == 56000
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["max"] == 59000
    # test data_metrics per beat dictionary
    assert per_twitch_dict[81000][TWITCH_PERIOD_UUID] == 56000
    assert per_twitch_dict[137000][TWITCH_PERIOD_UUID] == 59000
    assert per_twitch_dict[196000][TWITCH_PERIOD_UUID] == 59000


def test_new_A5_period(new_A5):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A5)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["n"] == 15
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"], 58000)
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["std"], 1265)
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["min"] == 55000
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["max"] == 59000
    # test data_metrics per beat dictionary
    assert per_twitch_dict[80000][TWITCH_PERIOD_UUID] == 58000
    assert per_twitch_dict[138000][TWITCH_PERIOD_UUID] == 59000
    assert per_twitch_dict[197000][TWITCH_PERIOD_UUID] == 58000


def test_new_A6_period(new_A6):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A6)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["n"] == 15
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"], 57667)
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["std"], 4922)
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["min"] == 48000
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["max"] == 66000
    # test data_metrics per beat dictionary
    assert per_twitch_dict[88000][TWITCH_PERIOD_UUID] == 60000
    assert per_twitch_dict[148000][TWITCH_PERIOD_UUID] == 53000
    assert per_twitch_dict[201000][TWITCH_PERIOD_UUID] == 54000


def test_maiden_voyage_data_period(maiden_voyage_data):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(maiden_voyage_data)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["n"] == 10
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["mean"], 81400)
    assert_percent_diff(aggregate_metrics_dict[TWITCH_PERIOD_UUID]["std"], 1480)
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["min"] == 78500
    assert aggregate_metrics_dict[TWITCH_PERIOD_UUID]["max"] == 83500

    # test data_metrics per beat dictionary
    assert per_twitch_dict[123500][TWITCH_PERIOD_UUID] == 83000
    assert per_twitch_dict[449500][TWITCH_PERIOD_UUID] == 80000
    assert per_twitch_dict[856000][TWITCH_PERIOD_UUID] == 81500


def test_new_A1_amplitude(new_A1):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A1)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["n"] == 11
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["mean"], 103286)
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["std"], 1855)
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["min"] == 100953
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["max"] == 106274

    # test data_metrics per beat dictionary
    assert per_twitch_dict[105000][AMPLITUDE_UUID] == 106274
    assert per_twitch_dict[186000][AMPLITUDE_UUID] == 104624
    assert per_twitch_dict[266000][AMPLITUDE_UUID] == 102671


def test_new_A2_amplitude(new_A2):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A2)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["n"] == 11
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["mean"], 95231)
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["std"], 1731)
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["min"] == 92662
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["max"] == 98873

    # test data_metrics per beat dictionary
    assert per_twitch_dict[104000][AMPLITUDE_UUID] == 93844
    assert per_twitch_dict[185000][AMPLITUDE_UUID] == 95950
    assert per_twitch_dict[262000][AMPLITUDE_UUID] == 98873


def test_new_A3_amplitude(new_A3):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A3)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["n"] == 10
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["mean"], 70491)
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["std"], 2136)
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["min"] == 67811
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["max"] == 73363

    # test data_metrics per beat dictionary
    assert per_twitch_dict[108000][AMPLITUDE_UUID] == 67811
    assert per_twitch_dict[193000][AMPLITUDE_UUID] == 73363
    assert per_twitch_dict[266000][AMPLITUDE_UUID] == 67866


def test_new_A4_amplitude(new_A4):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A4)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["n"] == 15
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["mean"], 130440)
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["std"], 3416)
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["min"] == 124836
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["max"] == 136096

    # test data_metrics per beat dictionary
    assert per_twitch_dict[81000][AMPLITUDE_UUID] == 131976
    assert per_twitch_dict[137000][AMPLITUDE_UUID] == 136096
    assert per_twitch_dict[196000][AMPLITUDE_UUID] == 129957


def test_new_A5_amplitude(new_A5):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A5)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["n"] == 15
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["mean"], 55863)
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["std"], 1144)
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["min"] == 54291
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["max"] == 58845

    # test data_metrics per beat dictionary
    assert per_twitch_dict[80000][AMPLITUDE_UUID] == 54540
    assert per_twitch_dict[138000][AMPLITUDE_UUID] == 54739
    assert per_twitch_dict[197000][AMPLITUDE_UUID] == 56341


def test_new_A6_amplitude(new_A6):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A6)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["n"] == 15
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["mean"], 10265)
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["std"], 568)
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["min"] == 9056
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["max"] == 11052

    # test data_metrics per beat dictionary
    assert per_twitch_dict[88000][AMPLITUDE_UUID] == 10761
    assert per_twitch_dict[148000][AMPLITUDE_UUID] == 10486
    assert per_twitch_dict[201000][AMPLITUDE_UUID] == 10348


def test_maiden_voyage_data_amplitude(maiden_voyage_data):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(maiden_voyage_data)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["n"] == 10
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["mean"], 477098)
    assert_percent_diff(aggregate_metrics_dict[AMPLITUDE_UUID]["std"], 40338)
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["min"] == 416896
    assert aggregate_metrics_dict[AMPLITUDE_UUID]["max"] == 531133

    # test data_metrics per beat dictionary
    assert per_twitch_dict[123500][AMPLITUDE_UUID] == 523198
    assert per_twitch_dict[449500][AMPLITUDE_UUID] == 435673
    assert per_twitch_dict[856000][AMPLITUDE_UUID] == 464154


def test_new_A1_twitch_widths(new_A1):
    pipeline, _ = new_A1
    filtered_data = pipeline.get_noise_filtered_gmr()
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A1)

    # plot and save results
    _plot_twitch_widths(
        filtered_data, per_twitch_dict, os.path.join(PATH_TO_PNGS, "new_A1_widths.png")
    )

    assert per_twitch_dict[105000][WIDTH_UUID][10][WIDTH_VALUE_UUID] == 10768
    assert per_twitch_dict[186000][WIDTH_UUID][50][WIDTH_VALUE_UUID] == 25340
    assert per_twitch_dict[266000][WIDTH_UUID][90][WIDTH_VALUE_UUID] == 43566

    assert per_twitch_dict[105000][WIDTH_UUID][10][WIDTH_FALLING_COORDS_UUID] == (
        109494,
        -211000,
    )
    assert per_twitch_dict[186000][WIDTH_UUID][50][WIDTH_RISING_COORDS_UUID] == (
        171482,
        -167630,
    )

    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][20]["mean"], 15758)
    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][50]["std"], 422)
    assert aggregate_metrics_dict[WIDTH_UUID][80]["min"] == 35534
    assert aggregate_metrics_dict[WIDTH_UUID][90]["max"] == 46182


def test_new_A2_twitch_widths(new_A2):
    pipeline, _ = new_A2
    filtered_data = pipeline.get_noise_filtered_gmr()
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A2)

    # plot and save results
    _plot_twitch_widths(
        filtered_data, per_twitch_dict, os.path.join(PATH_TO_PNGS, "new_A2_widths.png")
    )

    assert per_twitch_dict[104000][WIDTH_UUID][10][WIDTH_VALUE_UUID] == 9937
    assert per_twitch_dict[185000][WIDTH_UUID][50][WIDTH_VALUE_UUID] == 24890
    assert per_twitch_dict[262000][WIDTH_UUID][90][WIDTH_VALUE_UUID] == 43221

    assert per_twitch_dict[104000][WIDTH_UUID][10][WIDTH_FALLING_COORDS_UUID] == (
        109092,
        -51458,
    )
    assert per_twitch_dict[185000][WIDTH_UUID][50][WIDTH_RISING_COORDS_UUID] == (
        171670,
        -14468,
    )

    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][20]["mean"], 15343)
    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][50]["std"], 377)
    assert aggregate_metrics_dict[WIDTH_UUID][80]["min"] == 35131
    assert aggregate_metrics_dict[WIDTH_UUID][90]["max"] == 45237


def test_new_A3_twitch_widths(new_A3):
    pipeline, _ = new_A3
    filtered_data = pipeline.get_noise_filtered_gmr()
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A3)

    # plot and save results
    _plot_twitch_widths(
        filtered_data, per_twitch_dict, os.path.join(PATH_TO_PNGS, "new_A3_widths.png")
    )

    assert per_twitch_dict[108000][WIDTH_UUID][10][WIDTH_VALUE_UUID] == 10789
    assert per_twitch_dict[193000][WIDTH_UUID][50][WIDTH_VALUE_UUID] == 29398
    assert per_twitch_dict[266000][WIDTH_UUID][90][WIDTH_VALUE_UUID] == 46650

    assert per_twitch_dict[108000][WIDTH_UUID][10][WIDTH_FALLING_COORDS_UUID] == (
        113846,
        233811,
    )
    assert per_twitch_dict[193000][WIDTH_UUID][50][WIDTH_RISING_COORDS_UUID] == (
        173960,
        258988,
    )

    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][20]["mean"], 18868)
    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][50]["std"], 449)
    assert aggregate_metrics_dict[WIDTH_UUID][80]["min"] == 39533
    assert aggregate_metrics_dict[WIDTH_UUID][90]["max"] == 68356


def test_new_A4_twitch_widths(new_A4):
    pipeline, _ = new_A4
    filtered_data = pipeline.get_noise_filtered_gmr()
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A4)

    # plot and save results
    _plot_twitch_widths(
        filtered_data, per_twitch_dict, os.path.join(PATH_TO_PNGS, "new_A4_widths.png")
    )

    assert per_twitch_dict[81000][WIDTH_UUID][10][WIDTH_VALUE_UUID] == 8941
    assert per_twitch_dict[137000][WIDTH_UUID][50][WIDTH_VALUE_UUID] == 21595
    assert per_twitch_dict[196000][WIDTH_UUID][90][WIDTH_VALUE_UUID] == 35820

    assert per_twitch_dict[137000][WIDTH_UUID][10][WIDTH_FALLING_COORDS_UUID] == (
        142096,
        -43120,
    )
    assert per_twitch_dict[196000][WIDTH_UUID][50][WIDTH_RISING_COORDS_UUID] == (
        184750,
        13350,
    )

    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][20]["mean"], 13092)
    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][50]["std"], 255)
    assert aggregate_metrics_dict[WIDTH_UUID][80]["min"] == 29258
    assert aggregate_metrics_dict[WIDTH_UUID][90]["max"] == 41550


def test_new_A5_twitch_widths(new_A5):
    pipeline, _ = new_A5
    filtered_data = pipeline.get_noise_filtered_gmr()
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A5)

    # plot and save results
    _plot_twitch_widths(
        filtered_data, per_twitch_dict, os.path.join(PATH_TO_PNGS, "new_A5_widths.png")
    )

    assert per_twitch_dict[80000][WIDTH_UUID][10][WIDTH_VALUE_UUID] == 8051
    assert per_twitch_dict[138000][WIDTH_UUID][50][WIDTH_VALUE_UUID] == 20612
    assert per_twitch_dict[197000][WIDTH_UUID][90][WIDTH_VALUE_UUID] == 37710

    assert per_twitch_dict[138000][WIDTH_UUID][10][WIDTH_FALLING_COORDS_UUID] == (
        143597,
        75362,
    )
    assert per_twitch_dict[197000][WIDTH_UUID][50][WIDTH_RISING_COORDS_UUID] == (
        186268,
        96628,
    )

    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][20]["mean"], 11975)
    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][50]["std"], 430)
    assert aggregate_metrics_dict[WIDTH_UUID][80]["min"] == 29683
    assert aggregate_metrics_dict[WIDTH_UUID][90]["max"] == 44543


def test_new_A6_twitch_widths(new_A6):
    pipeline, _ = new_A6
    filtered_data = pipeline.get_noise_filtered_gmr()
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A6)

    # plot and save results
    _plot_twitch_widths(
        filtered_data, per_twitch_dict, os.path.join(PATH_TO_PNGS, "new_A6_widths.png")
    )

    assert per_twitch_dict[88000][WIDTH_UUID][10][WIDTH_VALUE_UUID] == 4815
    assert per_twitch_dict[148000][WIDTH_UUID][50][WIDTH_VALUE_UUID] == 28278
    assert per_twitch_dict[201000][WIDTH_UUID][90][WIDTH_VALUE_UUID] == 45761

    assert per_twitch_dict[148000][WIDTH_UUID][10][WIDTH_FALLING_COORDS_UUID] == (
        149352,
        55952,
    )
    assert per_twitch_dict[201000][WIDTH_UUID][50][WIDTH_RISING_COORDS_UUID] == (
        187783,
        59651,
    )

    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][20]["mean"], 10211)
    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][50]["std"], 1969)
    assert aggregate_metrics_dict[WIDTH_UUID][80]["min"] == 31854
    assert aggregate_metrics_dict[WIDTH_UUID][90]["max"] == 57570


def test_maiden_voyage_data_twitch_widths(maiden_voyage_data):
    pipeline, _ = maiden_voyage_data
    filtered_data = pipeline.get_noise_filtered_gmr()
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(maiden_voyage_data)

    # plot and save results
    _plot_twitch_widths(
        filtered_data,
        per_twitch_dict,
        os.path.join(PATH_TO_PNGS, "maiden_voyage_data_widths.png"),
    )

    assert per_twitch_dict[123500][WIDTH_UUID][10][WIDTH_VALUE_UUID] == 11711
    assert per_twitch_dict[449500][WIDTH_UUID][50][WIDTH_VALUE_UUID] == 23687
    assert per_twitch_dict[856000][WIDTH_UUID][90][WIDTH_VALUE_UUID] == 33911
    assert per_twitch_dict[123500][WIDTH_UUID][10][WIDTH_FALLING_COORDS_UUID] == (
        129431,
        301839,
    )
    assert per_twitch_dict[449500][WIDTH_UUID][50][WIDTH_RISING_COORDS_UUID] == (
        437648,
        562594,
    )

    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][20]["mean"], 16327)
    assert_percent_diff(aggregate_metrics_dict[WIDTH_UUID][50]["std"], 467)
    assert aggregate_metrics_dict[WIDTH_UUID][80]["min"] == 30252
    assert aggregate_metrics_dict[WIDTH_UUID][90]["max"] == 66476


def test_new_A1_auc(new_A1):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A1)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AUC_UUID]["n"] == 11
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["mean"], 2197883129)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["std"], 40391699)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["min"], 2145365902)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["max"], 2268446950)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_twitch_dict[105000][AUC_UUID], 2268446950)
    assert_percent_diff(per_twitch_dict[186000][AUC_UUID], 2203146703)
    assert_percent_diff(per_twitch_dict[266000][AUC_UUID], 2187484903)


def test_new_A2_auc(new_A2):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A2)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AUC_UUID]["n"] == 11
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["mean"], 1979655695)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["std"], 60891061)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["min"], 1880100989)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["max"], 2098455889)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_twitch_dict[104000][AUC_UUID], 1890483550)
    assert_percent_diff(per_twitch_dict[185000][AUC_UUID], 1995261562)
    assert_percent_diff(per_twitch_dict[262000][AUC_UUID], 2098455889)


def test_new_A3_auc(new_A3):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A3)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AUC_UUID]["n"] == 10
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["mean"], 1742767961)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["std"], 26476362)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["min"], 1700775183)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["max"], 1785602477)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_twitch_dict[108000][AUC_UUID], 1743723350)
    assert_percent_diff(per_twitch_dict[193000][AUC_UUID], 1719164790)
    assert_percent_diff(per_twitch_dict[266000][AUC_UUID], 1711830854)


def test_new_A4_auc(new_A4):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A4)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AUC_UUID]["n"] == 15
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["mean"], 2337802567)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["std"], 85977760)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["min"], 2204456864)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["max"], 2474957390)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_twitch_dict[81000][AUC_UUID], 2369406142)
    assert_percent_diff(per_twitch_dict[137000][AUC_UUID], 2474957390)
    assert_percent_diff(per_twitch_dict[196000][AUC_UUID], 2305482514)


def test_new_A5_auc(new_A5):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A5)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AUC_UUID]["n"] == 15
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["mean"], 975669720)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["std"], 39452029)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["min"], 916556595)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["max"], 1079880664)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_twitch_dict[80000][AUC_UUID], 962568392)
    assert_percent_diff(per_twitch_dict[138000][AUC_UUID], 978169492)
    assert_percent_diff(per_twitch_dict[197000][AUC_UUID], 989808351)


def test_new_A6_auc(new_A6):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(new_A6)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AUC_UUID]["n"] == 15
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["mean"], 225373714)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["std"], 24559045)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["min"], 180116348)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["max"], 265573223)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_twitch_dict[88000][AUC_UUID], 256551364)
    assert_percent_diff(per_twitch_dict[148000][AUC_UUID], 257671482)
    assert_percent_diff(per_twitch_dict[201000][AUC_UUID], 201413091)


def test_maiden_voyage_data_auc(maiden_voyage_data):
    per_twitch_dict, aggregate_metrics_dict = _get_data_metrics(maiden_voyage_data)

    # test data_metrics aggregate dictionary
    assert aggregate_metrics_dict[AUC_UUID]["n"] == 10
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["mean"], 9802421961)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["std"], 1211963395)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["min"], 8147044761)
    assert_percent_diff(aggregate_metrics_dict[AUC_UUID]["max"], 11738292588)

    # test data_metrics per beat dictionary
    assert_percent_diff(per_twitch_dict[123500][AUC_UUID], 10975158070)
    assert_percent_diff(per_twitch_dict[449500][AUC_UUID], 8678984942)
    assert_percent_diff(per_twitch_dict[856000][AUC_UUID], 8919661597)


def test_peak_detector_does_not_flip_data_by_default__because_default_kwarg_is_true():

    time, v = _load_file_tsv(os.path.join(PATH_TO_DATASETS, "new_A1_tsv.tsv"))

    # create numpy matrix
    raw_data = create_numpy_array_of_raw_gmr_from_python_arrays(time, v)
    peak_and_valley_indices = peak_detector(raw_data)
    peak_indices, valley_indices = peak_and_valley_indices

    expected_peak_indices = [70, 147, 220, 305, 397, 463, 555, 628, 713, 779, 871, 963]
    expected_valley_indices = [
        24,
        105,
        186,
        266,
        344,
        424,
        502,
        586,
        667,
        745,
        825,
        906,
        987,
    ]
    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)


def test_new_A1(new_A1):
    pipeline, peak_and_valley_indices = new_A1

    # plot and save results
    filtered_data = pipeline.get_noise_filtered_gmr()
    _plot_data(
        peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, "new_A1_peaks.png"),
    )

    peak_indices, valley_indices = peak_and_valley_indices

    expected_valley_indices = [
        70,
        147,
        220,
        305,
        397,
        463,
        555,
        628,
        713,
        779,
        871,
        963,
    ]
    expected_peak_indices = [
        24,
        105,
        186,
        266,
        344,
        424,
        502,
        586,
        667,
        745,
        825,
        906,
        987,
    ]
    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)


def test_new_A2(new_A2):
    pipeline, peak_and_valley_indices = new_A2

    # plot and save results
    filtered_data = pipeline.get_noise_filtered_gmr()
    _plot_data(
        peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, "new_A2_peaks.png"),
    )

    peak_indices, valley_indices = peak_and_valley_indices

    expected_valley_indices = [
        58,
        164,
        235,
        308,
        393,
        466,
        565,
        643,
        716,
        797,
        867,
        947,
    ]
    expected_peak_indices = [
        24,
        104,
        185,
        262,
        347,
        424,
        505,
        586,
        663,
        744,
        825,
        906,
        986,
    ]
    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)


def test_new_A3(new_A3):
    pipeline, peak_and_valley_indices = new_A3

    # plot and save results
    filtered_data = pipeline.get_noise_filtered_gmr()
    _plot_data(
        peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, "new_A3_peaks.png"),
    )

    peak_indices, valley_indices = peak_and_valley_indices

    expected_valley_indices = [
        62,
        165,
        239,
        324,
        404,
        470,
        562,
        628,
        727,
        805,
        871,
        963,
    ]
    expected_peak_indices = [28, 108, 193, 266, 351, 428, 509, 594, 667, 752, 832, 910]
    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)


def test_new_A4(new_A4):
    pipeline, peak_and_valley_indices = new_A4

    # plot and save results
    filtered_data = pipeline.get_noise_filtered_gmr()
    _plot_data(
        peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, "new_A4_peaks.png"),
    )

    peak_indices, valley_indices = peak_and_valley_indices

    expected_valley_indices = [
        58,
        117,
        176,
        235,
        290,
        349,
        407,
        466,
        518,
        584,
        639,
        697,
        756,
        815,
        867,
        926,
        985,
    ]
    expected_peak_indices = [
        23,
        81,
        137,
        196,
        255,
        311,
        369,
        427,
        486,
        545,
        601,
        659,
        715,
        774,
        832,
        890,
        946,
    ]
    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)


def test_new_A5(new_A5):
    pipeline, peak_and_valley_indices = new_A5

    # plot and save results
    filtered_data = pipeline.get_noise_filtered_gmr()
    _plot_data(
        peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, "new_A5_peaks.png"),
    )

    peak_indices, valley_indices = peak_and_valley_indices

    expected_valley_indices = [
        62,
        121,
        180,
        232,
        294,
        345,
        411,
        463,
        522,
        584,
        640,
        694,
        753,
        805,
        871,
        930,
        989,
    ]
    expected_peak_indices = [
        25,
        80,
        138,
        197,
        255,
        311,
        370,
        428,
        487,
        545,
        601,
        660,
        718,
        777,
        836,
        891,
        950,
    ]
    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)


def test_new_A6(new_A6):
    pipeline, peak_and_valley_indices = new_A6

    # plot and save results
    filtered_data = pipeline.get_noise_filtered_gmr()
    _plot_data(
        peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, "new_A6_peaks.png"),
    )

    peak_indices, valley_indices = peak_and_valley_indices

    expected_valley_indices = [
        51,
        117,
        176,
        238,
        294,
        341,
        400,
        459,
        525,
        580,
        625,
        698,
        757,
        808,
        874,
        926,
        981,
    ]
    expected_peak_indices = [
        31,
        88,
        148,
        201,
        255,
        318,
        376,
        428,
        490,
        552,
        604,
        663,
        722,
        784,
        832,
        898,
        953,
    ]
    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)


def test_maiden_voyage_data_peak_detection(maiden_voyage_data):
    pipeline, peak_and_valley_indices = maiden_voyage_data

    # plot and save results
    filtered_data = pipeline.get_noise_filtered_gmr()
    _plot_data(
        peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, "maiden_voyage_peaks.png"),
    )

    peak_indices, valley_indices = peak_and_valley_indices

    expected_valley_indices = [
        154,
        340,
        526,
        686,
        862,
        960,
        1185,
        1309,
        1439,
        1678,
        1769,
        # 1942,
    ]
    expected_peak_indices = [
        84,
        247,
        413,
        576,
        739,
        899,
        1059,
        1226,
        1383,
        1549,
        1712,
        1875,
    ]

    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)


def test_find_twitch_indices__raises_error_if_less_than_3_peaks_given():
    with pytest.raises(
        TooFewPeaksDetectedError,
        match=rf"A minimum of {MIN_NUMBER_PEAKS} peaks is required to extract twitch metrics, however only 2 peak\(s\) were detected",
    ):
        find_twitch_indices((np.array([1, 2]), None), None)


def test_find_twitch_indices__excludes_first_and_last_peak_when_no_outer_valleys(
    new_A1,
):
    pipeline, peak_and_valley_indices = new_A1
    filtered_data = pipeline.get_noise_filtered_gmr()

    actual_twitch_indices = find_twitch_indices(peak_and_valley_indices, filtered_data)
    actual_twitch_peak_indices = list(actual_twitch_indices.keys())
    expected_twitch_peak_indices = [
        105,
        186,
        266,
        344,
        424,
        502,
        586,
        667,
        745,
        825,
        906,
    ]
    assert actual_twitch_peak_indices == expected_twitch_peak_indices

    assert actual_twitch_indices[105] == {
        PRIOR_PEAK_INDEX_UUID: 24,
        PRIOR_VALLEY_INDEX_UUID: 70,
        SUBSEQUENT_PEAK_INDEX_UUID: 186,
        SUBSEQUENT_VALLEY_INDEX_UUID: 147,
    }


def test_find_twitch_indices__excludes_only_last_peak_when_no_outer_peak_at_beginning_and_no_outer_valley_at_end(
    new_A1,
):
    pipeline, peak_and_valley_indices = new_A1
    filtered_data = pipeline.get_noise_filtered_gmr()
    _, valley_indices = peak_and_valley_indices
    peak_indices = np.asarray(
        [105, 186, 266, 344, 424, 502, 586, 667, 745, 825, 906, 987], dtype=np.int32
    )
    actual_twitch_indices = find_twitch_indices(
        (peak_indices, valley_indices), filtered_data
    )
    actual_twitch_peak_indices = list(actual_twitch_indices.keys())
    expected_twitch_peak_indices = [
        105,
        186,
        266,
        344,
        424,
        502,
        586,
        667,
        745,
        825,
        906,
    ]
    assert actual_twitch_peak_indices == expected_twitch_peak_indices

    assert actual_twitch_indices[105] == {
        PRIOR_PEAK_INDEX_UUID: None,
        PRIOR_VALLEY_INDEX_UUID: 70,
        SUBSEQUENT_PEAK_INDEX_UUID: 186,
        SUBSEQUENT_VALLEY_INDEX_UUID: 147,
    }


@pytest.mark.parametrize(
    "test_data,expected_match,test_description",
    [
        (
            [24, 69, 105, 186, 266, 344, 424, 502, 586, 667, 745, 825, 906, 987],
            "24 and 69",
            "raises error when two peaks in a row at beginning",
        ),
        (
            [24, 105, 186, 266, 344, 424, 502, 586, 600, 667, 745, 825, 906, 987],
            "586 and 600",
            "raises error when two peaks in a row in middle",
        ),
        (
            [24, 105, 186, 266, 344, 424, 502, 586, 667, 745, 825, 906, 987, 1000],
            "987 and 1000",
            "raises error when two peaks in a row at end",
        ),
    ],
)
def test_find_twitch_indices__raises_error_if_two_peaks_in_a_row__and_start_with_peak(
    new_A1, test_data, expected_match, test_description
):
    pipeline, peak_and_valley_indices = new_A1
    filtered_data = pipeline.get_noise_filtered_gmr()
    _, valley_indices = peak_and_valley_indices
    peak_indices = np.asarray(test_data, dtype=np.int32)
    with pytest.raises(TwoPeaksInARowError, match=expected_match):
        find_twitch_indices((peak_indices, valley_indices), filtered_data)


@pytest.mark.parametrize(
    "test_data,expected_match,test_description",
    [
        (
            [71, 105, 186, 266, 344, 424, 502, 586, 667, 745, 825, 906, 987],
            "71 and 105",
            "raises error when two peaks in a row at beginning",
        ),
        (
            [105, 186, 266, 344, 424, 502, 586, 600, 667, 745, 825, 906, 987],
            "586 and 600",
            "raises error when two peaks in a row in middle",
        ),
        (
            [105, 186, 266, 344, 424, 502, 586, 667, 745, 825, 906, 987, 1000],
            "987 and 1000",
            "raises error when two peaks in a row at end",
        ),
    ],
)
def test_find_twitch_indices__raises_error_if_two_peaks_in_a_row__and_does_not_start_with_peak(
    new_A1, test_data, expected_match, test_description
):
    pipeline, peak_and_valley_indices = new_A1
    filtered_data = pipeline.get_noise_filtered_gmr()
    _, valley_indices = peak_and_valley_indices
    peak_indices = np.asarray(test_data, dtype=np.int32)
    with pytest.raises(TwoPeaksInARowError, match=expected_match):
        find_twitch_indices((peak_indices, valley_indices), filtered_data)


@pytest.mark.parametrize(
    "test_data,expected_match,test_description",
    [
        (
            [70, 100, 147, 220, 305, 397, 463, 555, 628, 713, 779, 871, 963],
            "70 and 100",
            "raises error when two valleys in a row at beginning",
        ),
        (
            [70, 147, 220, 305, 397, 400, 463, 555, 628, 713, 779, 871, 963],
            "397 and 400",
            "raises error when two valleys in a row in middle",
        ),
        (
            [70, 147, 220, 305, 397, 463, 555, 628, 713, 779, 871, 963, 1000, 1001],
            "1000 and 1001",
            "raises error when two valleys in a row at end",
        ),
    ],
)
def test_find_twitch_indices__raises_error_if_two_valleys_in_a_row__and_starts_with_peak(
    new_A1, test_data, expected_match, test_description
):
    pipeline, peak_and_valley_indices = new_A1
    filtered_data = pipeline.get_noise_filtered_gmr()
    peak_indices, _ = peak_and_valley_indices
    valley_indices = np.asarray(test_data, dtype=np.int32)
    with pytest.raises(TwoValleysInARowError, match=expected_match):
        find_twitch_indices((peak_indices, valley_indices), filtered_data)


@pytest.mark.parametrize(
    "test_data,expected_match,test_description",
    [
        (
            [0, 70, 100, 147, 220, 305, 397, 463, 555, 628, 713, 779, 871, 963],
            "70 and 100",
            "raises error when two valleys in a row at beginning",
        ),
        (
            [0, 70, 147, 220, 305, 397, 400, 463, 555, 628, 713, 779, 871, 963],
            "397 and 400",
            "raises error when two valleys in a row in middle",
        ),
        (
            [0, 70, 147, 220, 305, 397, 463, 555, 628, 713, 779, 871, 963, 1000, 1001],
            "1000 and 1001",
            "raises error when two valleys in a row at end",
        ),
    ],
)
def test_find_twitch_indices__raises_error_if_two_valleys_in_a_row__and_does_not_start_with_peak(
    new_A1, test_data, expected_match, test_description
):
    pipeline, peak_and_valley_indices = new_A1
    filtered_data = pipeline.get_noise_filtered_gmr()
    peak_indices, _ = peak_and_valley_indices
    valley_indices = np.asarray(test_data, dtype=np.int32)
    with pytest.raises(TwoValleysInARowError, match=expected_match):
        find_twitch_indices((peak_indices, valley_indices), filtered_data)


def test_find_twitch_indices__returns_correct_values_with_data_that_ends_in_peak():
    peak_indices = np.array(range(0, 10, 2), dtype=np.int32)
    valley_indices = np.array(range(1, 9, 2), dtype=np.int32)
    find_twitch_indices((peak_indices, valley_indices), None)


def test_noisy_data_A1(noisy_data_A1):
    pipeline, peak_and_valley_indices = noisy_data_A1

    # plot and save results
    filtered_data = pipeline.get_noise_filtered_gmr()
    _plot_data(
        peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, "new_noisy_data_A1_peaks.png"),
    )

    peak_indices, valley_indices = peak_and_valley_indices

    expected_peak_indices = [
        19,
        580,
        1165,
        1728,
        2341,
        2867,
        3393,
        3956,
        4530,
        5088,
        5710,
        6228,
        6797,
        7340,
        7964,
        8525,
        9102,
        9623,
        10184,
        10763,
        11358,
        11909,
        12521,
        13045,
    ]
    expected_valley_indices = [
        330,
        803,
        1573,
        2126,
        2712,
        3111,
        3681,
        4206,
        4968,
        5504,
        6094,
        6617,
        7067,
        7550,
        8379,
        8895,
        9427,
        9992,
        10501,
        10989,
        11730,
        12293,
        12820,
    ]
    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)


def test_noisy_data_B1(noisy_data_B1):
    pipeline, peak_and_valley_indices = noisy_data_B1
    # plot and save results
    filtered_data = pipeline.get_noise_filtered_gmr()
    _plot_data(
        peak_and_valley_indices,
        filtered_data,
        os.path.join(PATH_TO_PNGS, "new_noisy_data_B1_peaks.png"),
    )

    peak_indices, valley_indices = peak_and_valley_indices

    expected_peak_indices = [
        341,
        867,
        1392,
        1936,
        2451,
        3006,
        3507,
        4036,
        4584,
        5079,
        5625,
        6138,
        6676,
        7219,
        7751,
        8279,
        8778,
        9333,
        9847,
        10389,
        10908,
        11454,
        11981,
        12503,
        13034,
    ]
    expected_valley_indices = [
        701,
        1145,
        1721,
        2271,
        2829,
        3178,
        3816,
        4372,
        4815,
        5340,
        5897,
        6450,
        7040,
        7570,
        8091,
        8616,
        9155,
        9602,
        10134,
        10686,
        11233,
        11798,
        12347,
        12851,
    ]
    assert np.array_equal(peak_indices, expected_peak_indices)
    assert np.array_equal(valley_indices, expected_valley_indices)
