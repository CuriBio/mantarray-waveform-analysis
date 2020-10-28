# -*- coding: utf-8 -*-
import os

from mantarray_waveform_analysis import BUTTERWORTH_LOWPASS_30_UUID
from mantarray_waveform_analysis import CENTIMILLISECONDS_PER_SECOND
import pytest

from .fixtures_utils import _run_peak_detection


@pytest.fixture(scope="session", name="maiden_voyage_data")
def fixture_maiden_voyage_data():
    return _run_peak_detection(
        "maiden_voyage_data.tsv", sampling_rate_construct=200, flip_data=True
    )


@pytest.fixture(scope="session", name="noisy_data_A1")
def fixture_noisy_data_A1():
    return _run_peak_detection(
        os.path.join("noisy_data", "MA20001025__2020_09_21_163232__A1.h5"),
        sampling_rate_construct=600,
        flip_data=True,
    )


@pytest.fixture(scope="session", name="noisy_data_B1")
def fixture_noisy_data_B1():
    return _run_peak_detection(
        os.path.join("noisy_data", "MA20001025__2020_09_21_163232__B1.h5"),
        sampling_rate_construct=600,
        flip_data=True,
        time_scaling_factor=CENTIMILLISECONDS_PER_SECOND,
    )


@pytest.fixture(scope="session", name="MA20123123__2020_10_13_173812__B6")
def fixture_MA20123123__2020_10_13_173812__B6():
    return _run_peak_detection(
        os.path.join("two_valley_error", "MA20123123__2020_10_13_173812__B6.h5"),
        sampling_rate_construct=625,
        flip_data=True,
        time_scaling_factor=CENTIMILLISECONDS_PER_SECOND,
        noise_filter_uuid=BUTTERWORTH_LOWPASS_30_UUID,
    )
