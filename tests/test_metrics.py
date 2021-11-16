# -*- coding: utf-8 -*-
"""Tests for skeletal and cardiac muscle metrics."""

from mantarray_waveform_analysis import CENTIMILLISECONDS_PER_SECOND
from mantarray_waveform_analysis import metrics
from mantarray_waveform_analysis import PRIOR_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import SUBSEQUENT_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import TIME_VALUE_UUID
from mantarray_waveform_analysis import WIDTH_VALUE_UUID
import numpy as np
import pytest

from .fixtures_metrics import fixture_generate_twitch_amplitude
from .fixtures_metrics import fixture_generate_twitch_auc
from .fixtures_metrics import fixture_generate_twitch_baseline_to_peak
from .fixtures_metrics import fixture_generate_twitch_fraction_amplitude
from .fixtures_metrics import fixture_generate_twitch_frequency
from .fixtures_metrics import fixture_generate_twitch_irregularity
from .fixtures_metrics import fixture_generate_twitch_peak_time_contraction
from .fixtures_metrics import fixture_generate_twitch_peak_time_relaxation
from .fixtures_metrics import fixture_generate_twitch_peak_to_baseline
from .fixtures_metrics import fixture_generate_twitch_period
from .fixtures_metrics import fixture_generate_twitch_velocity_contraction
from .fixtures_metrics import fixture_generate_twitch_velocity_relaxation
from .fixtures_metrics import fixture_generate_twitch_width
from .fixtures_metrics import fixture_generic_well_features
from .fixtures_utils import fixture_raw_generic_well_a1
from .fixtures_utils import fixture_raw_generic_well_a2
from .fixtures_utils import fixture_sample_tissue_reading
from .fixtures_utils import fixture_sample_reference_reading

__fixtures__ = [
    fixture_raw_generic_well_a1,
    fixture_raw_generic_well_a2,
    fixture_generic_well_features,
    fixture_generate_twitch_amplitude,
    fixture_generate_twitch_auc,
    fixture_generate_twitch_fraction_amplitude,
    fixture_generate_twitch_frequency,
    fixture_generate_twitch_irregularity,
    fixture_generate_twitch_baseline_to_peak,
    fixture_generate_twitch_peak_to_baseline,
    fixture_generate_twitch_peak_time_contraction,
    fixture_generate_twitch_peak_time_relaxation,
    fixture_generate_twitch_period,
    fixture_generate_twitch_velocity_contraction,
    fixture_generate_twitch_velocity_relaxation,
    fixture_generate_twitch_width,
    fixture_sample_tissue_reading,
    fixture_sample_reference_reading,
]


def test_metrics__peaks_greater_than_prior_and_subsequent_valleys(generic_well_features):

    filtered_data, _, twitch_indices = generic_well_features

    for twitch, pv in twitch_indices.items():
        assert filtered_data[1, twitch] > filtered_data[1, pv[PRIOR_VALLEY_INDEX_UUID]]
        assert filtered_data[1, twitch] > filtered_data[1, pv[SUBSEQUENT_VALLEY_INDEX_UUID]]


def test_metrics__TwitchAmplitude(generic_well_features, generate_twitch_amplitude):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    metric = metrics.TwitchAmplitude(rounded=False)
    estimate = metric.fit(**PARAMS)

    assert np.all(estimate >= 0)
    # regression
    assert np.allclose(estimate, generate_twitch_amplitude)


def test_metrics__TwitchAUC(generic_well_features, generate_twitch_auc):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    metric = metrics.TwitchAUC(rounded=False)
    estimate = metric.fit(**PARAMS)

    assert np.all(estimate > 0)
    # regression
    assert np.allclose(estimate, generate_twitch_auc)


def test_metrics__TwitchFractionAmplitude(generic_well_features, generate_twitch_fraction_amplitude):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    metric = metrics.TwitchFractionAmplitude(rounded=False)
    estimate = metric.fit(**PARAMS)

    assert np.max(estimate) == 1
    assert np.min(estimate) >= 0
    # regression
    assert np.allclose(estimate, generate_twitch_fraction_amplitude)


def test_metrics__TwitchFrequency(generic_well_features, generate_twitch_frequency):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    frequency_metric = metrics.TwitchFrequency(rounded=False)
    frequencies = frequency_metric.fit(**PARAMS)

    period_metric = metrics.TwitchPeriod(rounded=False)
    periods = period_metric.fit(**PARAMS)

    assert np.min(frequencies) >= 0
    assert np.all(frequencies == 1 / (periods.astype(float) / CENTIMILLISECONDS_PER_SECOND))
    # regression
    assert np.allclose(frequencies, generate_twitch_frequency)


def test_metrics__TwitchIrregularity(generic_well_features, generate_twitch_irregularity):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    metric = metrics.TwitchIrregularity(rounded=False)
    estimate = metric.fit(**PARAMS)

    # check that only first and last indices are NAN
    assert np.isnan(estimate[0])
    assert np.isnan(estimate[-1])
    assert np.isnan(estimate).sum() == 2
    # regression
    assert np.allclose(estimate[1:-1], generate_twitch_irregularity[1:-1])


def test_metrics__TwitchPeakTime(
    generic_well_features, generate_twitch_peak_time_contraction, generate_twitch_peak_time_relaxation
):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    percents = range(10, 95, 5)

    contraction_metric = metrics.TwitchPeakTime(
        rounded=False, is_contraction=True, twitch_width_percents=percents
    )
    contractions = contraction_metric.fit(**PARAMS)

    relaxation_metric = metrics.TwitchPeakTime(
        rounded=False, is_contraction=False, twitch_width_percents=percents
    )
    relaxations = relaxation_metric.fit(**PARAMS)

    for i in range(len(percents) - 1):
        assert (
            contractions[0][percents[i]][TIME_VALUE_UUID] > contractions[0][percents[i + 1]][TIME_VALUE_UUID]
        )

    for i in range(len(percents) - 1):
        assert relaxations[0][percents[i]][TIME_VALUE_UUID] < relaxations[0][percents[i + 1]][TIME_VALUE_UUID]

    # regression
    assert np.all(contractions == generate_twitch_peak_time_contraction)
    assert np.all(relaxations == generate_twitch_peak_time_relaxation)


def test_metrics__TwitchPeakToBaseline_is_contraction(
    generic_well_features, generate_twitch_baseline_to_peak
):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    metric = metrics.TwitchPeakToBaseline(rounded=False, is_contraction=True)
    estimate = metric.fit(**PARAMS)

    assert np.all(estimate > 0)
    # regression
    assert np.all(estimate == generate_twitch_baseline_to_peak)


def test_metrics__TwitchPeakToBaseline_is_relaxation(generic_well_features, generate_twitch_peak_to_baseline):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    metric = metrics.TwitchPeakToBaseline(rounded=False, is_contraction=False)
    estimate = metric.fit(**PARAMS)

    assert np.all(estimate > 0)
    # regression
    assert np.all(estimate == generate_twitch_peak_to_baseline)


def test_metrics__TwitchPeriod(generic_well_features, generate_twitch_period):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    metric = metrics.TwitchPeriod(rounded=False)
    estimate = metric.fit(**PARAMS)

    assert np.all(estimate > 0)
    # regression
    assert np.all(estimate == generate_twitch_period)


def test_metrics__TwitchVelocity(
    generic_well_features, generate_twitch_velocity_contraction, generate_twitch_velocity_relaxation
):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    contraction_metric = metrics.TwitchVelocity(
        rounded=False, is_contraction=True, twitch_width_percents=range(10, 95, 5)
    )
    contractions = contraction_metric.fit(**PARAMS)

    relaxation_metric = metrics.TwitchVelocity(
        rounded=False, is_contraction=False, twitch_width_percents=range(10, 95, 5)
    )
    relaxations = relaxation_metric.fit(**PARAMS)

    assert np.all(contractions > 0)
    assert np.all(relaxations > 0)

    # regression
    assert np.all(contractions == generate_twitch_velocity_contraction)
    assert np.all(relaxations == generate_twitch_velocity_relaxation)


def test_metrics__TwitchWidth(generic_well_features, generate_twitch_width):

    [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
    PARAMS = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    metric = metrics.TwitchWidth(rounded=False, twitch_width_percents=(50, 90))
    estimate = metric.fit(**PARAMS)

    assert estimate[0][90][WIDTH_VALUE_UUID] > estimate[0][50][WIDTH_VALUE_UUID]
    # regression test
    assert estimate == generate_twitch_width


def test_metrics__x_interpolation():

    with pytest.raises(ZeroDivisionError):
        metrics.interpolate_x_for_y_between_two_points(1, 0, 10, 0, 5)


def test_metrics__y_interpolation():

    with pytest.raises(ZeroDivisionError):
        metrics.interpolate_y_for_x_between_two_points(1, 0, 10, 0, 5)


def test_metrics__create_statistics():

    estimates = np.asarray([1, 2, 3, 4, 5])

    statistics = metrics.BaseMetric.create_statistics_dict(estimates, rounded=False)
    assert statistics["mean"] == np.nanmean(estimates)
    assert statistics["std"] == np.nanstd(estimates)
    assert statistics["min"] == np.nanmin(estimates)
    assert statistics["max"] == np.nanmax(estimates)

    statistics = metrics.BaseMetric.create_statistics_dict(estimates, rounded=True)
    assert statistics["mean"] == int(round(np.nanmean(estimates)))
    assert statistics["std"] == int(round(np.nanstd(estimates)))
    assert statistics["min"] == int(round(np.nanmin(estimates)))
    assert statistics["max"] == int(round(np.nanmax(estimates)))

    estimates = []
    statistics = metrics.BaseMetric.create_statistics_dict(estimates, rounded=False)
    assert statistics["mean"] is None
    assert statistics["std"] is None
    assert statistics["min"] is None
    assert statistics["max"] is None
