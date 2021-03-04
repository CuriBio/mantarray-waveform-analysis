# -*- coding: utf-8 -*-
import copy
from typing import Any
import uuid

from mantarray_waveform_analysis import apply_empty_plate_calibration
from mantarray_waveform_analysis import apply_noise_filtering
from mantarray_waveform_analysis import apply_sensitivity_calibration
from mantarray_waveform_analysis import BESSEL_BANDPASS_UUID
from mantarray_waveform_analysis import BESSEL_LOWPASS_10_UUID
from mantarray_waveform_analysis import BESSEL_LOWPASS_30_UUID
from mantarray_waveform_analysis import BUTTERWORTH_LOWPASS_30_UUID
from mantarray_waveform_analysis import calculate_displacement_from_voltage
from mantarray_waveform_analysis import calculate_voltage_from_gmr
from mantarray_waveform_analysis import create_filter
from mantarray_waveform_analysis import FILTER_CHARACTERISTICS
from mantarray_waveform_analysis import FilterCreationNotImplementedError
from mantarray_waveform_analysis import MIDSCALE_CODE
from mantarray_waveform_analysis import noise_cancellation
from mantarray_waveform_analysis import RAW_TO_SIGNED_CONVERSION_VALUE
from mantarray_waveform_analysis import UnrecognizedFilterUuidError
from nptyping import Int
from nptyping import NDArray
import numpy as np
import pytest

from .fixtures_transforms import fixture_bessel_lowpass_10_for_100hz
from .fixtures_utils import fixture_raw_generic_well_a1

__fixtures__ = [fixture_raw_generic_well_a1, fixture_bessel_lowpass_10_for_100hz]


def test_apply_sensitivity_calibration(raw_generic_well_a1):

    calibrated_gmr = apply_sensitivity_calibration(raw_generic_well_a1)
    assert isinstance(calibrated_gmr, NDArray[(2, Any), Int[32]])

    # more assertions will be added when there is actual calibration to apply


def test_noise_cancellation(raw_generic_well_a1):
    noise_cancelled_gmr = noise_cancellation(raw_generic_well_a1, raw_generic_well_a1)
    assert isinstance(noise_cancelled_gmr, NDArray[(2, Any), Int[32]])

    # more assertions will be added when there is actual cancellation methods that are taking place


def test_apply_empty_plate_calibration(raw_generic_well_a1):
    fully_calibrated_gmr = apply_empty_plate_calibration(raw_generic_well_a1)
    assert isinstance(fully_calibrated_gmr, NDArray[(2, Any), Int[32]])
    # more assertions will be added when empty plate calibration is better understood


def test_create_filter__raises_error_for_unrecognized_uuid():
    with pytest.raises(
        UnrecognizedFilterUuidError, match="0ecf0e52-0a29-453f-a6ff-46f5ec3ae782"
    ):
        create_filter(uuid.UUID("0ecf0e52-0a29-453f-a6ff-46f5ec3ae782"), 44)


def test_create_filter__raises_error_for_code_missing_to_generate_filter(mocker):
    expected_uuid = uuid.UUID("0ecf0e52-0a29-453f-a6ff-46f5ec3ae782")
    mocker.patch.dict(FILTER_CHARACTERISTICS)
    FILTER_CHARACTERISTICS[expected_uuid] = {"filter_type": "fake_filter"}
    with pytest.raises(FilterCreationNotImplementedError, match=str(expected_uuid)):
        create_filter(expected_uuid, 48)


@pytest.mark.parametrize(
    ",".join(
        [
            "filter_uuid",
            "sampling_period_centimilliseconds",
            "expected_sos",
            "test_description",
        ]
    ),
    [
        (
            BESSEL_BANDPASS_UUID,
            1000,
            [
                [0.004146, 0.008293, 0.004146, 1.0, -1.091367, 0.308968],
                [1.0, 2.0, 1.0, 1.0, -1.150406, 0.454009],
                [1.0, -2.0, 1.0, 1.0, -1.987071, 0.987117],
                [1.0, -2.0, 1.0, 1.0, -1.992648, 0.992684],
            ],
            "bessel bandpass at 100 Hz / 1000 cms sampling, 0.1 Hz and 10 Hz cutoffs",
        ),
        (
            BESSEL_BANDPASS_UUID,
            250,
            [
                [
                    2.89771627e-05,
                    5.79543254e-05,
                    2.89771627e-05,
                    1,
                    -1.73792987e00,
                    7.56492551e-01,
                ],
                [
                    1,
                    2,
                    1,
                    1,
                    -1.79217579e00,
                    8.16847162e-01,
                ],
                [
                    1,
                    -2,
                    1,
                    1,
                    -1.99675894e00,
                    9.96761815e-01,
                ],
                [
                    1,
                    -2,
                    1,
                    1,
                    -1.99816395e00,
                    9.98166184e-01,
                ],
            ],
            "bessel bandpass at 400 Hz / 250 cms sampling, 0.1 Hz and 10 Hz cutoffs",
        ),
        (
            BESSEL_LOWPASS_10_UUID,
            1000,
            [
                [0.004287, 0.008575, 0.004287, 1.0, -1.077012, 0.300943],
                [1.0, 2.0, 1.0, 1.0, -1.140961, 0.4473],
            ],
            "bessel lowpass at 100 Hz / 1000 cms sampling, 10 Hz cutoff",
        ),
        (
            BESSEL_LOWPASS_30_UUID,
            160,
            [
                [3.357338e-04, 6.714677e-04, 3.357338e-04, 1.0, -1.511952, 0.5755379],
                [1.0, 2.0, 1.0, 1.0, -1.589599, 0.6740787],
            ],
            "bessel lowpass at 625 Hz / 160 cms sampling, 30 Hz cutoff",
        ),
        (
            BUTTERWORTH_LOWPASS_30_UUID,
            160,
            [
                [3.588408e-04, 7.176816e-04, 3.588408e-04, 1.0, -1.498496, 0.5693282],
                [1.0, 2.0, 1.0, 1.0, -1.7148030, 0.7958595],
            ],
            "butterworth lowpass at 625 Hz / 160 cms sampling, 30 Hz cutoff",
        ),
    ],
)
def test_create_filter__bessel_bandpass__returns_correct_coefficients(
    filter_uuid,
    sampling_period_centimilliseconds,
    expected_sos,
    test_description,
):
    expected_sos = np.array(expected_sos, dtype=float)

    actual_sos = create_filter(filter_uuid, sampling_period_centimilliseconds)
    np.testing.assert_array_almost_equal(actual_sos, expected_sos)


def test_apply_noise_filtering__bessel_bandpass(
    raw_generic_well_a1, bessel_lowpass_10_for_100hz
):
    # confirm pre-condition
    assert raw_generic_well_a1[1, 5] == -124293

    filtered_gmr = apply_noise_filtering(
        raw_generic_well_a1, bessel_lowpass_10_for_100hz
    )
    assert isinstance(filtered_gmr, NDArray[(2, Any), Int[32]])

    # confirm change in GMR after filtering
    assert filtered_gmr[1, 5] == -129210

    # confirm time hasn't change
    assert filtered_gmr[0, 5] == raw_generic_well_a1[0, 5]


def test_calculate_voltage_from_gmr__returns_correct_values():
    test_data = np.array(
        [-0x80000, MIDSCALE_CODE - RAW_TO_SIGNED_CONVERSION_VALUE, 0x7FFFFF]
    )
    test_data = np.vstack((np.zeros(3), test_data))
    original_test_data = copy.deepcopy(test_data)
    reference_voltage = 2.5
    adc_gain = 2

    actual_converted_data = calculate_voltage_from_gmr(
        test_data, reference_voltage=reference_voltage, adc_gain=adc_gain
    )

    # confirm original data was not modified
    np.testing.assert_array_equal(test_data, original_test_data)

    assert isinstance(actual_converted_data, NDArray[(2, Any), np.float32])

    expected_first_val = (
        test_data[1, 0].astype(np.float32)
        * 1000
        * reference_voltage
        * RAW_TO_SIGNED_CONVERSION_VALUE
        / adc_gain
    )
    expected_last_val = (
        test_data[1, 2].astype(np.float32)
        * 1000
        * reference_voltage
        * RAW_TO_SIGNED_CONVERSION_VALUE
        / adc_gain
    )

    expected_first_val = expected_first_val.astype(np.float32)
    expected_last_val = expected_last_val.astype(np.float32)

    expected_data = [expected_first_val, 0, expected_last_val]

    np.testing.assert_almost_equal(
        actual_converted_data[1, :], expected_data, decimal=0
    )


def test_calculate_displacement_from_voltage():
    test_data = np.array([-1, 0, 1])
    test_data = np.vstack((np.zeros(3), test_data))
    original_test_data = copy.deepcopy(test_data)

    actual_converted_data = calculate_displacement_from_voltage(test_data)

    # confirm original data was not modified
    np.testing.assert_array_equal(test_data, original_test_data)

    assert isinstance(actual_converted_data, NDArray[(2, Any), np.float32])

    expected_data = [1.1, 0.1, -0.9]

    np.testing.assert_almost_equal(
        actual_converted_data[1, :], expected_data, decimal=6
    )
