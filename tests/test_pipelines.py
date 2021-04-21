# -*- coding: utf-8 -*-
from unittest.mock import ANY

from mantarray_waveform_analysis import BESSEL_LOWPASS_10_UUID
from mantarray_waveform_analysis import DataAlreadyLoadedInPipelineError
from mantarray_waveform_analysis import Pipeline
from mantarray_waveform_analysis import pipelines
from mantarray_waveform_analysis import PipelineTemplate
import numpy as np
import pytest
from scipy import signal

from .fixtures_pipelines import fixture_generic_pipeline
from .fixtures_pipelines import fixture_generic_pipeline_template
from .fixtures_pipelines import fixture_loaded_generic_pipeline
from .fixtures_utils import fixture_raw_generic_well_a1
from .fixtures_utils import fixture_raw_generic_well_a2

__fixtures__ = [
    fixture_raw_generic_well_a1,
    fixture_raw_generic_well_a2,
    fixture_generic_pipeline,
    fixture_loaded_generic_pipeline,
    fixture_generic_pipeline_template,
]


def test_PipelineTemeplate__create_pipeline__creates_pipeline_linked_to_template():
    template = PipelineTemplate(
        noise_filter_uuid=BESSEL_LOWPASS_10_UUID, tissue_sampling_period=960
    )
    pipeline = template.create_pipeline()
    assert isinstance(pipeline, Pipeline)

    assert pipeline.get_template() is template


def test_PipelineTemplate__get_filter_coefficients__calls_create_filter_with_expected_signature(
    mocker, generic_pipeline_template
):
    expected_return = "0832"
    mocked_create_filter = mocker.patch.object(
        pipelines, "create_filter", autospec=True, return_value=expected_return
    )
    actual_coefficients = generic_pipeline_template.get_filter_coefficients()
    mocked_create_filter.assert_called_once_with(
        generic_pipeline_template.noise_filter_uuid,
        generic_pipeline_template.tissue_sampling_period,
    )
    assert actual_coefficients == expected_return


def test_Pipeline__load_raw_gmr_data__sets_data(
    loaded_generic_pipeline, raw_generic_well_a1, raw_generic_well_a2
):
    np.testing.assert_array_equal(
        loaded_generic_pipeline.get_raw_tissue_magnetic_data(), raw_generic_well_a1
    )
    np.testing.assert_array_equal(
        loaded_generic_pipeline.get_raw_reference_magnetic_data(), raw_generic_well_a2
    )


def test_Pipeline__load_raw_gmr_data__raises_error_if_data_already_loaded(
    loaded_generic_pipeline, raw_generic_well_a1, raw_generic_well_a2
):
    with pytest.raises(DataAlreadyLoadedInPipelineError):
        loaded_generic_pipeline.load_raw_magnetic_data(
            raw_generic_well_a1, raw_generic_well_a2
        )


def test_Pipeline__get_sensitivity_calibrated_tissue_gmr__calls_correct_methods_to_calibrate__but_does_not_call_again_repeatedly(
    mocker, loaded_generic_pipeline, raw_generic_well_a1
):
    expected_return = "blah"
    mocked_apply_sensitivity_calibration = mocker.patch.object(
        pipelines,
        "apply_sensitivity_calibration",
        autospec=True,
        return_value=expected_return,
    )
    calibrated_tissue = loaded_generic_pipeline.get_sensitivity_calibrated_tissue_gmr()
    # Eli (7/6/20): NumPy arrays don't play well with assert_called_once_with, so asserting things separately
    assert mocked_apply_sensitivity_calibration.call_count == 1
    actual_array_arg = mocked_apply_sensitivity_calibration.call_args_list[0][0][0]
    np.testing.assert_array_equal(actual_array_arg, raw_generic_well_a1)
    assert calibrated_tissue == expected_return

    actual_return_2 = loaded_generic_pipeline.get_sensitivity_calibrated_tissue_gmr()
    assert mocked_apply_sensitivity_calibration.call_count == 1
    assert actual_return_2 == expected_return


def test_Pipeline__get_sensitivity_calibrated_reference_gmr__calls_correct_methods_to_calibrate__but_does_not_call_again_repeatedly(
    mocker, loaded_generic_pipeline, raw_generic_well_a2
):
    expected_return = "wow"
    mocked_apply_sensitivity_calibration = mocker.patch.object(
        pipelines,
        "apply_sensitivity_calibration",
        autospec=True,
        return_value=expected_return,
    )
    calibrated_reference = (
        loaded_generic_pipeline.get_sensitivity_calibrated_reference_gmr()
    )
    # Eli (7/6/20): NumPy arrays don't play well with assert_called_once_with, so asserting things separately
    assert mocked_apply_sensitivity_calibration.call_count == 1
    actual_array_arg = mocked_apply_sensitivity_calibration.call_args_list[0][0][0]
    np.testing.assert_array_equal(actual_array_arg, raw_generic_well_a2)
    assert calibrated_reference == expected_return

    actual_return_2 = loaded_generic_pipeline.get_sensitivity_calibrated_reference_gmr()
    assert mocked_apply_sensitivity_calibration.call_count == 1
    assert actual_return_2 == expected_return


@pytest.mark.parametrize(
    ",".join(
        [
            "expected_return",
            "function_name_to_mock",
            "lambda_of_method_under_test",
            "list_of_lambdas_to_get_call_args",
            "test_description2",
        ]
    ),
    [
        (
            "54321",
            "noise_cancellation",
            lambda the_pipeline: the_pipeline.get_noise_cancelled_gmr(),
            [
                lambda the_pipeline: the_pipeline.get_sensitivity_calibrated_tissue_gmr(),
                lambda the_pipeline: the_pipeline.get_sensitivity_calibrated_reference_gmr(),
            ],
            "applying noise cancellation calibration",
        ),
        (
            "12345",
            "apply_empty_plate_calibration",
            lambda the_pipeline: the_pipeline.get_fully_calibrated_gmr(),
            [lambda the_pipeline: the_pipeline.get_noise_cancelled_gmr()],
            "applying empty plate calibration",
        ),
        (
            "abcde",
            "apply_noise_filtering",
            lambda the_pipeline: the_pipeline.get_noise_filtered_magnetic_data(),
            [lambda the_pipeline: the_pipeline.get_fully_calibrated_gmr()],
            "applying noise filtering",
        ),
        (
            "qed",
            "compress_filtered_gmr",
            lambda the_pipeline: the_pipeline.get_compressed_magnetic_data(),
            [lambda the_pipeline: the_pipeline.get_noise_filtered_magnetic_data()],
            "applying compression",
        ),
        (
            "wakka",
            "calculate_voltage_from_gmr",
            lambda the_pipeline: the_pipeline.get_compressed_voltage(),
            [lambda the_pipeline: the_pipeline.get_compressed_magnetic_data()],
            "converting compressed GMR data to voltage",
        ),
        (
            "banjo",
            "calculate_displacement_from_voltage",
            lambda the_pipeline: the_pipeline.get_compressed_displacement(),
            [lambda the_pipeline: the_pipeline.get_compressed_voltage()],
            "converting compressed voltage data to displacement",
        ),
        (
            "hello",
            "calculate_force_from_displacement",
            lambda the_pipeline: the_pipeline.get_compressed_force(),
            [lambda the_pipeline: the_pipeline.get_compressed_displacement()],
            "converting compressed displacement data to force",
        ),
        (
            "wakka",
            "calculate_voltage_from_gmr",
            lambda the_pipeline: the_pipeline.get_voltage(),
            [lambda the_pipeline: the_pipeline.get_noise_filtered_magnetic_data()],
            "converting GMR data to voltage",
        ),
        (
            "banjo",
            "calculate_displacement_from_voltage",
            lambda the_pipeline: the_pipeline.get_displacement(),
            [lambda the_pipeline: the_pipeline.get_voltage()],
            "converting voltage data to displacement",
        ),
        (
            "hello",
            "calculate_force_from_displacement",
            lambda the_pipeline: the_pipeline.get_force(),
            [lambda the_pipeline: the_pipeline.get_displacement()],
            "converting displacement data to force",
        ),
        (
            "trampoline",
            "peak_detector",
            lambda the_pipeline: the_pipeline.get_peak_detection_results(),
            [lambda the_pipeline: the_pipeline.get_noise_filtered_magnetic_data()],
            "detecting the peaks in the magnetic traces",
        ),
        (
            "airplane",
            "data_metrics",
            lambda the_pipeline: the_pipeline.get_magnetic_data_metrics(),
            [
                lambda the_pipeline: the_pipeline.get_peak_detection_results(),
                lambda the_pipeline: the_pipeline.get_noise_filtered_magnetic_data(),
            ],
            "calculate data metrics on the traces from the raw magentic readings",
        ),
        (
            "dog",
            "data_metrics",
            lambda the_pipeline: the_pipeline.get_displacement_data_metrics(),
            [
                lambda the_pipeline: the_pipeline.get_peak_detection_results(),
                lambda the_pipeline: the_pipeline.get_displacement(),
            ],
            "calculate data metrics on the traces from displacement data",
        ),
        (
            "cat",
            "data_metrics",
            lambda the_pipeline: the_pipeline.get_force_data_metrics(),
            [
                lambda the_pipeline: the_pipeline.get_peak_detection_results(),
                lambda the_pipeline: the_pipeline.get_force(),
            ],
            "calculate data metrics on the traces from force data",
        ),
    ],
)
def test_Pipeline__get_data_type__calls_correct_methods_to_perform_action__but_does_not_call_again_repeatedly(
    expected_return,
    function_name_to_mock,
    lambda_of_method_under_test,
    list_of_lambdas_to_get_call_args,
    test_description2,
    mocker,
    loaded_generic_pipeline,
):

    mocked_function_under_test = mocker.patch.object(
        pipelines,
        function_name_to_mock,
        autospec=True,
        return_value=expected_return,
    )
    actual_return_1 = lambda_of_method_under_test(loaded_generic_pipeline)
    # Eli (7/6/20): NumPy arrays don't play well with assert_called_once_with, so asserting things separately
    assert mocked_function_under_test.call_count == 1
    actual_array_arg = mocked_function_under_test.call_args_list[0][0][0]
    expected_array_arg = list_of_lambdas_to_get_call_args[0](loaded_generic_pipeline)
    if isinstance(expected_array_arg, np.ndarray):
        np.testing.assert_array_equal(
            actual_array_arg,
            expected_array_arg,
        )
    else:
        assert actual_array_arg == expected_array_arg
    if len(list_of_lambdas_to_get_call_args) == 2:
        actual_array_arg = mocked_function_under_test.call_args_list[0][0][1]
        expected_array_arg = list_of_lambdas_to_get_call_args[1](
            loaded_generic_pipeline
        )
        if isinstance(expected_array_arg, np.ndarray):
            np.testing.assert_array_equal(
                actual_array_arg,
                expected_array_arg,
            )
        else:
            assert actual_array_arg == expected_array_arg

    assert actual_return_1 == expected_return

    actual_return_2 = lambda_of_method_under_test(loaded_generic_pipeline)
    assert mocked_function_under_test.call_count == 1
    assert actual_return_2 == expected_return


def test_Pipeline__get_noise_filtered_gmr__creates_and_uses_filter_supplied_by_template(
    mocker, loaded_generic_pipeline, generic_pipeline_template
):
    mocked_function_under_test = mocker.patch.object(
        pipelines,
        "apply_noise_filtering",
        autospec=True,
    )
    loaded_generic_pipeline.get_noise_filtered_magnetic_data()
    # Eli (7/6/20): NumPy arrays don't play well with assert_called_once_with, so asserting things separately
    assert mocked_function_under_test.call_count == 1
    actual_filter_array = mocked_function_under_test.call_args_list[0][0][1]
    expected_filter_array = generic_pipeline_template.get_filter_coefficients()
    np.testing.assert_array_equal(actual_filter_array, expected_filter_array)


def test_Pipeline__get_noise_filtered_gmr__returns_same_data_if_no_filter_defined(
    raw_generic_well_a1, raw_generic_well_a2
):
    no_filter_pipeline_template = PipelineTemplate(tissue_sampling_period=1000)
    pipeline = no_filter_pipeline_template.create_pipeline()
    pipeline.load_raw_magnetic_data(raw_generic_well_a1, raw_generic_well_a2)
    calibrated_data = pipeline.get_fully_calibrated_gmr()
    filtered_data = pipeline.get_noise_filtered_magnetic_data()
    assert filtered_data is calibrated_data


def test_Pipeline__get_filter_coefficients__does_not_repeatedly_generate_new_filters_each_time(
    mocker, generic_pipeline_template
):
    spied_bessel = mocker.spy(signal, "bessel")
    original_coefficients = generic_pipeline_template.get_filter_coefficients()
    assert spied_bessel.call_count == 1  # confirm pre-condition

    new_coefficients = generic_pipeline_template.get_filter_coefficients()
    assert spied_bessel.call_count == 1
    assert new_coefficients is original_coefficients


def test_Pipeline__get_peak_detection_info__passes_twitches_point_up_parameter_when_default_false(
    mocker, raw_generic_well_a1
):
    pt = PipelineTemplate(100)
    pipeline = pt.create_pipeline()
    pipeline.load_raw_magnetic_data(raw_generic_well_a1, raw_generic_well_a1)
    mocked_peak_detection = mocker.patch.object(
        pipelines, "peak_detector", autospec=True
    )
    pipeline.get_peak_detection_results()
    mocked_peak_detection.assert_called_once_with(ANY, twitches_point_up=True)


def test_Pipeline__get_peak_detection_info__passes_force_data_parameter_when_true(
    mocker, raw_generic_well_a1
):
    pt = PipelineTemplate(100, is_force_data=True)
    pipeline = pt.create_pipeline()
    pipeline.load_raw_magnetic_data(raw_generic_well_a1, raw_generic_well_a1)
    mocked_peak_detection = mocker.patch.object(
        pipelines, "peak_detector", autospec=True
    )
    pipeline.get_peak_detection_results()
    mocked_peak_detection.assert_called_once_with(ANY, twitches_point_up=True)


def test_Pipeline__get_peak_detection_info__passes_force_data_parameter_when_false(
    mocker, raw_generic_well_a1
):
    pt = PipelineTemplate(100, is_force_data=False)
    pipeline = pt.create_pipeline()
    pipeline.load_raw_magnetic_data(raw_generic_well_a1, raw_generic_well_a1)
    mocked_peak_detection = mocker.patch.object(
        pipelines, "peak_detector", autospec=True
    )
    pipeline.get_peak_detection_results()
    mocked_peak_detection.assert_called_once_with(ANY, twitches_point_up=False)
