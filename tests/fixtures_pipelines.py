# -*- coding: utf-8 -*-
"""Fixtures for testing the pipelines."""

from mantarray_waveform_analysis import BESSEL_LOWPASS_10_UUID
from mantarray_waveform_analysis import PipelineTemplate
import pytest

from .fixtures_utils import fixture_raw_generic_well_a1
from .fixtures_utils import fixture_raw_generic_well_a2

__fixtures__ = [
    fixture_raw_generic_well_a1,
    fixture_raw_generic_well_a2,
]


@pytest.fixture(scope="function", name="generic_beta_1_pipeline_template")
def fixture_generic_beta_1_pipeline_template():
    template = PipelineTemplate(
        is_beta_1_data=True, noise_filter_uuid=BESSEL_LOWPASS_10_UUID, tissue_sampling_period=10000
    )
    yield template


@pytest.fixture(scope="function", name="generic_beta_1_pipeline")
def fixture_generic_beta_1_pipeline(generic_beta_1_pipeline_template):
    beta_1_pipeline = generic_beta_1_pipeline_template.create_pipeline()
    yield beta_1_pipeline


@pytest.fixture(scope="function", name="loaded_generic_beta_1_pipeline")
def fixture_loaded_generic_beta_1_pipeline(generic_beta_1_pipeline, raw_generic_well_a1, raw_generic_well_a2):
    generic_beta_1_pipeline.load_raw_magnetic_data(raw_generic_well_a1, raw_generic_well_a2)
    yield generic_beta_1_pipeline


@pytest.fixture(scope="function", name="generic_beta_2_pipeline_template")
def fixture_generic_beta_2_pipeline_template():
    template = PipelineTemplate(
        is_beta_1_data=False, noise_filter_uuid=BESSEL_LOWPASS_10_UUID, tissue_sampling_period=10000
    )
    yield template


@pytest.fixture(scope="function", name="generic_beta_2_pipeline")
def fixture_generic_beta_2_pipeline(generic_beta_2_pipeline_template):
    beta_2_pipeline = generic_beta_2_pipeline_template.create_pipeline()
    yield beta_2_pipeline


@pytest.fixture(scope="function", name="loaded_generic_beta_2_pipeline")
def fixture_loaded_generic_beta_2_pipeline(generic_beta_2_pipeline, raw_generic_well_a1, raw_generic_well_a2):
    generic_beta_2_pipeline.load_raw_magnetic_data(raw_generic_well_a1, raw_generic_well_a2)
    yield generic_beta_2_pipeline
