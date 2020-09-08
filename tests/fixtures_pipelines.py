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


@pytest.fixture(scope="function", name="generic_pipeline_template")
def fixture_generic_pipeline_template():
    template = PipelineTemplate(
        noise_filter_uuid=BESSEL_LOWPASS_10_UUID, tissue_sampling_period=1000
    )
    yield template


@pytest.fixture(scope="function", name="generic_pipeline")
def fixture_generic_pipeline(generic_pipeline_template):
    pipeline = generic_pipeline_template.create_pipeline()
    yield pipeline


@pytest.fixture(scope="function", name="loaded_generic_pipeline")
def fixture_loaded_generic_pipeline(
    generic_pipeline, raw_generic_well_a1, raw_generic_well_a2
):
    generic_pipeline.load_raw_magnetic_data(raw_generic_well_a1, raw_generic_well_a2)
    yield generic_pipeline
