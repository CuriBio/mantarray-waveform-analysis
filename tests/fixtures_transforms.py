# -*- coding: utf-8 -*-
"""Fixtures for testing the transforms."""

from mantarray_waveform_analysis import BESSEL_LOWPASS_10_UUID
from mantarray_waveform_analysis import create_filter
from mantarray_waveform_analysis import MICRO_TO_BASE_CONVERSION
import pytest


@pytest.fixture(scope="session", name="bessel_lowpass_10_for_100hz")
def fixture_bessel_lowpass_10_for_100hz():
    return create_filter(BESSEL_LOWPASS_10_UUID, (1 / 100) * MICRO_TO_BASE_CONVERSION)
