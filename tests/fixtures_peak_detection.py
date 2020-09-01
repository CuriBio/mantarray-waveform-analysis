# -*- coding: utf-8 -*-
import pytest

from .fixtures_utils import _run_peak_detection


@pytest.fixture(scope="session", name="maiden_voyage_data")
def fixture_maiden_voyage_data():
    return _run_peak_detection(
        "maiden_voyage_data.tsv", sampling_rate_construct=200, flip_data=True
    )
