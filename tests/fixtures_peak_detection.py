# -*- coding: utf-8 -*-
import os

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
        os.path.join("noisy_data", "noisy_data_A1.tsv"),
        sampling_rate_construct=600,
        flip_data=True,
    )
