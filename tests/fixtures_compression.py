# -*- coding: utf-8 -*-
"""Data is from MA 13, Gordana Plate 2 on May 24, 2020."""

import pytest

from .fixtures_utils import _run_peak_detection


@pytest.fixture(scope="session", name="new_A1")
def fixture_new_A1():
    return _run_peak_detection("new_A1_tsv.tsv")


@pytest.fixture(scope="session", name="new_A2")
def fixture_new_A2():
    return _run_peak_detection("new_A2_tsv.tsv")


@pytest.fixture(scope="session", name="new_A3")
def fixture_new_A3():
    return _run_peak_detection("new_A3_tsv.tsv")


@pytest.fixture(scope="session", name="new_A4")
def fixture_new_A4():
    return _run_peak_detection("new_A4_tsv.tsv")


@pytest.fixture(scope="session", name="new_A5")
def fixture_new_A5():
    return _run_peak_detection("new_A5_tsv.tsv")


@pytest.fixture(scope="session", name="new_A6")
def fixture_new_A6():
    return _run_peak_detection("new_A6_tsv.tsv")
