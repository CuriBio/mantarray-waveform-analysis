# -*- coding: utf-8 -*-
import csv
import inspect
import os
from typing import List
from typing import Tuple

from mantarray_waveform_analysis import peak_detection
import numpy as np
import pytest

PATH_OF_CURRENT_FILE = os.path.dirname((inspect.stack()[0][1]))

PATH_TO_DATASETS = os.path.join(PATH_OF_CURRENT_FILE, "datasets")
PATH_TO_PNGS = os.path.join(PATH_OF_CURRENT_FILE, "pngs")

CONVERSION_TO_INT = 100000


def _load_file(file_path: str) -> Tuple[List[str], List[str]]:
    time = []
    v = []
    header_placer = []  # used to get rid of the header
    with open(file_path, "r") as file_name:
        file_reader = csv.reader(file_name, delimiter=",")
        header = next(file_reader)
        header_placer.append(header)
        for row in file_reader:
            # row variable is a list that represents a row in csv
            time.append(row[0])
            v.append(row[1])
    return time, v


def _load_file_tsv(file_path: str) -> Tuple[List[str], List[str]]:
    time = []
    v = []
    with open(file_path, "r") as file_name:
        file_reader = csv.reader(file_name, delimiter="\t")
        for row in file_reader:
            time.append(row[0])
            v.append(row[1])
    return time, v


def create_numpy_array_of_raw_gmr_from_python_arrays(time_array, gmr_array):
    time = np.array(time_array, dtype=np.int32)
    v = np.array(gmr_array, dtype=np.int32)

    data = np.zeros((2, len(time_array)), dtype=np.int32)
    for i in range(len(time_array)):
        data[0, i] = time[i]
        data[1, i] = v[i]
    return data


def _run_peak_detection(filename, sampling_rate_construct=100, flip_data=True):
    sampling_rate = sampling_rate_construct
    time_series = np.arange(
        0, 1000000, int(CONVERSION_TO_INT / sampling_rate_construct)
    )
    my_local_path_data_2 = os.path.join(PATH_TO_DATASETS, filename)

    time, v = _load_file_tsv(my_local_path_data_2)

    # create numpy matrix
    data = create_numpy_array_of_raw_gmr_from_python_arrays(time, v)

    # flip the data
    if flip_data:
        data = peak_detection.convert_voltage_to_displacement(data)

    # noise cancellation
    noise_free_data, sampling_rate = peak_detection.noise_filtering(data, sampling_rate)

    # call peak_detection
    peakind = peak_detection.peak_detector(noise_free_data, sampling_rate, data)

    return data, time_series, peakind, sampling_rate, noise_free_data


@pytest.fixture(scope="session", name="raw_generic_well_a1")
def fixture_raw_generic_well_a1():
    time, gmr = _load_file_tsv(os.path.join(PATH_TO_DATASETS, "new_A1_tsv.tsv"))
    raw_gmr_data = create_numpy_array_of_raw_gmr_from_python_arrays(time, gmr)
    return raw_gmr_data


@pytest.fixture(scope="session", name="raw_generic_well_a2")
def fixture_raw_generic_well_a2():
    time, gmr = _load_file_tsv(os.path.join(PATH_TO_DATASETS, "new_A2_tsv.tsv"))
    raw_gmr_data = create_numpy_array_of_raw_gmr_from_python_arrays(time, gmr)
    return raw_gmr_data
