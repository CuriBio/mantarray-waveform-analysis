# -*- coding: utf-8 -*-
import csv
import os
from typing import List
from typing import Tuple

from mantarray_file_manager import WellFile
from mantarray_waveform_analysis import BESSEL_LOWPASS_30_UUID
from mantarray_waveform_analysis import CENTIMILLISECONDS_PER_SECOND
from mantarray_waveform_analysis import peak_detection
from mantarray_waveform_analysis import peak_detector
from mantarray_waveform_analysis import PipelineTemplate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from stdlib_utils import get_current_file_abs_directory


matplotlib.use("Agg")
PATH_OF_CURRENT_FILE = get_current_file_abs_directory()


PATH_TO_DATASETS = os.path.join(PATH_OF_CURRENT_FILE, "datasets")
PATH_TO_PNGS = os.path.join(PATH_OF_CURRENT_FILE, "pngs")


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


def _load_file_h5(file_path: str) -> Tuple[List[str], List[str]]:
    wf = WellFile(file_path)
    tissue_data = wf.get_raw_tissue_reading()
    # only analyze first 21 seconds of data so png graphs are readablw
    last_index = 0
    for idx, t in enumerate(tissue_data[0]):
        last_index = idx
        if t > CENTIMILLISECONDS_PER_SECOND * 21:
            break
    return tissue_data[0][:last_index], tissue_data[1][:last_index]


def create_numpy_array_of_raw_gmr_from_python_arrays(time_array, gmr_array):
    time = np.array(time_array, dtype=np.int32)
    v = np.array(gmr_array, dtype=np.int32)

    data = np.array([time, v], dtype=np.int32)
    return data


def _run_peak_detection(
    filename, sampling_rate_construct=100, flip_data=True, time_scaling_factor=None
):
    the_path = os.path.join(PATH_TO_DATASETS, filename)
    time, v = (
        _load_file_h5(the_path)
        if filename.endswith(".h5")
        else _load_file_tsv(the_path)
    )
    noise_filter_uuid = BESSEL_LOWPASS_30_UUID if filename.endswith(".h5") else None
    # create numpy matrix
    raw_data = create_numpy_array_of_raw_gmr_from_python_arrays(time, v)
    simple_pipeline_template = PipelineTemplate(
        tissue_sampling_period=1 / sampling_rate_construct * 100000,
        noise_filter_uuid=noise_filter_uuid,
    )
    pipeline = simple_pipeline_template.create_pipeline()
    pipeline.load_raw_gmr_data(raw_data, raw_data)
    filtered_data = pipeline.get_noise_filtered_gmr()
    peak_and_valley_timepoints = peak_detector(
        filtered_data, twitches_point_up=not flip_data
    )
    return pipeline, peak_and_valley_timepoints


def _plot_data(peak_and_valley_indices, filtered_data, my_local_path_graphs):
    time_series = filtered_data[0, :]
    peak_indices, valley_indices = peak_and_valley_indices
    plt.figure()
    plt.plot(time_series, filtered_data[1, :], "b")
    plt.plot(
        time_series[peak_indices],
        filtered_data[1][peak_indices],
        "ro",
        label="peaks",
        fillstyle="none",
    )
    plt.plot(
        time_series[valley_indices],
        filtered_data[1][valley_indices],
        "go",
        label="valleys",
        fillstyle="none",
    )
    plt.xlabel("Time (centimilliseconds)")
    plt.ylabel("Magnetic Signal")
    plt.legend(bbox_to_anchor=(0, -0.25), loc="lower center")
    plt.tight_layout()
    plt.savefig(my_local_path_graphs)
    plt.close()


def _get_data_metrics(well_fixture):
    pipeline, peak_and_valley_indices = well_fixture
    filtered_data = pipeline.get_noise_filtered_gmr()
    return peak_detection.data_metrics(peak_and_valley_indices, filtered_data)


def assert_percent_diff(actual, expected, threshold=0.0006):
    percent_diff = abs(actual - expected) / expected
    assert percent_diff < threshold


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
