# -*- coding: utf-8 -*-
"""Transforming arrays of Mantarray data throughout the analysis pipeline."""
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
import uuid

import attr
from nptyping import NDArray
import numpy as np

from .exceptions import DataAlreadyLoadedInPipelineError
from .peak_detection import peak_detector
from .transforms import apply_empty_plate_calibration
from .transforms import apply_noise_filtering
from .transforms import apply_sensitivity_calibration
from .transforms import calculate_displacement_from_voltage
from .transforms import calculate_voltage_from_gmr
from .transforms import create_filter
from .transforms import noise_cancellation

if (
    6 < 9
):  # pragma: no cover # protect this from zimports deleting the pylint disable statement
    from .compression_cy import (  # pylint: disable=import-error # Eli (8/18/20) unsure why pylint is unable to recognize cython import...
        compress_filtered_gmr,
    )


class Pipeline:
    """A pipeline used for analysis of a set of data.

    Args:
        pipeline_template: The PipelineTemplate instance that should be used to control the settings and operation of this analysis pipeline.
    """

    # pylint:disable=too-many-instance-attributes # Eli (9/8/20): it's a pipeline with lots of stages
    def __init__(self, pipeline_template: "PipelineTemplate") -> None:
        self._pipeline_template = pipeline_template
        self._raw_tissue_gmr: NDArray[(2, Any), int]
        self._raw_reference_gmr: NDArray[(2, Any), int]
        self._sensitivity_calibrated_tissue_gmr: NDArray[(2, Any), int]
        self._sensitivity_calibrated_reference_gmr: NDArray[(2, Any), int]
        self._noise_cancelled_gmr: NDArray[(2, Any), int]
        self._fully_calibrated_gmr: NDArray[(2, Any), int]
        self._noise_filtered_gmr: NDArray[(2, Any), int]
        self._compressed_gmr: NDArray[(2, Any), int]
        self._compressed_voltage: NDArray[(2, Any), np.float32]
        self._compressed_displacement: NDArray[(2, Any), np.float32]
        self._peak_detection_results: Tuple[List[int], List[int]]

    def get_template(self) -> "PipelineTemplate":
        return self._pipeline_template

    def load_raw_magnetic_data(
        self,
        raw_tissue_magnetic_data: NDArray[(2, Any), int],
        raw_reference_magnetic_data: NDArray[(2, Any), int],
    ) -> None:
        """Load the raw magnetic sensor numpy arrays into the pipeline.

        Args:
            raw_tissue_magnetic_data: 2D array of time (in centimilliseconds) and magnetic reading for the sensor under the tissue construct
            raw_reference_magnetic_data: 2D array of time (in centimilliseconds) and magnetic reading for the sensor nearby the tissue construct
        """
        try:
            self._raw_tissue_gmr
        except AttributeError:
            self._raw_tissue_gmr = raw_tissue_magnetic_data
            self._raw_reference_gmr = raw_reference_magnetic_data
            return
        raise DataAlreadyLoadedInPipelineError()

    def load_raw_gmr_data(
        self,
        raw_tissue_gmr: NDArray[(2, Any), int],
        raw_reference_gmr: NDArray[(2, Any), int],
    ) -> None:
        """Call magnetic data using this deprecated alias."""
        return self.load_raw_magnetic_data(raw_tissue_gmr, raw_reference_gmr)

    def get_raw_tissue_gmr(self) -> NDArray[(2, Any), int]:
        return self._raw_tissue_gmr

    def get_raw_reference_gmr(self) -> NDArray[(2, Any), int]:
        return self._raw_reference_gmr

    def get_sensitivity_calibrated_tissue_gmr(self) -> NDArray[(2, Any), int]:
        try:
            return self._sensitivity_calibrated_tissue_gmr
        except AttributeError:
            pass
        self._sensitivity_calibrated_tissue_gmr = apply_sensitivity_calibration(
            self._raw_tissue_gmr
        )
        return self._sensitivity_calibrated_tissue_gmr

    def get_sensitivity_calibrated_reference_gmr(self) -> NDArray[(2, Any), int]:
        try:
            return self._sensitivity_calibrated_reference_gmr
        except AttributeError:
            pass
        self._sensitivity_calibrated_reference_gmr = apply_sensitivity_calibration(
            self._raw_reference_gmr
        )
        return self._sensitivity_calibrated_reference_gmr

    def get_noise_cancelled_gmr(self) -> NDArray[(2, Any), int]:
        try:
            return self._noise_cancelled_gmr
        except AttributeError:
            pass
        self._noise_cancelled_gmr = noise_cancellation(
            self.get_sensitivity_calibrated_tissue_gmr(),
            self.get_sensitivity_calibrated_reference_gmr(),
        )
        return self._noise_cancelled_gmr

    def get_fully_calibrated_gmr(self) -> NDArray[(2, Any), int]:
        try:
            return self._fully_calibrated_gmr
        except AttributeError:
            pass
        self._fully_calibrated_gmr = apply_empty_plate_calibration(
            self.get_noise_cancelled_gmr()
        )
        return self._fully_calibrated_gmr

    def get_noise_filtered_gmr(self) -> NDArray[(2, Any), int]:
        """Return data after user-acceptable noise filtering."""
        try:
            return self._noise_filtered_gmr
        except AttributeError:
            pass
        if self._pipeline_template.noise_filter_uuid is None:
            self._noise_filtered_gmr = self.get_fully_calibrated_gmr()
        else:
            self._noise_filtered_gmr = apply_noise_filtering(
                self.get_fully_calibrated_gmr(),
                self._pipeline_template.get_filter_coefficients(),
            )
        return self._noise_filtered_gmr

    def get_peak_detection_results(self) -> Tuple[List[int], List[int]]:
        """Return peak detection results on noise filtered magnetic data."""
        try:
            return self._peak_detection_results
        except AttributeError:
            pass
        self._peak_detection_results = peak_detector(
            self.get_noise_filtered_gmr(),
            twitches_point_up=self._pipeline_template.magnetic_twitches_point_up,
        )
        return self._peak_detection_results

    def get_compressed_gmr(self) -> NDArray[(2, Any), int]:
        try:
            return self._compressed_gmr
        except AttributeError:
            pass
        self._compressed_gmr = compress_filtered_gmr(self.get_noise_filtered_gmr())
        return self._compressed_gmr

    def get_compressed_voltage(self) -> NDArray[(2, Any), np.float32]:
        try:
            return self._compressed_voltage
        except AttributeError:
            pass
        self._compressed_voltage = calculate_voltage_from_gmr(self.get_compressed_gmr())
        return self._compressed_voltage

    def get_compressed_displacement(self) -> NDArray[(2, Any), np.float32]:
        try:
            return self._compressed_displacement
        except AttributeError:
            pass
        self._compressed_displacement = calculate_displacement_from_voltage(
            self.get_compressed_voltage()
        )
        return self._compressed_displacement


@attr.s
class PipelineTemplate:  # pylint: disable=too-few-public-methods # This is a simple dataclass/attrs module with just a few methods for validation and pipeline creation
    """A template the defines how an analysis pipeline should run.

    Args:
        noise_filter_uuid: The UUID corresponding to the filter settings that should be used when computationally filtering noise in the data.
        tissue_sampling_period: the sampling period for the tissues, in centimilliseconds
    """

    tissue_sampling_period: int = attr.ib()
    noise_filter_uuid: Optional[uuid.UUID] = attr.ib(default=None)
    magnetic_twitches_point_up: bool = attr.ib(default=False)
    _filter_coefficients: NDArray[(Any, Any), float]

    def create_pipeline(self) -> Pipeline:
        return Pipeline(self)

    def get_filter_coefficients(self) -> NDArray[(Any, Any), float]:
        """Get the coefficients for the signal filter.

        Creating a filter can take a non-trivial amount of time, so the
        coefficients are cached after the first time they are generated
        (since they are immutable within the template)
        """
        if self.noise_filter_uuid is None:
            raise NotImplementedError("Cannot create a filter when no UUID is set.")
        try:
            return self._filter_coefficients
        except AttributeError:
            pass
        self._filter_coefficients = create_filter(
            self.noise_filter_uuid, self.tissue_sampling_period
        )
        return self._filter_coefficients
