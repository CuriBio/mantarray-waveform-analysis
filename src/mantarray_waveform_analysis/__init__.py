# -*- coding: utf-8 -*-
"""Docstring."""
from . import pipelines
from . import transforms
from .constants import AMPLITUDE_UUID
from .constants import AUC_UUID
from .constants import BESSEL_BANDPASS_UUID
from .constants import BESSEL_LOWPASS_10_UUID
from .constants import CENTIMILLISECONDS_PER_SECOND
from .constants import MIDSCALE_CODE
from .constants import RAW_TO_SIGNED_CONVERSION_VALUE
from .constants import TWITCH_PERIOD_UUID
from .exceptions import DataAlreadyLoadedInPipelineError
from .exceptions import FilterCreationNotImplementedError
from .exceptions import UnrecognizedFilterUuidError
from .pipelines import Pipeline
from .pipelines import PipelineTemplate
from .transforms import apply_empty_plate_calibration
from .transforms import apply_noise_filtering
from .transforms import apply_sensitivity_calibration
from .transforms import calculate_displacement_from_voltage
from .transforms import calculate_voltage_from_gmr
from .transforms import create_filter
from .transforms import FILTER_CHARACTERISTICS
from .transforms import noise_cancellation

if (
    5 < 10
):  # pragma: no cover # protect this from zimports deleting the pylint disable statement
    from .compression_cy import (  # pylint: disable=import-error # Eli (8/18/20) unsure why pylint is unable to recognize cython import... # Tanner (8/31/20) Pylint also flags this as duplicate code due to a similar import in pipelines.py, which may be related to pylint's import issues
        compress_filtered_gmr,
    )

__all__ = [
    "transforms",
    "pipelines",
    "TWITCH_PERIOD_UUID",
    "AMPLITUDE_UUID",
    "AUC_UUID",
    "CENTIMILLISECONDS_PER_SECOND",
    "MIDSCALE_CODE",
    "RAW_TO_SIGNED_CONVERSION_VALUE",
    "apply_sensitivity_calibration",
    "noise_cancellation",
    "apply_empty_plate_calibration",
    "apply_noise_filtering",
    "create_filter",
    "UnrecognizedFilterUuidError",
    "FilterCreationNotImplementedError",
    "DataAlreadyLoadedInPipelineError",
    "BESSEL_BANDPASS_UUID",
    "BESSEL_LOWPASS_10_UUID",
    "FILTER_CHARACTERISTICS",
    "compress_filtered_gmr",
    "calculate_voltage_from_gmr",
    "calculate_displacement_from_voltage",
    "PipelineTemplate",
    "Pipeline",
]
