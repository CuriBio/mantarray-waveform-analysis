# -*- coding: utf-8 -*-
"""Docstring."""
from . import pipelines
from . import transforms
from .constants import ADC_GAIN
from .constants import AMPLITUDE_UUID
from .constants import AUC_UUID
from .constants import BESSEL_BANDPASS_UUID
from .constants import BESSEL_LOWPASS_10_UUID
from .constants import BESSEL_LOWPASS_30_UUID
from .constants import BUTTERWORTH_LOWPASS_30_UUID
from .constants import CENTIMILLISECONDS_PER_SECOND
from .constants import CONTRACTION_VELOCITY_UUID
from .constants import MIDSCALE_CODE
from .constants import MILLI_TO_BASE_CONVERSION
from .constants import MILLIMETERS_PER_MILLITESLA
from .constants import MILLIVOLTS_PER_MILLITESLA
from .constants import MIN_NUMBER_PEAKS
from .constants import MIN_NUMBER_VALLEYS
from .constants import NEWTONS_PER_MILLIMETER
from .constants import PRIOR_PEAK_INDEX_UUID
from .constants import PRIOR_VALLEY_INDEX_UUID
from .constants import RAW_TO_SIGNED_CONVERSION_VALUE
from .constants import REFERENCE_VOLTAGE
from .constants import RELAXATION_VELOCITY_UUID
from .constants import SUBSEQUENT_PEAK_INDEX_UUID
from .constants import SUBSEQUENT_VALLEY_INDEX_UUID
from .constants import TWITCH_FREQUENCY_UUID
from .constants import TWITCH_PERIOD_UUID
from .constants import WIDTH_FALLING_COORDS_UUID
from .constants import WIDTH_RISING_COORDS_UUID
from .constants import WIDTH_UUID
from .constants import WIDTH_VALUE_UUID
from .exceptions import DataAlreadyLoadedInPipelineError
from .exceptions import FilterCreationNotImplementedError
from .exceptions import TooFewPeaksDetectedError
from .exceptions import TwoPeaksInARowError
from .exceptions import TwoValleysInARowError
from .exceptions import UnrecognizedFilterUuidError
from .peak_detection import find_twitch_indices
from .peak_detection import peak_detector
from .pipelines import Pipeline
from .pipelines import PipelineTemplate
from .transforms import apply_empty_plate_calibration
from .transforms import apply_noise_filtering
from .transforms import apply_sensitivity_calibration
from .transforms import calculate_displacement_from_voltage
from .transforms import calculate_force_from_displacement
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
    "TWITCH_FREQUENCY_UUID",
    "PRIOR_PEAK_INDEX_UUID",
    "PRIOR_VALLEY_INDEX_UUID",
    "SUBSEQUENT_PEAK_INDEX_UUID",
    "SUBSEQUENT_VALLEY_INDEX_UUID",
    "AMPLITUDE_UUID",
    "AUC_UUID",
    "WIDTH_UUID",
    "WIDTH_VALUE_UUID",
    "WIDTH_RISING_COORDS_UUID",
    "WIDTH_FALLING_COORDS_UUID",
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
    "TwoPeaksInARowError",
    "TwoValleysInARowError",
    "BESSEL_BANDPASS_UUID",
    "BESSEL_LOWPASS_10_UUID",
    "FILTER_CHARACTERISTICS",
    "compress_filtered_gmr",
    "calculate_voltage_from_gmr",
    "calculate_displacement_from_voltage",
    "PipelineTemplate",
    "Pipeline",
    "peak_detector",
    "find_twitch_indices",
    "TooFewPeaksDetectedError",
    "MIN_NUMBER_PEAKS",
    "MIN_NUMBER_VALLEYS",
    "BESSEL_LOWPASS_30_UUID",
    "BUTTERWORTH_LOWPASS_30_UUID",
    "RELAXATION_VELOCITY_UUID",
    "CONTRACTION_VELOCITY_UUID",
    "calculate_force_from_displacement",
    "MILLI_TO_BASE_CONVERSION",
    "MILLIVOLTS_PER_MILLITESLA",
    "MILLIMETERS_PER_MILLITESLA",
    "NEWTONS_PER_MILLIMETER",
    "REFERENCE_VOLTAGE",
    "ADC_GAIN",
]
