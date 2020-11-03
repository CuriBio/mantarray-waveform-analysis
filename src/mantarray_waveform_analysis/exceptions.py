# -*- coding: utf-8 -*-
"""Generic exceptions for the Mantarray SDK."""
from typing import Any
from typing import Tuple

from nptyping import NDArray
import numpy as np


class UnrecognizedFilterUuidError(Exception):
    pass


class FilterCreationNotImplementedError(Exception):
    pass


class DataAlreadyLoadedInPipelineError(Exception):
    pass


class PeakDetectionError(Exception):
    pass


class InsufficientPeaksDetectedError(PeakDetectionError):
    pass


class TwoFeaturesInARowError(PeakDetectionError):
    """There must always be a peak in between valleys and vice-versa."""

    def __init__(
        self,
        peak_and_valley_timepoints: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        back_to_back_points: Tuple[int, int],
        feature_name: str = "feature",
    ) -> None:
        peak_timepoints, valley_timepoints = peak_and_valley_timepoints
        with np.printoptions(threshold=np.inf):  # don't truncate the numpy array

            base_msg = f"Detected peak timepoints: {peak_timepoints}\nDetected valley timepoints: {valley_timepoints}\nData used in peak detection: {filtered_data}\n"
        prepend_msg = f"Two back-to-back {feature_name}s in a row were detected at timepoints: {back_to_back_points[0]} and {back_to_back_points[1]}\n"
        super().__init__(prepend_msg + base_msg)


class TwoValleysInARowError(TwoFeaturesInARowError):
    """There must always be a peak in between valleys."""

    def __init__(
        self,
        peak_and_valley_timepoints: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        back_to_back_points: Tuple[int, int],
    ) -> None:
        super().__init__(
            peak_and_valley_timepoints,
            filtered_data,
            back_to_back_points,
            feature_name="valley",
        )


class TwoPeaksInARowError(TwoFeaturesInARowError):
    """There must always be a valley in between peaks."""

    def __init__(
        self,
        peak_and_valley_timepoints: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        back_to_back_points: Tuple[int, int],
    ) -> None:
        super().__init__(
            peak_and_valley_timepoints,
            filtered_data,
            back_to_back_points,
            feature_name="peak",
        )


class TooFewPeaksDetectedError(PeakDetectionError):
    pass
