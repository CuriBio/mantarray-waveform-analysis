# -*- coding: utf-8 -*-
import uuid

from mantarray_waveform_analysis import AMPLITUDE_UUID
from mantarray_waveform_analysis import AUC_UUID
from mantarray_waveform_analysis import BESSEL_BANDPASS_UUID
from mantarray_waveform_analysis import BESSEL_LOWPASS_10_UUID
from mantarray_waveform_analysis import CENTIMILLISECONDS_PER_SECOND
from mantarray_waveform_analysis import FILTER_CHARACTERISTICS
from mantarray_waveform_analysis import MIDSCALE_CODE
from mantarray_waveform_analysis import RAW_TO_SIGNED_CONVERSION_VALUE
from mantarray_waveform_analysis import TWITCH_PERIOD_UUID


def test_misc_constants():
    assert CENTIMILLISECONDS_PER_SECOND == 100000


def test_filter_uuids():
    assert BESSEL_BANDPASS_UUID == uuid.UUID("0ecf0e52-0a29-453f-a6ff-46f5ec3ae783")
    assert BESSEL_LOWPASS_10_UUID == uuid.UUID("7d64cac3-b841-4912-b734-c0cf20a81e7a")


def test_filter_characteristics():
    assert FILTER_CHARACTERISTICS == {
        BESSEL_BANDPASS_UUID: {
            "filter_type": "bessel",
            "order": 4,
            "high_pass_hz": 0.1,
            "low_pass_hz": 10,
        },
        BESSEL_LOWPASS_10_UUID: {
            "filter_type": "bessel",
            "order": 4,
            "low_pass_hz": 10,
        },
    }


def test_data_metric_uuids():
    assert TWITCH_PERIOD_UUID == uuid.UUID("6e0cd81c-7861-4c49-ba14-87b2739d65fb")
    assert AMPLITUDE_UUID == uuid.UUID("89cf1105-a015-434f-b527-4169b9400e26")
    assert AUC_UUID == uuid.UUID("e7b9a6e4-c43d-4e8b-af7e-51742e252030")


def test_gmr_conversion_factors():
    assert MIDSCALE_CODE == 8388608
    assert RAW_TO_SIGNED_CONVERSION_VALUE == 2 ** 23
