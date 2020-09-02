# -*- coding: utf-8 -*-
import uuid

from mantarray_waveform_analysis import AMPLITUDE_UUID
from mantarray_waveform_analysis import AUC_UUID
from mantarray_waveform_analysis import BESSEL_BANDPASS_UUID
from mantarray_waveform_analysis import BESSEL_LOWPASS_10_UUID
from mantarray_waveform_analysis import BESSEL_LOWPASS_30_UUID
from mantarray_waveform_analysis import CENTIMILLISECONDS_PER_SECOND
from mantarray_waveform_analysis import FILTER_CHARACTERISTICS
from mantarray_waveform_analysis import MIDSCALE_CODE
from mantarray_waveform_analysis import MIN_NUMBER_PEAKS
from mantarray_waveform_analysis import PRIOR_PEAK_INDEX_UUID
from mantarray_waveform_analysis import PRIOR_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import RAW_TO_SIGNED_CONVERSION_VALUE
from mantarray_waveform_analysis import SUBSEQUENT_PEAK_INDEX_UUID
from mantarray_waveform_analysis import SUBSEQUENT_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import TWITCH_PERIOD_UUID
from mantarray_waveform_analysis import WIDTH_FALLING_COORDS_UUID
from mantarray_waveform_analysis import WIDTH_RISING_COORDS_UUID
from mantarray_waveform_analysis import WIDTH_UUID
from mantarray_waveform_analysis import WIDTH_VALUE_UUID


def test_misc_constants():
    assert CENTIMILLISECONDS_PER_SECOND == 100000


def test_filter_uuids():
    assert BESSEL_BANDPASS_UUID == uuid.UUID("0ecf0e52-0a29-453f-a6ff-46f5ec3ae783")
    assert BESSEL_LOWPASS_10_UUID == uuid.UUID("7d64cac3-b841-4912-b734-c0cf20a81e7a")
    assert BESSEL_LOWPASS_30_UUID == uuid.UUID("eee66c75-4dc4-4eb4-8d48-6c608bf28d91")


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
        BESSEL_LOWPASS_30_UUID: {
            "filter_type": "bessel",
            "order": 4,
            "low_pass_hz": 30,
        },
    }


def test_data_metric_uuids():
    assert TWITCH_PERIOD_UUID == uuid.UUID("6e0cd81c-7861-4c49-ba14-87b2739d65fb")
    assert AMPLITUDE_UUID == uuid.UUID("89cf1105-a015-434f-b527-4169b9400e26")
    assert AUC_UUID == uuid.UUID("e7b9a6e4-c43d-4e8b-af7e-51742e252030")
    assert WIDTH_UUID == uuid.UUID("c4c60d55-017a-4783-9600-f19606de26f3")
    assert WIDTH_VALUE_UUID == uuid.UUID("05041f4e-c77d-42d9-a2ae-8902f912e9ac")
    assert WIDTH_RISING_COORDS_UUID == uuid.UUID("2a16acb6-4df7-4064-9d47-5d27ea7a98ad")
    assert WIDTH_FALLING_COORDS_UUID == uuid.UUID(
        "26e5637d-42c9-4060-aa5d-52209b349c84"
    )

    assert PRIOR_PEAK_INDEX_UUID == uuid.UUID("80df90dc-21f8-4cad-a164-89436909b30a")
    assert PRIOR_VALLEY_INDEX_UUID == uuid.UUID("72ba9466-c203-41b6-ac30-337b4a17a124")
    assert SUBSEQUENT_PEAK_INDEX_UUID == uuid.UUID(
        "7e37325b-6681-4623-b192-39f154350f36"
    )
    assert SUBSEQUENT_VALLEY_INDEX_UUID == uuid.UUID(
        "fd47ba6b-ee4d-4674-9a89-56e0db7f3d97"
    )


def test_gmr_conversion_factors():
    assert MIDSCALE_CODE == 8388608
    assert RAW_TO_SIGNED_CONVERSION_VALUE == 2 ** 23


def test_peak_detection_vals():
    assert MIN_NUMBER_PEAKS == 3
