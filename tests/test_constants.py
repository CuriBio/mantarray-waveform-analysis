# -*- coding: utf-8 -*-
import uuid

from mantarray_waveform_analysis import ADC_GAIN
from mantarray_waveform_analysis import ALL_METRICS
from mantarray_waveform_analysis import AMPLITUDE_UUID
from mantarray_waveform_analysis import AUC_UUID
from mantarray_waveform_analysis import BESSEL_BANDPASS_UUID
from mantarray_waveform_analysis import BESSEL_LOWPASS_10_UUID
from mantarray_waveform_analysis import BESSEL_LOWPASS_30_UUID
from mantarray_waveform_analysis import BUTTERWORTH_LOWPASS_30_UUID
from mantarray_waveform_analysis import CENTIMILLISECONDS_PER_SECOND
from mantarray_waveform_analysis import CONTRACTION_TIME_UUID
from mantarray_waveform_analysis import CONTRACTION_VELOCITY_UUID
from mantarray_waveform_analysis import FILTER_CHARACTERISTICS
from mantarray_waveform_analysis import FRACTION_MAX_UUID
from mantarray_waveform_analysis import IRREGULARITY_INTERVAL_UUID
from mantarray_waveform_analysis import MIDSCALE_CODE
from mantarray_waveform_analysis import MILLI_TO_BASE_CONVERSION
from mantarray_waveform_analysis import MILLIMETERS_PER_MILLITESLA
from mantarray_waveform_analysis import MILLIVOLTS_PER_MILLITESLA
from mantarray_waveform_analysis import MIN_NUMBER_PEAKS
from mantarray_waveform_analysis import MIN_NUMBER_VALLEYS
from mantarray_waveform_analysis import NEWTONS_PER_MILLIMETER
from mantarray_waveform_analysis import PRIOR_PEAK_INDEX_UUID
from mantarray_waveform_analysis import PRIOR_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import RAW_TO_SIGNED_CONVERSION_VALUE
from mantarray_waveform_analysis import REFERENCE_VOLTAGE
from mantarray_waveform_analysis import RELAXATION_TIME_UUID
from mantarray_waveform_analysis import RELAXATION_VELOCITY_UUID
from mantarray_waveform_analysis import SUBSEQUENT_PEAK_INDEX_UUID
from mantarray_waveform_analysis import SUBSEQUENT_VALLEY_INDEX_UUID
from mantarray_waveform_analysis import TIME_DIFFERENCE_UUID
from mantarray_waveform_analysis import TWITCH_FREQUENCY_UUID
from mantarray_waveform_analysis import TWITCH_PERIOD_UUID
from mantarray_waveform_analysis import WIDTH_FALLING_COORDS_UUID
from mantarray_waveform_analysis import WIDTH_RISING_COORDS_UUID
from mantarray_waveform_analysis import WIDTH_UUID
from mantarray_waveform_analysis import WIDTH_VALUE_UUID


def test_misc_constants():
    assert CENTIMILLISECONDS_PER_SECOND == 100000
    assert MILLI_TO_BASE_CONVERSION == 1000


def test_filter_uuids():
    assert BESSEL_BANDPASS_UUID == uuid.UUID("0ecf0e52-0a29-453f-a6ff-46f5ec3ae783")
    assert BESSEL_LOWPASS_10_UUID == uuid.UUID("7d64cac3-b841-4912-b734-c0cf20a81e7a")
    assert BESSEL_LOWPASS_30_UUID == uuid.UUID("eee66c75-4dc4-4eb4-8d48-6c608bf28d91")
    assert BUTTERWORTH_LOWPASS_30_UUID == uuid.UUID("de8d8cef-65bf-4119-ada7-bdecbbaa897a")


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
        BUTTERWORTH_LOWPASS_30_UUID: {
            "filter_type": "butterworth",
            "order": 4,
            "low_pass_hz": 30,
        },
    }


def test_data_metric_uuids():
    assert TWITCH_PERIOD_UUID == uuid.UUID("6e0cd81c-7861-4c49-ba14-87b2739d65fb")
    assert TWITCH_FREQUENCY_UUID == uuid.UUID("472d0707-ff87-4198-9374-c28900bb216c")
    assert AMPLITUDE_UUID == uuid.UUID("89cf1105-a015-434f-b527-4169b9400e26")
    assert AUC_UUID == uuid.UUID("e7b9a6e4-c43d-4e8b-af7e-51742e252030")
    assert FRACTION_MAX_UUID == uuid.UUID('d9758590-9f0d-418d-8913-db3bec2be65f')
    assert TIME_DIFFERENCE_UUID == uuid.UUID("1363817a-b1fb-468e-9f1c-ec54fce72dfe")
    assert RELAXATION_TIME_UUID == uuid.UUID("0ad56cd1-7bcc-4b57-8076-14366d7f3c6a")
    assert CONTRACTION_TIME_UUID == uuid.UUID("33b5b0a8-f197-46ef-a451-a254e530757b")
    assert WIDTH_UUID == uuid.UUID("c4c60d55-017a-4783-9600-f19606de26f3")
    assert WIDTH_VALUE_UUID == uuid.UUID("05041f4e-c77d-42d9-a2ae-8902f912e9ac")
    assert WIDTH_RISING_COORDS_UUID == uuid.UUID("2a16acb6-4df7-4064-9d47-5d27ea7a98ad")
    assert WIDTH_FALLING_COORDS_UUID == uuid.UUID("26e5637d-42c9-4060-aa5d-52209b349c84")
    assert RELAXATION_VELOCITY_UUID == uuid.UUID("0fcc0dc3-f9aa-4f1b-91b3-e5b5924279a9")
    assert CONTRACTION_VELOCITY_UUID == uuid.UUID("73961e7c-17ec-42b0-b503-a23195ec249c")
    assert TIME_DIFFERENCE_UUID == uuid.UUID("58ae3d02-df1b-419c-925a-7a772053bddf")
    assert RELAXATION_TIME_UUID == uuid.UUID("f263e0d0-c55e-49b2-b5a6-93e9a7836be5")
    assert CONTRACTION_TIME_UUID == uuid.UUID("5a92d051-703b-412b-b4e9-968e48331e8b")

    assert PRIOR_PEAK_INDEX_UUID == uuid.UUID("80df90dc-21f8-4cad-a164-89436909b30a")
    assert PRIOR_VALLEY_INDEX_UUID == uuid.UUID("72ba9466-c203-41b6-ac30-337b4a17a124")
    assert SUBSEQUENT_PEAK_INDEX_UUID == uuid.UUID("7e37325b-6681-4623-b192-39f154350f36")
    assert SUBSEQUENT_VALLEY_INDEX_UUID == uuid.UUID("fd47ba6b-ee4d-4674-9a89-56e0db7f3d97")
    assert IRREGULARITY_INTERVAL_UUID == uuid.UUID("61046076-66b9-4b8b-bfec-1e00603743c0")

    assert ALL_METRICS == frozenset(
        [
            TWITCH_PERIOD_UUID,
            FRACTION_MAX_UUID,
            AMPLITUDE_UUID,
            WIDTH_UUID,
            AUC_UUID,
            TWITCH_FREQUENCY_UUID,
            CONTRACTION_VELOCITY_UUID,
            RELAXATION_VELOCITY_UUID,
            IRREGULARITY_INTERVAL_UUID,
            TIME_DIFFERENCE_UUID,
        ]
    )


def test_gmr_conversion_factors():
    assert MIDSCALE_CODE == 8388608
    assert RAW_TO_SIGNED_CONVERSION_VALUE == 2 ** 23
    assert MILLIVOLTS_PER_MILLITESLA == 1073.6
    assert MILLIMETERS_PER_MILLITESLA == 23.25
    assert NEWTONS_PER_MILLIMETER == 0.000159
    assert REFERENCE_VOLTAGE == 2.5
    assert ADC_GAIN == 2


def test_peak_detection_vals():
    assert MIN_NUMBER_PEAKS == 3
    assert MIN_NUMBER_VALLEYS == 3
