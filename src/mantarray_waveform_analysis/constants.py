# -*- coding: utf-8 -*-
"""Global constants."""
import uuid

CENTIMILLISECONDS_PER_SECOND = 100000

TWITCH_PERIOD_UUID = uuid.UUID("6e0cd81c-7861-4c49-ba14-87b2739d65fb")
TWITCH_FREQUENCY_UUID = uuid.UUID(
    "472d0707-ff87-4198-9374-c28900bb216c"
)  # This is just the reciprocal of twitch period, but is pre-computed to make downstream pipelines simpler. Frequency is reported in Hz
AMPLITUDE_UUID = uuid.UUID("89cf1105-a015-434f-b527-4169b9400e26")
FRACTION_MAX_UUID = uuid.UUID("8fe142e2-2504-4c9e-b3dc-817b24c7447e")
AUC_UUID = uuid.UUID("e7b9a6e4-c43d-4e8b-af7e-51742e252030")
WIDTH_UUID = uuid.UUID("c4c60d55-017a-4783-9600-f19606de26f3")
WIDTH_VALUE_UUID = uuid.UUID("05041f4e-c77d-42d9-a2ae-8902f912e9ac")
WIDTH_RISING_COORDS_UUID = uuid.UUID("2a16acb6-4df7-4064-9d47-5d27ea7a98ad")
WIDTH_FALLING_COORDS_UUID = uuid.UUID("26e5637d-42c9-4060-aa5d-52209b349c84")
RELAXATION_VELOCITY_UUID = uuid.UUID("0fcc0dc3-f9aa-4f1b-91b3-e5b5924279a9")
CONTRACTION_VELOCITY_UUID = uuid.UUID("73961e7c-17ec-42b0-b503-a23195ec249c")
IRREGULARITY_INTERVAL_UUID = uuid.UUID("61046076-66b9-4b8b-bfec-1e00603743c0")

TIME_DIFFERENCE_UUID = uuid.UUID("1363817a-b1fb-468e-9f1c-ec54fce72dfe")
RELAXATION_TIME_UUID = uuid.UUID("0ad56cd1-7bcc-4b57-8076-14366d7f3c6a")
CONTRACTION_TIME_UUID = uuid.UUID("33b5b0a8-f197-46ef-a451-a254e530757b")

ALL_METRICS = frozenset(
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


PRIOR_PEAK_INDEX_UUID = uuid.UUID("80df90dc-21f8-4cad-a164-89436909b30a")
PRIOR_VALLEY_INDEX_UUID = uuid.UUID("72ba9466-c203-41b6-ac30-337b4a17a124")
SUBSEQUENT_PEAK_INDEX_UUID = uuid.UUID("7e37325b-6681-4623-b192-39f154350f36")
SUBSEQUENT_VALLEY_INDEX_UUID = uuid.UUID("fd47ba6b-ee4d-4674-9a89-56e0db7f3d97")

BESSEL_BANDPASS_UUID = uuid.UUID("0ecf0e52-0a29-453f-a6ff-46f5ec3ae783")
BESSEL_LOWPASS_10_UUID = uuid.UUID("7d64cac3-b841-4912-b734-c0cf20a81e7a")
BESSEL_LOWPASS_30_UUID = uuid.UUID("eee66c75-4dc4-4eb4-8d48-6c608bf28d91")
BUTTERWORTH_LOWPASS_30_UUID = uuid.UUID("de8d8cef-65bf-4119-ada7-bdecbbaa897a")

# GMR conversion factors
# Conversion values were obtained 03/09/2021 by Kevin Grey
MIDSCALE_CODE = 0x800000
RAW_TO_SIGNED_CONVERSION_VALUE = 2 ** 23  # subtract this value from raw hardware data
MILLIVOLTS_PER_MILLITESLA = 1073.6
MILLIMETERS_PER_MILLITESLA = 23.25
NEWTONS_PER_MILLIMETER = 0.000159
REFERENCE_VOLTAGE = 2.5
ADC_GAIN = 2
MILLI_TO_BASE_CONVERSION = 1000


MIN_NUMBER_PEAKS = 3
MIN_NUMBER_VALLEYS = 3
