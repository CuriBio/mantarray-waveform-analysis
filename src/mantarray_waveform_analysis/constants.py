# -*- coding: utf-8 -*-
"""Global constants."""
import uuid

CENTIMILLISECONDS_PER_SECOND = 100000
TWITCH_PERIOD_UUID = uuid.UUID("6e0cd81c-7861-4c49-ba14-87b2739d65fb")
AMPLITUDE_UUID = uuid.UUID("89cf1105-a015-434f-b527-4169b9400e26")
AUC_UUID = uuid.UUID("e7b9a6e4-c43d-4e8b-af7e-51742e252030")

BESSEL_BANDPASS_UUID = uuid.UUID("0ecf0e52-0a29-453f-a6ff-46f5ec3ae783")
BESSEL_LOWPASS_10_UUID = uuid.UUID("7d64cac3-b841-4912-b734-c0cf20a81e7a")

# GMR conversion factors
MIDSCALE_CODE = 0x800000
RAW_TO_SIGNED_CONVERSION_VALUE = 2 ** 23  # subtract this value from raw hardware data

# REFERENCE_VOLTAGE = 3.3

# ADC_GAIN = 32
