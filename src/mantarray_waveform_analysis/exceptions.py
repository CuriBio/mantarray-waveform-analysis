# -*- coding: utf-8 -*-
"""Generic exceptions for the Mantarray SDK."""


class UnrecognizedFilterUuidError(Exception):
    pass


class FilterCreationNotImplementedError(Exception):
    pass


class DataAlreadyLoadedInPipelineError(Exception):
    pass
