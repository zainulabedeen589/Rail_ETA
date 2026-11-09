"""Custom exception hierarchy for Rail ETA.

Every domain error inherits from :class:`RailEtaError` so callers can catch
the entire family with a single ``except`` clause while still being able to
distinguish specific failure modes.
"""


class RailEtaError(Exception):
    """Base exception for all Rail ETA errors."""

    def __init__(self, message: str, code: str = "RAIL_ETA_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


class ConfigurationError(RailEtaError):
    """Raised when settings are invalid or missing."""


class DataLoadError(RailEtaError):
    """Raised when the dataset cannot be loaded or parsed."""


class SchemaValidationError(RailEtaError):
    """Raised when the dataset schema does not match expectations."""


class PreprocessingError(RailEtaError):
    """Raised during time parsing or journey-duration computation."""


class ModelNotTrainedError(RailEtaError):
    """Raised when inference is attempted before a model exists."""


class ModelRegistryError(RailEtaError):
    """Raised when a requested model is not registered."""


class InferenceError(RailEtaError):
    """Raised when prediction fails for any reason."""
