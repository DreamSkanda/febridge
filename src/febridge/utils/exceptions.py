"""Custom exceptions for febridge."""


class FebridgeError(Exception):
    """Base exception for febridge package."""
    pass


class SampleSizeError(FebridgeError):
    """Raised when sample set size is invalid."""
    pass


class SizeMismatchError(FebridgeError):
    """Raised when X0 and X1 sizes do not match."""
    pass
