class ChronoFrameBuilderError(Exception):
    """Base exception for chrono-frame-builder."""


class BodyNotFoundError(ChronoFrameBuilderError):
    """Raised when a body cannot be found in a project."""


class FrameNotFoundError(ChronoFrameBuilderError):
    """Raised when a frame cannot be found in a project."""


class InvalidFrameError(ChronoFrameBuilderError):
    """Raised when a frame definition is invalid."""


class InvalidBodyError(ChronoFrameBuilderError):
    """Raised when a body definition is invalid."""
