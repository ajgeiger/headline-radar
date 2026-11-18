"""Raised when a GDELT GKG zip has not been published yet."""


class FileNotReadyError(RuntimeError):
    """Raised when a GKG zip has not been published yet (HTTP 404)."""
