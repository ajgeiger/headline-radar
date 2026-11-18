"""Error raised by HttpClient after retries are exhausted."""


class HttpError(RuntimeError):
    """Raised when a request still fails after retries."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
