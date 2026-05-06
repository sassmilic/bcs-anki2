from __future__ import annotations


class CustomError(Exception):
    """Base class for domain-specific errors in bcs-anki."""


class ConfigError(CustomError):
    """Raised when application configuration is missing or invalid."""


class MissingApiKeyError(ConfigError):
    """Raised when a required API key is not configured."""


class UnsupportedConfigFormatError(ConfigError):
    """Raised when a config file has an unsupported format."""


class LlmError(CustomError):
    """Base class for LLM-provider related errors."""


class EmptyLlmResponseError(LlmError):
    """Raised when an LLM provider returns an empty response payload."""


class HttpTransientError(CustomError):
    """Raised for transient HTTP responses that should be retried."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
