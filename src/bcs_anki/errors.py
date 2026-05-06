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

    def __init__(self, status_code: int, excerpt: str = "") -> None:
        message = f"HTTP {status_code}"
        if excerpt:
            message = f"{message}: {excerpt}"
        super().__init__(message)
        self.status_code = status_code


class ImageError(CustomError):
    """Base class for image-pipeline failures."""


class ImageRejectedError(ImageError):
    """Raised when an AI image generator refuses a prompt (e.g. safety filter)."""


class ImageProviderError(ImageError):
    """Raised on provider-side failures during image fetch (HTTP, decoding, etc.)."""


class NoStockResultsError(ImageError):
    """Raised when a stock-image search returns zero usable results."""


class UnsupportedStockProviderError(ConfigError):
    """Raised when stock_image_api is set to a value with no implementation."""
