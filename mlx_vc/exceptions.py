"""Structured exceptions for mlx-vc.

Raised in place of `print(...) + sys.exit(1)` patterns at the library
boundary, so callers (CLI, FastAPI server, downstream Python apps) can
distinguish failure modes and produce sensible UX.

Hierarchy:

    MlxVcError                        — base
        ├── BackendError              — subprocess backend failed
        │   ├── BackendNotFoundError  — backend name unknown
        │   ├── ModelNotAvailableError — required weights/repos missing
        │   └── ConversionError       — model ran but output is invalid
        └── ConfigError               — env var / config file mismatch
"""

from __future__ import annotations

from typing import Optional


class MlxVcError(Exception):
    """Base class for all mlx-vc errors."""


class ConfigError(MlxVcError):
    """Configuration / environment-variable problem before any model runs."""


class BackendError(MlxVcError):
    """Subprocess backend failed.

    Attributes:
        backend: Name of the backend (e.g. "seed-vc").
        returncode: Exit code from the subprocess (None if it didn't run).
        stderr: Last ~500 chars of stderr (truncated for readability).
    """

    def __init__(
        self,
        backend: str,
        message: str,
        *,
        returncode: Optional[int] = None,
        stderr: Optional[str] = None,
    ):
        self.backend = backend
        self.returncode = returncode
        self.stderr = stderr
        prefix = f"[{backend}]"
        if returncode is not None:
            prefix += f" exit {returncode}"
        super().__init__(f"{prefix} {message}")


class BackendNotFoundError(BackendError):
    """Requested backend name is not registered in BACKENDS."""

    def __init__(self, backend: str, available: list):
        super().__init__(
            backend,
            f"unknown backend; available: {sorted(available)}",
        )


class ModelNotAvailableError(BackendError):
    """Required weights, reference repo, or environment is missing.

    Use this for "user hasn't completed setup" errors that should produce
    actionable messages rather than tracebacks.
    """


class ConversionError(BackendError):
    """Model ran to completion but the output is missing or invalid."""
