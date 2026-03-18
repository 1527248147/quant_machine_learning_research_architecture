"""
Custom exceptions for the quantitative research framework.
"""


class FrameworkError(Exception):
    """Base class for all framework errors."""


class SchemaError(FrameworkError):
    """Raised when a DataFrame does not satisfy its contract."""


class ConfigError(FrameworkError):
    """Raised when configuration is invalid or missing required fields."""


class SourceError(FrameworkError):
    """Raised when raw data cannot be read or normalised."""


class AssetBuildError(FrameworkError):
    """Raised when an asset table cannot be built."""


class PanelBuildError(FrameworkError):
    """Raised when panel_base cannot be assembled."""


class ContractViolation(FrameworkError):
    """Raised when a model-label contract is violated (preflight check)."""


class TargetBuildError(FrameworkError):
    """Raised when a target recipe fails."""


class ViewBuildError(FrameworkError):
    """Raised when a view builder fails (e.g., memmap creation)."""


class TrainingError(FrameworkError):
    """Raised when training fails (e.g., no valid samples, NaN loss)."""


class ModelError(FrameworkError):
    """Raised when model instantiation or forward pass fails."""
