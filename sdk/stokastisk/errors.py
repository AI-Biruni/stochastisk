"""Stokastisk exception hierarchy."""


class StokastiskError(RuntimeError):
    """Base class for all SDK errors."""


class EnvironmentValidationError(StokastiskError):
    """The runtime does not match the SDK build target."""


class GPUDiscoveryError(EnvironmentValidationError):
    """Required GPUs are missing or have the wrong architecture."""


class EngineError(StokastiskError):
    """vLLM engine failed to initialise or generate."""


class ConfigurationError(StokastiskError, ValueError):
    """Invalid Stokastisk configuration (e.g. a FlashAttention backend was requested)."""


class AttentionBackendError(EngineError):
    """The engine did not end up with the requested Triton attention backend."""
