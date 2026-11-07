"""Typed configuration with T4-safe defaults and environment overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

from .errors import ConfigurationError
from .limits import LIMITS, cpu_thread_env

ALLOWED_ATTENTION_BACKENDS = ("TRITON_ATTN",)
FORBIDDEN_ATTENTION_MARKERS = ("FLASH", "FA2", "FA3", "FA4")
_BOOL = {"1": True, "true": True, "yes": True, "on": True, "0": False, "false": False, "no": False, "off": False}

# NCCL settings for 2x T4 without CUDA peer access (Kaggle: PHB topology, no P2P).
NCCL_ENV_NO_P2P: dict[str, str] = {
    "NCCL_P2P_DISABLE": "1",   # peer access unavailable -> go through shared memory / host
    "NCCL_IB_DISABLE": "1",    # no InfiniBand on Kaggle
    "NCCL_CUMEM_ENABLE": "0",  # vLLM cumem sleep-mode allocator is not used here
}

# vLLM FlashInfer integrations are not shipped in this build; pin them off (user-set values are never overridden).
VLLM_ENV_NO_FLASHINFER: dict[str, str] = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
}


@dataclass
class StokastiskConfig:
    tensor_parallel_size: int = 1                # 1 or 2; TP=2 adds NCCL traffic over PCIe/SHM (no P2P)
    dtype: str = "half"                       # T4 has no bf16 tensor cores
    kv_cache_dtype: str = "auto"
    attention_backend: str = "TRITON_ATTN"    # vLLM built-in Triton backend; FlashAttention is never used
    enforce_eager: bool = True
    disable_custom_all_reduce: bool = True    # custom all-reduce requires P2P
    gpu_memory_utilization: float = 0.88
    max_model_len: int | None = 4096
    apply_nccl_env: bool = True
    nccl_env: dict[str, str] = field(default_factory=lambda: dict(NCCL_ENV_NO_P2P))

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        b = str(self.attention_backend).upper()
        if any(m in b for m in FORBIDDEN_ATTENTION_MARKERS):
            raise ConfigurationError(f"FlashAttention backends are not supported by this release: {self.attention_backend!r}")
        if b not in ALLOWED_ATTENTION_BACKENDS:
            raise ConfigurationError(f"attention_backend must be one of {ALLOWED_ATTENTION_BACKENDS}, got {self.attention_backend!r}")
        self.attention_backend = b
        if self.tensor_parallel_size not in (1, 2):
            raise ConfigurationError("tensor_parallel_size must be 1 or 2 (two Tesla T4 target)")
        if not 0.1 <= self.gpu_memory_utilization <= 0.98:
            raise ConfigurationError("gpu_memory_utilization must be within [0.1, 0.98]")
        if self.gpu_memory_utilization * LIMITS.gpu_vram_gib > LIMITS.gpu_vram_gib:   # pragma: no cover - guards future edits of the bounds
            raise ConfigurationError("gpu_memory_utilization exceeds the Kaggle T4 VRAM")
        if self.dtype in ("bfloat16", "bf16"):
            raise ConfigurationError("T4 (SM75) has no bfloat16 support; use 'half'")

    @classmethod
    def from_env(cls, prefix: str = "STOKASTISK_") -> "StokastiskConfig":
        def b(x: str) -> bool:
            if x.lower() not in _BOOL:
                raise ConfigurationError(f"invalid boolean {x!r}")
            return _BOOL[x.lower()]
        cfg = cls()
        for name, ftype in (("tensor_parallel_size", int), ("gpu_memory_utilization", float),
                            ("max_model_len", int), ("dtype", str), ("attention_backend", str),
                            ("enforce_eager", b), ("disable_custom_all_reduce", b)):
            raw = os.environ.get(prefix + name.upper())
            if raw is not None:
                try:
                    setattr(cfg, name, ftype(raw))
                except ValueError as e:
                    raise ConfigurationError(f"{prefix}{name.upper()}={raw!r}: {e}") from e
        cfg.validate()
        return cfg

    def apply_environment(self) -> dict[str, str]:
        """Export NCCL settings (never overriding values the user already set)."""
        applied = {}
        for k, v in VLLM_ENV_NO_FLASHINFER.items():
            os.environ.setdefault(k, v)
        for k, v in cpu_thread_env(self.tensor_parallel_size).items():   # 4 Kaggle cores shared by the TP worker processes
            os.environ.setdefault(k, v)
        if self.apply_nccl_env:
            for k, v in self.nccl_env.items():
                if k not in os.environ:
                    os.environ[k] = v
                    applied[k] = v
        return applied

    def engine_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = dict(
            tensor_parallel_size=self.tensor_parallel_size, dtype=self.dtype,
            kv_cache_dtype=self.kv_cache_dtype, enforce_eager=self.enforce_eager,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            gpu_memory_utilization=self.gpu_memory_utilization,
            attention_config={"backend": self.attention_backend},
        )
        if self.max_model_len is not None:
            kw["max_model_len"] = self.max_model_len
        return kw

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
