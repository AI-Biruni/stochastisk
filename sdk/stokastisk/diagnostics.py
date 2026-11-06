"""Runtime diagnostics. torch is imported lazily so this module works on CPU-only hosts."""
from __future__ import annotations

import importlib
import logging
import platform
import sys
from typing import Any

from ._version import CUDA_BUILD_TARGET, GPU_COMPUTE_CAPABILITY, VLLM_BASE_VERSION, __version__
from .errors import EnvironmentValidationError, GPUDiscoveryError

log = logging.getLogger("stokastisk")

NATIVE_EXTENSIONS = (
    "vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch",
    "vllm.cumem_allocator", "vllm.spinloop", "vllm.fs_io_C",
)


def discover_gpus() -> list[dict[str, Any]]:
    import torch
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append({"index": i, "name": p.name, "compute_capability": (p.major, p.minor),
                    "total_memory": p.total_memory})
    return out


def peer_access_matrix() -> dict[str, bool]:
    import torch
    n = torch.cuda.device_count()
    return {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b))
            for a in range(n) for b in range(n) if a != b}


def check_native_extensions() -> dict[str, str]:
    """Import every bundled extension; value is 'ok' or the error text."""
    res = {}
    for name in NATIVE_EXTENSIONS:
        try:
            importlib.import_module(name)
            res[name] = "ok"
        except Exception as e:  # noqa: BLE001 - diagnostics must never raise
            res[name] = f"{type(e).__name__}: {e}"
    return res


FLASH_NATIVE_MODULES = ("vllm.vllm_flash_attn._vllm_fa2_C", "vllm.vllm_flash_attn._vllm_fa3_C")


def flash_attn_native_loaded() -> list[str]:
    """FlashAttention native modules present in this process (expected: none)."""
    return [m for m in FLASH_NATIVE_MODULES if m in sys.modules]


PROHIBITED_DISTS = ("flash-attn", "flash_attn", "vllm-flash-attn", "flashinfer", "flashinfer-python", "flashinfer-cubin",
                    "flashinfer-jit-cache")
PROHIBITED_MODULES = ("flashinfer", "flash_attn", "vllm.vllm_flash_attn")


def prohibited_report() -> dict[str, Any]:
    """Detect FlashAttention / FlashInfer distributions, modules and native binaries (expected: all empty)."""
    import importlib.metadata as md
    import importlib.util
    import pathlib
    norm = lambda s: s.replace("_", "-").lower()  # noqa: E731
    dists = sorted({d.metadata["Name"] for d in md.distributions() if d.metadata["Name"] and norm(d.metadata["Name"]) in {norm(x) for x in PROHIBITED_DISTS}})
    mods = []
    for m in PROHIBITED_MODULES:
        try:
            if importlib.util.find_spec(m) is not None:
                mods.append(m)
        except (ImportError, ValueError):
            pass
    bins = []
    try:
        spec = importlib.util.find_spec("vllm")
        root = pathlib.Path(next(iter(spec.submodule_search_locations))) if spec and spec.submodule_search_locations else None
        if root:
            bins = sorted(str(p.relative_to(root)) for p in root.rglob("*.so") if any(k in p.name.lower() for k in ("flash", "_fa2", "_fa3", "_fa4")))
    except Exception:  # noqa: BLE001
        pass
    return {"distributions": dists, "modules": mods, "binaries": bins, "clean": not (dists or mods or bins)}


def validate_environment(*, require_gpus: int = 2, strict_torch: bool = True) -> dict[str, Any]:
    """Validate the runtime against the SDK build target (CUDA 13.0 / SM75 / torch 2.11 cu130)."""
    import torch
    if sys.version_info[:2] != (3, 13):
        raise EnvironmentValidationError(f"CPython 3.13 required, found {platform.python_version()}")
    if strict_torch and not str(torch.__version__).startswith("2.11.0"):
        raise EnvironmentValidationError(f"torch 2.11.0 required, found {torch.__version__}")
    if torch.version.cuda is None or not torch.version.cuda.startswith(CUDA_BUILD_TARGET):
        raise EnvironmentValidationError(f"PyTorch CUDA {CUDA_BUILD_TARGET} required, found {torch.version.cuda}")
    pr = prohibited_report()
    if not pr["clean"]:
        raise EnvironmentValidationError(f"FlashAttention/FlashInfer must not be present: {pr}")
    gpus = discover_gpus()
    if len(gpus) < require_gpus:
        raise GPUDiscoveryError(f"Need {require_gpus} GPU(s), found {len(gpus)}")
    for g in gpus:
        if tuple(g["compute_capability"]) != GPU_COMPUTE_CAPABILITY:
            raise GPUDiscoveryError(f"GPU {g['index']} ({g['name']}) is SM{g['compute_capability']}; "
                                    f"this build targets SM75 only")
    info = {"sdk_version": __version__, "vllm_source_version": VLLM_BASE_VERSION,
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "gpus": gpus,
            "peer_access": peer_access_matrix() if len(gpus) > 1 else {}}
    log.info("environment ok: %s", info)
    return info


def report() -> dict[str, Any]:
    """Non-raising environment report (safe on CPU hosts)."""
    r: dict[str, Any] = {"sdk_version": __version__, "python": platform.python_version()}
    try:
        import torch
        r.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                 cuda_available=torch.cuda.is_available(), gpus=discover_gpus())
    except Exception as e:  # noqa: BLE001
        r["torch_error"] = str(e)
    try:
        import vllm
        r["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        r["vllm_error"] = f"{type(e).__name__}: {e}"
    r["flash_attn_native_loaded"] = flash_attn_native_loaded()
    r["prohibited"] = prohibited_report()
    r["native_extensions"] = check_native_extensions()
    return r
