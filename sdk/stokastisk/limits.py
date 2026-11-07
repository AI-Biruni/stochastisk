"""Hard resource limits of the Kaggle target and helpers that enforce / report them.

Kaggle (GPU T4 x2 session): 4 CPU cores, ~30 GiB system RAM, /kaggle/working limited to 20 GiB, two Tesla T4 (SM75) with ~15 GiB VRAM each.
These are the design limits of the SDK; on other hosts the same checks report WARNING instead of pretending the limits are met."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class KaggleLimits:
    cpu_cores: int = 4
    system_ram_gib: float = 30.0
    working_dir: str = "/kaggle/working"
    working_dir_gib: float = 20.0
    gpu_count: int = 2
    gpu_name_contains: str = "T4"
    gpu_compute_capability: tuple[int, int] = (7, 5)
    gpu_vram_gib: float = 15.0


LIMITS = KaggleLimits()


def available_cpus() -> int:
    """CPUs this process may really use (affinity-aware; equals `nproc`), never more than the Kaggle budget of 4 for planning."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux
        n = os.cpu_count() or 1
    return max(1, n)


def cpu_budget() -> int:
    return max(1, min(available_cpus(), LIMITS.cpu_cores))


def threads_per_worker(tensor_parallel_size: int) -> int:
    """Split the CPU budget across the TP worker processes (4 cores, TP=2 -> 2 threads each) so CPU-side libraries never oversubscribe."""
    return max(1, cpu_budget() // max(1, tensor_parallel_size))


def cpu_thread_env(tensor_parallel_size: int) -> dict[str, str]:
    n = str(threads_per_worker(tensor_parallel_size))
    return {k: n for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def system_ram_gib() -> float | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return None


def dir_size_bytes(path: str | os.PathLike) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def working_dir_status(path: str | None = None) -> dict[str, Any]:
    """Usage of the (quota-limited) output directory against the 20 GiB Kaggle limit."""
    path = path or LIMITS.working_dir
    if not os.path.isdir(path):
        return {"path": path, "present": False, "status": "SKIPPED"}
    used = dir_size_bytes(path) / 1024 ** 3
    free = shutil.disk_usage(path).free / 1024 ** 3
    st = "FAIL" if used > LIMITS.working_dir_gib else ("WARNING" if used > 0.75 * LIMITS.working_dir_gib else "PASS")
    return {"path": path, "present": True, "used_gib": round(used, 3), "quota_gib": LIMITS.working_dir_gib, "filesystem_free_gib": round(free, 1), "status": st}


def check_resources() -> dict[str, Any]:
    """Compare the host with the Kaggle limits. Never raises. Statuses: PASS (within the limit) / WARNING (host differs from the Kaggle design)."""
    ram = system_ram_gib()
    cpus = available_cpus()
    out: dict[str, Any] = {"limits": asdict(LIMITS), "cpu": {"nproc": cpus, "budget": cpu_budget(), "status": "PASS" if cpus <= LIMITS.cpu_cores else "WARNING"},
                           "ram": {"total_gib": None if ram is None else round(ram, 1), "status": "UNKNOWN" if ram is None else ("PASS" if ram <= LIMITS.system_ram_gib * 1.1 else "WARNING")},
                           "working_dir": working_dir_status()}
    try:
        import torch
        gpus = [{"index": i, "name": torch.cuda.get_device_name(i), "capability": tuple(torch.cuda.get_device_capability(i)),
                 "vram_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 2)} for i in range(torch.cuda.device_count())]
    except Exception as e:  # noqa: BLE001
        gpus = []; out["gpu_error"] = f"{type(e).__name__}: {e}"
    ok = len(gpus) == LIMITS.gpu_count and all(LIMITS.gpu_name_contains in g["name"] and g["capability"] == LIMITS.gpu_compute_capability and g["vram_gib"] <= LIMITS.gpu_vram_gib + 0.5 for g in gpus)
    out["gpus"] = {"found": gpus, "status": "PASS" if ok else "WARNING"}
    return out
