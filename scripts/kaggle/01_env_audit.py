"""Stage 01: environment audit. Writes 01_env_audit.json. Run before installing anything."""
import os, platform, re, sys, shutil
from _common import check, finish, sh, save

R = []; info = {}
info["python"] = platform.python_version()
check(R, "python 3.13", sys.version_info[:2] == (3, 13), info["python"])
import torch
info["torch"] = torch.__version__; info["torch_cuda"] = torch.version.cuda
check(R, "torch 2.11.0+cu130", torch.__version__.startswith("2.11.0") and torch.version.cuda == "13.0", f"{torch.__version__} cuda={torch.version.cuda}")
try:
    import triton; info["triton"] = triton.__version__
except Exception as e:
    info["triton"] = f"ERR {e}"
check(R, "triton 3.6.0", info["triton"].startswith("3.6."), info["triton"])
info["nccl"] = ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None
info["nvcc"] = sh("nvcc --version | tail -1") if shutil.which("nvcc") else "nvcc not found (not required at runtime)"
info["glibc"] = platform.libc_ver()[1]
g = tuple(int(x) for x in re.findall(r"\d+", info["glibc"])[:2]) if info["glibc"] else (0, 0)
check(R, "glibc >= 2.38 (required by vllm/_C_stable_libtorch)", g >= (2, 38), info["glibc"])
n = torch.cuda.device_count(); info["gpu_count"] = n
check(R, "two GPUs visible", n == 2, n)
gpus = []
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    gpus.append({"index": i, "name": p.name, "cc": (p.major, p.minor), "total_MiB": total >> 20, "free_MiB": free >> 20})
    check(R, f"gpu{i} Tesla T4 / SM75", "T4" in p.name and (p.major, p.minor) == (7, 5), f"{p.name} {p.major}.{p.minor}")
info["gpus"] = gpus
info["peer_access"] = {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b)) for a in range(n) for b in range(n) if a != b}
info["topology"] = sh("nvidia-smi topo -m")
info["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
info["pip_freeze_subset"] = sh("python -m pip list --format=freeze 2>/dev/null | grep -iE '^(torch|triton|numpy|transformers|vllm|stokastisk|nvidia-nccl|nvidia-cuda-runtime)'")
print(json_dump := __import__("json").dumps(info, indent=1, default=str))
finish("01_env_audit", R, {"environment": info})
