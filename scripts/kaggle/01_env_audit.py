"""Stage 01: environment audit. READ-ONLY: installs nothing, changes nothing. Writes 01_env_audit.json (+ package inventory)."""
import ctypes, json, os, platform, re, shutil, sys
from _common import OUT, check, finish, sh

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
info["nvcc"] = sh("nvcc --version | tail -2 | head -1") if shutil.which("nvcc") else "nvcc not found (not required at runtime)"
info["glibc"] = platform.libc_ver()[1]
g = tuple(int(x) for x in re.findall(r"\d+", info["glibc"])[:2]) if info["glibc"] else (0, 0)
check(R, "glibc >= 2.38 (native extensions need 2.38)", g >= (2, 38), info["glibc"])
try:
    cu = ctypes.CDLL("libcuda.so.1"); rc = cu.cuInit(0); info["cuInit"] = rc
except OSError as e:
    rc = -1; info["cuInit"] = str(e)
check(R, "CUDA driver initialises (cuInit(0)==0)", rc == 0, rc)
n = torch.cuda.device_count(); info["gpu_count"] = n
check(R, "two GPUs visible to PyTorch", n == 2 and torch.cuda.is_available(), n)
gpus = []
for i in range(n):
    p = torch.cuda.get_device_properties(i); free, total = torch.cuda.mem_get_info(i)
    gpus.append({"index": i, "name": p.name, "cc": (p.major, p.minor), "total_MiB": total >> 20, "free_MiB": free >> 20})
    check(R, f"gpu{i} Tesla T4 / SM75", "T4" in p.name and (p.major, p.minor) == (7, 5), f"{p.name} {p.major}.{p.minor}")
info["gpus"] = gpus
info["peer_access"] = {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b)) for a in range(n) for b in range(n) if a != b}
info["topology"] = sh("nvidia-smi topo -m"); info["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
import importlib.metadata as md
inv = {d.metadata["Name"]: d.version for d in md.distributions() if d.metadata["Name"]}
(OUT / "01_package_inventory.json").write_text(json.dumps(dict(sorted(inv.items())), indent=0))
info["package_count"] = len(inv)
info["relevant_packages"] = {k: v for k, v in inv.items() if re.match(r"(?i)(torch|triton|numpy|transformers|tokenizers|huggingface|nvidia-(cuda-nvcc|nvvm|nccl)|flash|vllm|stokastisk|fastapi|xgrammar|msgspec)", k)}
check(R, "baseline has no FlashAttention/FlashInfer", not [k for k in inv if re.match(r"(?i)flash", k)], [k for k in inv if re.match(r"(?i)flash", k)] or "none")
print(json.dumps(info, indent=1, default=str))
finish("01_env_audit", R, {"environment": info})
