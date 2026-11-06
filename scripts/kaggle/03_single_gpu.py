"""Stage 03: per-GPU SM75 execution (matmul vs CPU reference, Triton kernel) + native-extension/ELF checks."""
import glob, os, re, subprocess, sys
import torch
from _common import check, finish

R = []
import stokastisk, vllm
ext = stokastisk.check_native_extensions()
check(R, "native extensions import", all(v == "ok" for v in ext.values()), ext)
for so in sorted(glob.glob(os.path.join(os.path.dirname(vllm.__file__), "*.so"))):
    needed = subprocess.run(["readelf", "-d", so], capture_output=True, text=True).stdout
    c12 = re.findall(r"lib\w*\.so\.12\b", needed)
    check(R, f"{os.path.basename(so)} has no CUDA12 NEEDED", not c12, c12 or "ok")
    if "stable_libtorch" in so:
        cu = subprocess.run(["cuobjdump", "--list-elf", so], capture_output=True, text=True)
        if cu.returncode == 0:
            check(R, f"{os.path.basename(so)} contains sm_75 cubin", "sm_75" in cu.stdout)
import triton, triton.language as tl

@triton.jit
def _add(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

for d in range(torch.cuda.device_count()):
    torch.cuda.set_device(d)
    for dt in (torch.float32, torch.float16):
        torch.manual_seed(0)
        a = torch.randn(512, 512, dtype=dt); b = torch.randn(512, 512, dtype=dt)
        ref = (a.float() @ b.float())
        got = (a.to(f"cuda:{d}") @ b.to(f"cuda:{d}")).float().cpu()
        check(R, f"gpu{d} {dt} matmul vs CPU", torch.allclose(ref, got, rtol=5e-2, atol=5e-1), f"max err {(ref-got).abs().max():.4g}")
    x = torch.randn(10000, device=f"cuda:{d}"); y = torch.randn(10000, device=f"cuda:{d}"); o = torch.empty_like(x)
    _add[(triton.cdiv(10000, 1024),)](x, y, o, 10000, B=1024); torch.cuda.synchronize()
    check(R, f"gpu{d} triton kernel", torch.equal(o, x + y))
    # torch/vllm custom op smoke: rms_norm from the SM75 cubins
    try:
        from vllm import _custom_ops as ops
        w = torch.ones(128, device=f"cuda:{d}", dtype=torch.float16); xi = torch.randn(4, 128, device=f"cuda:{d}", dtype=torch.float16)
        out = torch.empty_like(xi); ops.rms_norm(out, xi, w, 1e-6); torch.cuda.synchronize()
        ref = xi.float() * torch.rsqrt(xi.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        check(R, f"gpu{d} vllm _C rms_norm (SM75 kernel)", torch.allclose(out.float(), ref, atol=1e-2))
    except Exception as e:
        check(R, f"gpu{d} vllm _C rms_norm (SM75 kernel)", False, f"{type(e).__name__}: {e}")
check(R, "no FlashAttention native module loaded", stokastisk.report()["flash_attn_native_loaded"] == [])
finish("03_single_gpu", R)
