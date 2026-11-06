"""Stage 03: single-GPU validation, each T4 individually.
WHAT IS ACTUALLY TESTED (and what is not):
  * PyTorch fp32/fp16 matmul vs a CPU reference            -> PyTorch/cuBLAS on SM75, NOT vLLM code
  * a hand-written Triton kernel                            -> Triton 3.6 JIT on SM75, NOT vLLM's attention kernel
  * vllm._custom_ops.rms_norm / silu_and_mul on the GPU     -> REAL vLLM SM75 native kernels (vllm/_C_stable_libtorch)
  * static ELF checks of every installed vLLM .so            -> no CUDA 12 linkage, sm_75 cubins present
vLLM's Triton attention kernel itself is exercised in stage 05 (real model)."""
import glob, os, re, subprocess
import torch
from _common import check, finish

R = []
import stokastisk, vllm
ext = stokastisk.check_native_extensions()
check(R, "vLLM native extensions import", all(v == "ok" for v in ext.values()), ext)
for so in sorted(glob.glob(os.path.join(os.path.dirname(vllm.__file__), "*.so"))):
    needed = subprocess.run(["readelf", "-d", so], capture_output=True, text=True).stdout
    c12 = re.findall(r"lib\w*\.so\.12\b", needed)
    check(R, f"{os.path.basename(so)}: no CUDA 12 NEEDED", not c12, c12 or "ok")
    if "stable_libtorch" in so:
        cuo = subprocess.run(["cuobjdump", "--list-elf", so], capture_output=True, text=True)
        if cuo.returncode == 0:
            archs = sorted(set(re.findall(r"sm_\d+", cuo.stdout)))
            check(R, f"{os.path.basename(so)}: cubins are exactly sm_75", archs == ["sm_75"], archs)
        else:
            check(R, f"{os.path.basename(so)}: cuobjdump unavailable (informational)", True, "cuobjdump not found; skipped")
import triton, triton.language as tl

@triton.jit
def _add(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

from vllm import _custom_ops as ops
for d in range(torch.cuda.device_count()):
    torch.cuda.set_device(d); dev = f"cuda:{d}"
    for dt in (torch.float32, torch.float16):
        torch.manual_seed(0)
        a = torch.randn(512, 512, dtype=dt); b = torch.randn(512, 512, dtype=dt)
        ref = a.float() @ b.float(); got = (a.to(dev) @ b.to(dev)).float().cpu()
        check(R, f"gpu{d} PyTorch {dt} matmul vs CPU", torch.allclose(ref, got, rtol=5e-2, atol=5e-1), f"max err {(ref-got).abs().max():.4g}")
    x = torch.randn(10000, device=dev); y = torch.randn(10000, device=dev); o = torch.empty_like(x)
    _add[(triton.cdiv(10000, 1024),)](x, y, o, 10000, B=1024); torch.cuda.synchronize()
    check(R, f"gpu{d} hand-written Triton kernel", torch.equal(o, x + y))
    try:
        w = torch.ones(128, device=dev, dtype=torch.float16); xi = torch.randn(4, 128, device=dev, dtype=torch.float16); out = torch.empty_like(xi)
        ops.rms_norm(out, xi, w, 1e-6); torch.cuda.synchronize()
        ref = xi.float() * torch.rsqrt(xi.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        check(R, f"gpu{d} vLLM native rms_norm (SM75 cubin)", torch.allclose(out.float(), ref, atol=2e-2))
        xs = torch.randn(4, 256, device=dev, dtype=torch.float16); outs = torch.empty(4, 128, device=dev, dtype=torch.float16)
        torch.ops._C.silu_and_mul(outs, xs); torch.cuda.synchronize()
        refs = torch.nn.functional.silu(xs[:, :128].float()) * xs[:, 128:].float()
        check(R, f"gpu{d} vLLM native silu_and_mul (SM75 cubin)", torch.allclose(outs.float(), refs, atol=2e-2))
    except Exception as e:
        check(R, f"gpu{d} vLLM native kernels", False, f"{type(e).__name__}: {e}")
pr = stokastisk.prohibited_report()
check(R, "no FlashAttention/FlashInfer distributions, modules or binaries", pr["clean"], pr)
finish("03_single_gpu", R)
