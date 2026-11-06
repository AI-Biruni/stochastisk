"""Stage 03: single-GPU validation, each T4 individually.

WHAT IS ACTUALLY TESTED (and what is not):
  * PyTorch fp32/fp16 matmul vs a CPU reference     -> PyTorch/cuBLAS on SM75, NOT vLLM code
  * a hand-written Triton kernel                     -> Triton 3.6 JIT on SM75, NOT vLLM's attention kernel
  * torch.ops._C.rms_norm / torch.ops._C.silu_and_mul -> REAL vLLM native SM75 kernels (vllm/_C_stable_libtorch).
    (vllm._custom_ops has a rms_norm wrapper but NO silu_and_mul wrapper in 0.25.1; vLLM itself calls
    torch.ops._C.silu_and_mul(result, input). Schemas: rms_norm(result, input, weight?, eps), silu_and_mul(result, input).
    The ops have no CPU implementation, so a pass proves the CUDA kernel ran.)
  * static ELF/cubin audit of EVERY installed vLLM .so (recursive)
vLLM's Triton attention kernel itself is exercised in stage 05 (real model).

Every stage is isolated: one failure is recorded with its full traceback and the remaining stages still run.
Outputs: $STOKASTISK_OUT/03_single_gpu.json, 03_single_gpu.log (complete console), plus detail JSON files."""
import glob, os, re, shutil, subprocess, sys, traceback
from _common import OUT, check, finish, require_installed, save, sha256, warn

LOG = open(OUT / "03_single_gpu.log", "w", buffering=1)


class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x): [s.write(x) for s in self.s]; return len(x)
    def flush(self): [s.flush() for s in self.s]


sys.stdout = _Tee(sys.__stdout__, LOG)
sys.stderr = _Tee(sys.__stderr__, LOG)

R = []
EXTRA = {"script": os.path.abspath(__file__), "script_sha256": sha256(__file__)}
print("script:", EXTRA["script"], EXTRA["script_sha256"])
require_installed("03_single_gpu")        # exits with a FAIL json + hint if stokastisk/vllm are not installed in THIS session


def stage(name):
    """Decorator: run a stage; any exception becomes a FAIL with full traceback; later stages still run."""
    def deco(fn):
        try:
            fn()
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            tb = traceback.format_exc()
            print(tb)
            check(R, f"{name}: unhandled exception", False, f"{type(e).__name__}: {e}")
            EXTRA.setdefault("tracebacks", {})[name] = tb
        return fn
    return deco


import importlib.metadata as md
import torch, stokastisk, vllm


# ---------------------------------------------------------------- A. SDK and vLLM imports
@stage("A imports")
def _a():
    info = {"stokastisk": {"version": stokastisk.__version__, "file": stokastisk.__file__},
            "vllm": {"version": vllm.__version__, "file": vllm.__file__},
            "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    for dist in ("stokastisk", "vllm", "torch", "triton", "transformers"):
        try: info.setdefault("dists", {})[dist] = md.version(dist)
        except md.PackageNotFoundError: info.setdefault("dists", {})[dist] = None
    import importlib.util
    info["velocita_importable"] = importlib.util.find_spec("velocita") is not None
    EXTRA["imports"] = info
    print(info)
    check(R, "stokastisk 0.3.0 / vllm 0.25.1 versions", stokastisk.__version__ == "0.3.0" and vllm.__version__ == "0.25.1",
          f"{stokastisk.__version__} / {vllm.__version__}")
    check(R, "obsolete 'velocita' package is not importable", not info["velocita_importable"])


# ---------------------------------------------------------------- B. GPU discovery
GPUS = []


@stage("B gpu discovery")
def _b():
    ok = torch.cuda.is_available()
    check(R, "CUDA available", ok)
    n = torch.cuda.device_count()
    for d in range(n):
        p = torch.cuda.get_device_properties(d)
        free, total = torch.cuda.mem_get_info(d)
        g = {"index": d, "name": p.name, "cc": (p.major, p.minor), "free": free, "total": total}
        GPUS.append(g)
        check(R, f"gpu{d} is Tesla T4 SM75", (p.major, p.minor) == (7, 5) and "T4" in p.name, g)
    check(R, "exactly two T4 GPUs visible", n == 2 and all(g["cc"] == (7, 5) for g in GPUS), f"count={n}")
    EXTRA["gpus"] = GPUS


# ---------------------------------------------------------------- C. native extension imports (one check each)
@stage("C native extensions")
def _c():
    ext = stokastisk.check_native_extensions()
    EXTRA["native_extensions"] = ext
    for name, status in ext.items():
        check(R, f"native extension import: {name}", status == "ok", status)
    reg = {}
    for op in ("rms_norm", "silu_and_mul"):
        reg[op] = hasattr(torch.ops._C, op)
        check(R, f"torch.ops._C.{op} is registered", reg[op])
    EXTRA["registered_ops"] = reg


# ---------------------------------------------------------------- D-G. per-GPU numerics
import triton, triton.language as tl


@triton.jit
def _add(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)


EXTRA["triton"] = {"version": triton.__version__, "cache_dir": os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")}
NG = torch.cuda.device_count()
for d in range(NG):
    dev = f"cuda:{d}"

    @stage(f"gpu{d} D pytorch matmul")
    def _d():
        torch.cuda.set_device(d)
        for dt in (torch.float32, torch.float16):
            torch.manual_seed(0)
            a = torch.randn(512, 512, dtype=dt); b = torch.randn(512, 512, dtype=dt)
            ref = a.float() @ b.float(); got = (a.to(dev) @ b.to(dev)).float().cpu(); torch.cuda.synchronize(d)
            err = (ref - got).abs().max().item()
            check(R, f"gpu{d} PyTorch {dt} matmul vs CPU", torch.allclose(ref, got, rtol=5e-2, atol=5e-1), f"max abs err {err:.4g}")

    @stage(f"gpu{d} E triton")
    def _e():
        torch.cuda.set_device(d)
        x = torch.randn(10000, device=dev); y = torch.randn(10000, device=dev); o = torch.empty_like(x)
        _add[(triton.cdiv(10000, 1024),)](x, y, o, 10000, B=1024); torch.cuda.synchronize(d)
        check(R, f"gpu{d} hand-written Triton kernel (compile+launch+sync, equals PyTorch x+y)", torch.equal(o, x + y))

    @stage(f"gpu{d} F rms_norm")
    def _f():
        torch.cuda.set_device(d)
        torch.manual_seed(1)
        w = (torch.rand(128, device=dev) + 0.5).half(); xi = torch.randn(4, 128, device=dev, dtype=torch.float16)
        out = torch.full_like(xi, float("nan"))               # untouched output would be NaN -> fails
        torch.ops._C.rms_norm(out, xi, w, 1e-6); torch.cuda.synchronize(d)
        xf = xi.float(); ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
        err = (out.float() - ref).abs().max().item()
        check(R, f"gpu{d} vLLM native rms_norm (CUDA={out.is_cuda})", out.is_cuda and torch.allclose(out.float(), ref, atol=1e-2, rtol=5e-3), f"max abs err {err:.4g}")

    @stage(f"gpu{d} G silu_and_mul")
    def _g():
        torch.cuda.set_device(d)
        torch.manual_seed(2)
        xs = torch.randn(4, 256, device=dev, dtype=torch.float16); outs = torch.full((4, 128), float("nan"), device=dev, dtype=torch.float16)
        torch.ops._C.silu_and_mul(outs, xs); torch.cuda.synchronize(d)
        refs = torch.nn.functional.silu(xs[:, :128].float()) * xs[:, 128:].float()
        err = (outs.float() - refs).abs().max().item()
        check(R, f"gpu{d} vLLM native silu_and_mul (CUDA={outs.is_cuda})", outs.is_cuda and torch.allclose(outs.float(), refs, atol=1e-2, rtol=5e-3), f"max abs err {err:.4g}")


# ---------------------------------------------------------------- H. prohibited dependency audit
@stage("H prohibited")
def _h():
    pr = stokastisk.prohibited_report()
    EXTRA["prohibited"] = pr
    check(R, "no FlashAttention/FlashInfer distributions, modules or binaries", pr["clean"], pr)


# ---------------------------------------------------------------- I. ELF + CUDA architecture audit (recursive)
@stage("I elf audit")
def _i():
    root = os.path.dirname(vllm.__file__)
    readelf = shutil.which("readelf")
    cuobjdump = shutil.which("cuobjdump") or next(iter(glob.glob("/usr/local/cuda*/bin/cuobjdump")), None)
    check(R, "readelf available (an unavailable readelf is NOT a clean audit)", readelf is not None, readelf or "missing")
    if cuobjdump is None:
        warn(R, "cuobjdump unavailable: SM75 cubin inventory NOT verified on this machine", "static sm_75 evidence is in the release's CUDA-ARCH-MANIFEST.json")
    sos = sorted(glob.glob(os.path.join(root, "**", "*.so"), recursive=True) + glob.glob(os.path.join(root, "**", "*.so.*"), recursive=True))
    check(R, "vLLM native extensions found (recursive)", len(sos) > 0, len(sos))
    audit = []
    for so in sos:
        rel = os.path.relpath(so, root)
        rec = {"path": so, "rel": rel, "sha256": sha256(so)}
        if readelf:
            h = subprocess.run([readelf, "-h", so], capture_output=True, text=True)
            dyn = subprocess.run([readelf, "-d", so], capture_output=True, text=True)
            rec["readelf_h_rc"], rec["readelf_d_rc"] = h.returncode, dyn.returncode
            ok = h.returncode == 0 and dyn.returncode == 0 and "ELF64" in h.stdout
            needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dyn.stdout)
            rec["needed"] = needed
            rec["rpath_runpath"] = re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library \w+: \[([^\]]+)\]", dyn.stdout)
            rec["cuda_deps"] = [n for n in needed if re.match(r"lib(cudart|cublas|cublasLt|cudnn|nccl|cufft|curand|cusparse|cusolver|nvrtc)", n)]
            rec["cuda12"] = [n for n in needed if re.search(r"\.so\.12\b", n)]
            check(R, f"{rel}: readelf parsed (rc {h.returncode}/{dyn.returncode})", ok, dyn.stderr.strip() or "ok")
            check(R, f"{rel}: no CUDA 12 NEEDED", ok and not rec["cuda12"], rec["cuda12"] or "none")
        if cuobjdump and "stable_libtorch" in rel:
            e = subprocess.run([cuobjdump, "--list-elf", so], capture_output=True, text=True)
            p = subprocess.run([cuobjdump, "--list-ptx", so], capture_output=True, text=True)
            archs = sorted(set(re.findall(r"sm_\d+", e.stdout)))
            rec.update(cubin_archs=archs, cubin_count=len(re.findall(r"sm_\d+", e.stdout)), ptx_archs=sorted(set(re.findall(r"sm_\d+|compute_\d+", p.stdout))), cuobjdump_rc=e.returncode)
            check(R, f"{rel}: cuobjdump ok and cubins are exactly sm_75", e.returncode == 0 and archs == ["sm_75"], f"{archs} x{rec['cubin_count']} ptx={rec['ptx_archs']}")
        audit.append(rec)
    save("03_elf_audit.json", {"cuobjdump": cuobjdump, "readelf": readelf, "files": audit})
    EXTRA["elf_audit_file"] = str(OUT / "03_elf_audit.json")


save("03_native_extensions.json", EXTRA.get("native_extensions", {}))
finish("03_single_gpu", R, EXTRA)
