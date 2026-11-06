#!/usr/bin/env python3
"""Stage 02: verified offline install. usage: 02_install.py [BUNDLE_DIR | --wheels DIR]
Discovers the bundle (arg, $STOKASTISK_BUNDLE, or /kaggle/input/**/manifests/SHA256SUMS), verifies SHA256
BEFORE installing, installs both wheels with --no-deps, then installs only MISSING third-party requirements
of vllm under a constraint file that pins torch/triton/numpy to the Kaggle-provided versions."""
import glob, hashlib, importlib.metadata as md, json, os, pathlib, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import check, finish

PROTECT = ("torch", "torchvision", "torchaudio", "triton", "numpy", "xformers", "flashinfer", "flash-attn", "flash_attn")

def find_bundle():
    cands = [sys.argv[1]] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else []
    cands += [os.environ.get("STOKASTISK_BUNDLE", "")]
    cands += [str(pathlib.Path(p).parent.parent) for p in glob.glob("/kaggle/input/**/manifests/SHA256SUMS", recursive=True)]
    cands += [str(pathlib.Path(__file__).resolve().parents[2])]
    for c in cands:
        if c and (pathlib.Path(c) / "manifests" / "SHA256SUMS").is_file():
            return pathlib.Path(c)
    sys.exit("bundle not found: pass its directory (containing manifests/SHA256SUMS) as argument")

R = []
b = find_bundle(); print("bundle:", b)
bad = []
for line in (b / "manifests" / "SHA256SUMS").read_text().splitlines():
    h, name = line.split(None, 1); name = name.strip().lstrip("*")
    p = b / name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
        bad.append(name)
check(R, "SHA256SUMS verified", not bad, bad or "all files match")
if bad:
    finish("02_install", R)
wheels = sorted(glob.glob(str(b / "wheels" / "*.whl")))
check(R, "two wheels present", len(wheels) == 2, [os.path.basename(w) for w in wheels])
pip = [sys.executable, "-m", "pip"]
subprocess.check_call(pip + ["install", "--no-deps", "--no-index", "--force-reinstall"] + wheels)
# missing requirements (excluding protected + extras + non-applicable markers)
from packaging.requirements import Requirement
missing = []
for rs in md.requires("vllm") or []:
    r = Requirement(rs)
    if r.marker and not r.marker.evaluate({"extra": ""}):
        continue
    if r.name.lower().replace("_", "-") in PROTECT:
        continue
    try:
        v = md.version(r.name)
        if r.specifier and not r.specifier.contains(v, prereleases=True):
            missing.append(rs.split(";")[0].strip())
    except md.PackageNotFoundError:
        missing.append(rs.split(";")[0].strip())
print("missing third-party requirements:", missing)
if missing:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as c:
        for p in PROTECT:
            try: c.write(f"{p}=={md.version(p)}\n")
            except md.PackageNotFoundError: pass
    env = dict(os.environ, PIP_CONSTRAINT=c.name)
    rc = subprocess.call(pip + ["install"] + missing, env=env)
    check(R, "missing requirements installed under torch/triton/numpy constraint", rc == 0, f"pip rc={rc}")
import torch
check(R, "torch preserved", torch.__version__.startswith("2.11.0") and torch.version.cuda == "13.0", torch.__version__)
bad_imp = {}
for m in ("vllm", "stokastisk", "vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch", "vllm.cumem_allocator",
          "vllm.spinloop", "vllm.fs_io_C", "vllm.v1.attention.backends.triton_attn"):
    try: __import__(m)
    except Exception as e: bad_imp[m] = f"{type(e).__name__}: {e}"
check(R, "imports", not bad_imp, bad_imp or "ok")
import vllm, stokastisk
check(R, "versions", vllm.__version__ == "0.25.1" and stokastisk.__version__ == "0.3.0", f"vllm={vllm.__version__} stokastisk={stokastisk.__version__}")
fa = glob.glob(os.path.join(os.path.dirname(vllm.__file__), "vllm_flash_attn", "*.so"))
check(R, "no FlashAttention binaries installed", not fa, fa or "none")
finish("02_install", R)
