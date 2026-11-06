#!/usr/bin/env python3.13
"""Assemble dist/stokastisk-0.3.0 bundle + manifests + tarball from dist/_stage wheels."""
import hashlib, json, os, pathlib, shutil, subprocess, sys, tarfile, time, zipfile, re
ROOT = pathlib.Path(__file__).resolve().parents[2]
B = ROOT / "dist" / "stokastisk-0.3.0"; ST = ROOT / "dist" / "_stage"
if B.exists(): shutil.rmtree(B)
for d in ("wheels", "manifests", "LICENSES", "scripts", "notebooks", "docs", "cuda"): (B / d).mkdir(parents=True)
for w in ST.glob("*.whl"): shutil.copy2(w, B / "wheels")
shutil.copytree(ROOT / "scripts" / "kaggle", B / "scripts" / "kaggle", ignore=shutil.ignore_patterns("__pycache__"))
for f in ("make_notebooks.py",): shutil.copy2(ROOT / "scripts" / f, B / "scripts")
shutil.copytree(ROOT / "scripts" / "release", B / "scripts" / "release", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy2(ROOT / "scripts" / "build_vllm_sm75.sh", B / "scripts")
for n in (ROOT / "notebooks").glob("*.ipynb"): shutil.copy2(n, B / "notebooks")
for n in (ROOT / "docs").glob("*.md"): shutil.copy2(n, B / "docs")
for f in ("README.md", "INSTALL.md", "BUILD.md"): shutil.copy2(ROOT / f, B)
shutil.copy2(ROOT / "LICENSE", B / "LICENSES" / "vllm-Apache-2.0.txt")
vz = zipfile.ZipFile(next(B.glob("wheels/vllm-*.whl")))
for n in vz.namelist():
    if "dist-info/licenses/" in n and not n.endswith("/"): (B / "LICENSES" / ("vllm-wheel-" + pathlib.Path(n).name)).write_bytes(vz.read(n))
(B / "cuda" / "README.md").write_text("No CUDA libraries are bundled by design: libcudart.so.13, libtorch*, libc10* come from the target (Kaggle torch 2.11.0+cu130); libcuda.so.1 from the NVIDIA driver.\nNative extensions live inside the vLLM wheel; see manifests/ELF-DEPENDENCY-MANIFEST.json.\n")
(B / "RELEASE.md").write_text("# Stokastisk 0.3.0\nStatus: **EXPERIMENTAL — KAGGLE DUAL-T4 VALIDATION PENDING**\nKaggle dual-T4 runtime acceptance: PENDING USER EXECUTION. Static/packaging checks only; see docs/ARCHITECTURE.md and manifests/.\nvLLM 0.25.1 (Apache-2.0) wheel derived from a local SM75/CUDA13.0 build with FlashAttention binaries removed.\n")
# manifests from the audit of the repacked wheel
wx = pathlib.Path(sys.argv[1])  # extracted repacked vllm wheel
subprocess.check_call([sys.executable, str(ROOT / "scripts/release/audit_elf.py"), str(wx), str(B / "manifests") + "/"], stdout=subprocess.DEVNULL)
zz = zipfile.ZipFile(next(B.glob("wheels/vllm-*.whl")))
meta = zz.read("vllm-0.25.1.dist-info/METADATA").decode()
reqs = [l[15:] for l in meta.splitlines() if l.startswith("Requires-Dist: ")]
PROT = ("torch", "torchvision", "torchaudio", "triton", "numpy")
dep = [{"requirement": r, "action": "keep Kaggle's version (never reinstalled; pinned by constraint)" if re.match(r"(%s)\b" % "|".join(PROT), r) else
        "02_install.py installs only if missing/unsatisfied", "kaggle_supplies": "unknown (Kaggle package list not available to this build)", "bundled": False} for r in reqs]
json.dump({"vllm_requires_dist": dep, "note": "Kaggle audit supplied only torch 2.11.0+cu130, triton 3.6.0, numpy 2.1.3, NCCL 2.28.9; others unverified"}, open(B / "manifests/DEPENDENCY-MANIFEST.json", "w"), indent=1)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
log = ROOT / "logs" / "build-vllm-sm75.log"
orig = ROOT / "work/build/vllm-wheel/vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
json.dump({"source": {"tag": "v0.25.1", "commit": "752a3a504485790a2e8491cacbb35c137339ad34", "tarball_sha256": sha(ROOT / "work/downloads/vllm-v0.25.1.tar.gz"), "commit_verified_via": "tarball pax header + GitHub tag API"},
  "patches": [], "derivation": "FlashAttention .so removed by repack_vllm_wheel.py", "original_wheel": {"name": orig.name, "size": orig.stat().st_size, "sha256": sha(orig)},
  "toolchain": {"python": "3.13.15", "torch": "2.11.0+cu130", "nvcc": "13.0.48", "gcc": "13.3.0", "cmake": "4.4.4", "ninja": "1.13.2", "kaggle_nvcc": "13.0.88"},
  "env": {"TORCH_CUDA_ARCH_LIST": "7.5", "CMAKE_CUDA_ARCHITECTURES": "75", "MAX_JOBS": "1", "VLLM_TARGET_DEVICE": "cuda"},
  "command": "setup.py bdist_wheel --py-limited-api=cp38", "log_sha256": sha(log), "build_finished_mtime": time.strftime("%F %T", time.localtime(orig.stat().st_mtime)),
  "build_exit_code": "not recorded in log; wheel assembled", "known_failures": ["vllm-server (Rust frontend): protoc not found"]},
  open(B / "manifests/BUILD-MANIFEST.json", "w"), indent=1)
json.dump({"release": "stokastisk-0.3.0", "status": "EXPERIMENTAL — KAGGLE DUAL-T4 VALIDATION PENDING", "kaggle_acceptance": "PENDING USER EXECUTION",
  "wheels": {w.name: {"size": w.stat().st_size, "sha256": sha(w)} for w in sorted((B / "wheels").glob("*.whl"))}}, open(B / "manifests/RELEASE-MANIFEST.json", "w"), indent=1)
files = sorted(p for p in B.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
(B / "manifests/SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(B)}\n" for p in files))
tgz = ROOT / "dist" / "stokastisk-0.3.0.tar.gz"
with tarfile.open(tgz, "w:gz") as t: t.add(B, arcname="stokastisk-0.3.0")
(ROOT / "dist" / "stokastisk-0.3.0.tar.gz.sha256").write_text(f"{sha(tgz)}  {tgz.name}\n")
print(tgz.stat().st_size, sha(tgz))
