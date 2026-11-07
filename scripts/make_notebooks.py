#!/usr/bin/env python3.13
"""Generate the six Kaggle notebooks. Cells only call the bundled scripts; nothing is executed here."""
import json, pathlib
OUT = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
RELEASE = "stokastisk-0.3.0-kaggle-t4-sm75-rc5"      # the ONE release these notebooks are generated for (checked by tests against the assembler)
FIND = '''import glob, json, os, pathlib
# Explicit release identity: this notebook accepts ONLY the bundle directory named RELEASE whose RELEASE-MANIFEST.json says the same name.
# (Several Kaggle datasets may be attached; an older rc is never silently selected.) Override the search root with STOKASTISK_BUNDLE.
RELEASE = "@@RELEASE@@"
cands = [pathlib.Path(p).parent.parent for p in glob.glob("/kaggle/input/**/manifests/SHA256SUMS", recursive=True)]
if os.environ.get("STOKASTISK_BUNDLE"): cands = [pathlib.Path(os.environ["STOKASTISK_BUNDLE"])]
match = [c for c in cands if c.name == RELEASE]
assert match, f"release {RELEASE!r} not found. Attached bundles: {sorted(str(c) for c in cands)}. Attach the {RELEASE} dataset."
assert len({(c / "manifests" / "SHA256SUMS").read_bytes() for c in match}) == 1, f"several DIFFERENT copies of {RELEASE}: {match}"
BUNDLE = match[0]
_m = json.loads((BUNDLE / "manifests" / "RELEASE-MANIFEST.json").read_text())
assert _m.get("name") == RELEASE, f"manifest name {_m.get('name')!r} != expected {RELEASE!r}"
SCRIPTS = BUNDLE / "scripts" / "kaggle"
os.environ["STOKASTISK_OUT"] = "/kaggle/working/stokastisk-acceptance"
print("release:", RELEASE, "| bundle:", BUNDLE, "| other attached bundles ignored:", sorted(str(c) for c in cands if c != BUNDLE))'''.replace("@@RELEASE@@", RELEASE)
# Kaggle runs EVERY notebook in a fresh session: what notebook 02 installed does not exist in 03-06 (root cause of the
# notebook-03 'ModuleNotFoundError: No module named stokastisk'). 03-06 therefore re-run the idempotent installer first
# (SHA256-verified, --no-deps, ~0 pip calls when satisfied) and then print provenance; both fail loudly.
INSTALL = '''import subprocess, sys, os
r = subprocess.run([sys.executable, str(SCRIPTS / "02_install.py"), str(BUNDLE)],
                   env=dict(os.environ, STOKASTISK_OUT="/kaggle/working/stokastisk-acceptance/prereq-install"))
assert r.returncode == 0, "prerequisite install failed (see output above); do not continue"'''
PROVENANCE = '''import hashlib, glob
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print("bundle root     :", BUNDLE)
print("scripts/kaggle  :", SCRIPTS)
for s in sorted(SCRIPTS.glob("*.py")): print(f"{sha(s)[:16]}  {s.name}")
for w in sorted((BUNDLE / "wheels").glob("*.whl")): print(f"{sha(w)}  {w.name}")
expected = {l.split(None, 1)[1].strip(): l.split()[0] for l in (BUNDLE / "manifests" / "SHA256SUMS").read_text().splitlines()}
for rel in [f"scripts/kaggle/{s.name}" for s in SCRIPTS.glob("*.py")] + [f"wheels/{w.name}" for w in (BUNDLE / "wheels").glob("*.whl")]:
    assert expected.get(rel) == sha(BUNDLE / rel), f"{rel} does not match manifests/SHA256SUMS (stale or modified file)"
print("all executed scripts and wheels match manifests/SHA256SUMS")'''

# Every notebook >= 03 must ALSO prove the installed SDK is byte-identical to the bundle's wheel (the rc1/rc2 stale-wheel
# defect: the wheel's stokastisk/nccl.py was older than the source, so notebook 04 judged fields the SDK never wrote).
VERIFY = '''import importlib, importlib.metadata as md, zipfile, hashlib
for w in sorted((BUNDLE / "wheels").glob("*.whl")):
    pkg = w.name.split("-")[0]
    mod = importlib.import_module(pkg)
    root = pathlib.Path(mod.__file__).resolve().parent
    z = zipfile.ZipFile(w); bad = []
    for n in z.namelist():
        if n.startswith(pkg + "/") and n.endswith((".py", ".so")):
            f = root.parent / n
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != hashlib.sha256(z.read(n)).hexdigest(): bad.append(n)
    print(f"{pkg} {md.version(pkg)} imported from {root}: {'identical to ' + w.name if not bad else 'MISMATCH ' + str(bad)}")
    assert not bad, f"installed {pkg} differs from the bundle wheel {w.name}; uninstall it and re-run the install cell"'''
ZIP = '''import shutil
print(shutil.make_archive("/kaggle/working/{nn}-stokastisk-acceptance", "zip", "/kaggle/working", "stokastisk-acceptance"))'''
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip().splitlines(True)}
HEAD = "**EXPERIMENTAL — KAGGLE DUAL-T4 VALIDATION PENDING.** Not executed by the author; outputs are produced only when *you* run it.\n\nAccelerator: **GPU T4 x2**, Internet **ON** (notebooks 02–06 download the 8 pinned helper wheels, ~57 MB, and 05/06 download the model). Run notebooks in order 01 → 06 (each Kaggle notebook is a fresh session, so notebooks 03–06 re-run the idempotent installer first); each stage writes JSON to `/kaggle/working/stokastisk-acceptance/`."
BODY01 = 'Prereq: release dataset attached. Success: every check prints PASS (Python 3.13, torch 2.11.0+cu130, Triton 3.6, glibc>=2.38, 2x Tesla T4 SM75). Peer access/topology are recorded, not assumed.'
BODY02 = 'Verifies SHA256SUMS **before** installing; plans from a before-snapshot and aborts untouched on any conflict; installs both wheels `--no-deps`, then only the 8 exact pins in `requirements/kaggle-t4-core.txt` (also `--no-deps`; **no resolver ever runs**). Success: all PASS, every preinstalled package unchanged (torch, Triton, Transformers 5.x, CUDA 13.0 packages), no FlashAttention/FlashInfer, imports + `TRITON_ATTN` class resolve. Re-running changes nothing. Use `--dry-run` to see the plan only.'
BODY03 = "On each GPU: PyTorch fp32/fp16 matmul vs CPU (PyTorch, *not* vLLM code), a hand-written Triton kernel (Triton JIT, *not* vLLM's attention), and two real vLLM native SM75 kernels (`rms_norm`, `silu_and_mul`). Also inspects every installed vLLM `.so` (no CUDA 12 NEEDED, cubins exactly sm_75). vLLM's Triton attention kernel is only exercised in notebook 05."
BODY04 = 'Real `torchrun` 2-rank all-reduce / all-gather / reduce-scatter with process-group teardown. NCCL_DEBUG logs are saved so the actual transport (SHM vs sockets) can be read; nothing here assumes P2P.'
BODY05 = "Loads `Qwen/Qwen2.5-0.5B-Instruct` (public, ~1 GB fp16) from Hugging Face, runs short-decode and ~1.2k-token prefill generations at TP=1 and TP=2, checks determinism, and **greps vLLM's own log** for the selected backend. Set `HF_TOKEN` as a Kaggle secret only for gated models."
BODY06 = 'Same model, prompts and sampling for both runs (one process per TP). Reports measured decode tokens/s, prefill tokens/s, latency and GPU memory, and re-checks the `TRITON_ATTN` engine config; **no speed-up is assumed** (the T4s have no P2P). Ends with the overall acceptance verdict.'
L05_RUN = 'RC = {}\nenv = dict(os.environ, PYTHONPATH=str(SCRIPTS) + ":" + os.environ.get("PYTHONPATH", ""), VLLM_LOGGING_LEVEL="INFO")\nos.makedirs("/kaggle/working/stokastisk-acceptance", exist_ok=True)\nfor tp in (1, 2):\n    with open(f"/kaggle/working/stokastisk-acceptance/vllm_tp{tp}.log", "w") as f:\n        RC[f"05_inference_tp{tp}"] = subprocess.run([sys.executable, str(SCRIPTS / "05_inference.py"), str(tp), "Qwen/Qwen2.5-0.5B-Instruct"],\n                                                   cwd=SCRIPTS, env=env, stdout=f, stderr=subprocess.STDOUT).returncode\n    print("TP", tp, "rc", RC[f"05_inference_tp{tp}"], "-- log: vllm_tp%d.log (a nonzero rc is judged in the verdict cell, after the evidence is packaged)" % tp)\n'.rstrip()
L05_CHK = 'RC["05_backend_evidence"] = run_stage("05_backend_evidence.py", "1", "2")\nfor tp in (1, 2):\n    f = pathlib.Path(f"/kaggle/working/stokastisk-acceptance/05_inference_tp{tp}.json")\n    print(f"--- {f.name}:", (f.read_text()[:1500] if f.is_file() else "NOT WRITTEN"))\n'.rstrip()
L06_RUN = 'RC = {}\nenv = dict(os.environ, PYTHONPATH=str(SCRIPTS) + ":" + os.environ.get("PYTHONPATH", ""))\nfor tp in (1, 2):\n    RC[f"06_bench_tp{tp}"] = subprocess.run([sys.executable, str(SCRIPTS / "06_tp_benchmark.py"), str(tp), "Qwen/Qwen2.5-1.5B-Instruct"], cwd=SCRIPTS, env=env).returncode\n    print("TP", tp, "rc", RC[f"06_bench_tp{tp}"])\n'.rstrip()

RUNPY = '''import subprocess, sys, os
def run_stage(script, *args):
    """Run a bundled stage script; returns its exit code (a failure is asserted in the final verdict cell, after evidence is packaged)."""
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=SCRIPTS,
                       env=dict(os.environ, PYTHONPATH=str(SCRIPTS) + ":" + os.environ.get("PYTHONPATH", "")))
    print(f"{script} exit code: {r.returncode}")
    return r.returncode'''
def sec(t): return md(t)
def setup(install=True):
    c = [sec("## 1. Setup — locate the release bundle"), code(FIND)]
    if install:
        c += [sec("## 2. Install into this fresh Kaggle session (idempotent, `--no-deps`, SHA256-verified, protected packages untouched)\n\nKaggle gives every notebook a new session: what notebook 02 installed does not exist here. Internet must be ON."), code(INSTALL),
              sec("## 3. Provenance — every executed script and wheel must match `manifests/SHA256SUMS`"), code(PROVENANCE),
              sec("## 4. The installed packages must be byte-identical to the bundle's wheels"), code(VERIFY)]
    return c
def verdict(rcs): return [sec("## Verdict — this cell fails the notebook if any stage failed"), code(f"rcs = {rcs}\nassert all(v == 0 for v in rcs.values()), f'FAILED stage(s): {{rcs}} -- see the logs and JSON in /kaggle/working/stokastisk-acceptance/'\nprint('ALL STAGES PASSED', rcs)")]
def pack(nn): return [sec("## Package the evidence\n\nCreates `/kaggle/working/%s-stokastisk-acceptance.zip` (download it from the Output tab)." % nn), code(ZIP.replace("{nn}", nn))]
NB = {
 "01_stokastisk_environment_audit": ("Environment audit", BODY01,
   setup(False) + [sec("## 2. Run the audit (read-only: installs and changes nothing)"), code(RUNPY), code('RC = {"01_env_audit": run_stage("01_env_audit.py")}')] + pack("01") + verdict('RC')),
 "02_stokastisk_installation": ("Verified installation", BODY02,
   setup(False) + [sec("## 2. Dry run (plan only; nothing is installed). **The installation cell below is blocked unless this passes.**"),
    code('import subprocess, sys\nr = subprocess.run([sys.executable, str(SCRIPTS / "02_install.py"), str(BUNDLE), "--dry-run"])\nassert r.returncode == 0, "dry run failed: do NOT install; read the output above"\nprint("dry run OK")'),
    sec("## 3. Install"), code('r = subprocess.run([sys.executable, str(SCRIPTS / "02_install.py"), str(BUNDLE)])\nRC = {"02_install": r.returncode}\nprint(RC)'),
    sec("## 4. The installed packages must be byte-identical to the bundle's wheels"), code(VERIFY)] + pack("02") + verdict('RC')),
 "03_stokastisk_single_gpu_validation": ("Single-GPU SM75 validation (each T4 individually)", BODY03,
   setup() + [sec("## 5. Run the single-GPU validation (every stage isolated; full tracebacks are saved)"), code(RUNPY), code('RC = {"03_single_gpu": run_stage("03_single_gpu.py")}')] + pack("03") + verdict('RC')),
 "04_stokastisk_dual_gpu_nccl_validation": ("Dual-GPU NCCL validation", BODY04,
   setup() + [sec("## 5. Launch two processes, one per T4 (finite timeouts; NCCL's default transport selection is NOT altered)"), code(RUNPY),
    code('RC = {"04_nccl_launch": run_stage("04_nccl_launch.py")}'),
    sec("## 6. Judge the result: launcher exit code, both ranks' collective results, teardown, NCCL log evidence"), code('RC["04_nccl_summary"] = run_stage("04_nccl_summary.py")')] + pack("04") + verdict('RC')),
 "05_stokastisk_vllm_inference": ("Real vLLM inference with TRITON_ATTN", BODY05,
   setup() + [sec("## 5. Run Qwen2.5-0.5B-Instruct at TP=1 and TP=2"), code(L05_RUN),
    sec("## 6. Prove from vLLM's own logs that `TRITON_ATTN` was selected"), code(RUNPY), code(L05_CHK)] + pack("05") + verdict('RC')),
 "06_stokastisk_tensor_parallel_benchmark": ("TP=1 vs TP=2 benchmark and acceptance verdict", BODY06,
   setup() + [sec("## 5. Benchmark Qwen2.5-1.5B-Instruct at TP=1 and TP=2"), code(L06_RUN), sec("## 6. Measured comparison and the overall verdict"), code(RUNPY), code('RC["07_summary"] = run_stage("07_summary.py")')] + pack("06") + verdict('RC')),
}
for name, (title, body, cells) in NB.items():
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
          "language_info": {"name": "python"}}, "cells": [md(f"# {title}\n\n{HEAD}\n\n{body}")] + cells}
    (OUT / f"{name}.json.tmp").write_text(json.dumps(nb, indent=1)); (OUT / f"{name}.json.tmp").rename(OUT / f"{name}.ipynb")
