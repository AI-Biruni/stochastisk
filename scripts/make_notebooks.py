#!/usr/bin/env python3.13
"""Generate the six Kaggle notebooks. Cells only call the bundled scripts; nothing is executed here."""
import json, pathlib
OUT = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
FIND = '''import glob, os, pathlib
# Locate the release bundle (the Kaggle dataset holding manifests/SHA256SUMS). Override with STOKASTISK_BUNDLE.
hits = [pathlib.Path(p).parent.parent for p in glob.glob("/kaggle/input/**/manifests/SHA256SUMS", recursive=True)]
BUNDLE = pathlib.Path(os.environ.get("STOKASTISK_BUNDLE") or (hits[0] if hits else "")) 
assert (BUNDLE / "manifests" / "SHA256SUMS").is_file(), "attach the stokastisk-0.3.0 release dataset to this notebook"
SCRIPTS = BUNDLE / "scripts" / "kaggle"
os.environ["STOKASTISK_OUT"] = "/kaggle/working/stokastisk-acceptance"
print("bundle:", BUNDLE)'''
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip().splitlines(True)}
HEAD = "**EXPERIMENTAL — KAGGLE DUAL-T4 VALIDATION PENDING.** Not executed by the author; outputs are produced only when *you* run it.\n\nAccelerator: **GPU T4 x2**, Internet on only if the model must be downloaded. Run notebooks in order 01 → 06; each stage writes JSON to `/kaggle/working/stokastisk-acceptance/`."
NB = {
 "01_stokastisk_environment_audit": ("Environment audit", "Prereq: release dataset attached. Success: every check prints PASS (Python 3.13, torch 2.11.0+cu130, Triton 3.6, glibc>=2.38, 2x Tesla T4 SM75). Peer access/topology are recorded, not assumed.",
   [code(FIND), code('!cd {SCRIPTS} && python 01_env_audit.py')]),
 "02_stokastisk_installation": ("Verified installation", "Verifies SHA256SUMS **before** installing, installs both wheels with `--no-deps`, then installs only missing third-party requirements under a constraint pinning Kaggle's torch/triton/numpy. Success: all PASS, torch unchanged, no FlashAttention binaries.",
   [code(FIND), code('!python {SCRIPTS}/02_install.py {BUNDLE}')]),
 "03_stokastisk_single_gpu_validation": ("Single-GPU SM75 validation (each T4 individually)", "Runs fp32/fp16 matmuls vs a CPU reference, a Triton kernel and a vLLM `_C` kernel on each GPU, and inspects every native extension (no CUDA 12 NEEDED, sm_75 cubins).",
   [code(FIND), code('!cd {SCRIPTS} && python 03_single_gpu.py')]),
 "04_stokastisk_dual_gpu_nccl_validation": ("Dual-GPU NCCL validation", "Real `torchrun` 2-rank all-reduce / all-gather / reduce-scatter with process-group teardown. NCCL_DEBUG logs are saved so the actual transport (SHM vs sockets) can be read; nothing here assumes P2P.",
   [code(FIND), code('''import os
env = dict(os.environ, NCCL_DEBUG="INFO", NCCL_DEBUG_FILE="/kaggle/working/stokastisk-acceptance/nccl.%h.%p.log",
           PYTHONPATH=str(SCRIPTS) + ":" + os.environ.get("PYTHONPATH", ""))
import subprocess
r = subprocess.run(["torchrun", "--standalone", "--nproc-per-node=2", str(SCRIPTS / "04_nccl_dual_gpu.py")], env=env)
print("torchrun rc", r.returncode)'''), code('!cd {SCRIPTS} && python 04_nccl_summary.py')]),
 "05_stokastisk_vllm_inference": ("Real vLLM inference with TRITON_ATTN", "Loads `Qwen/Qwen2.5-0.5B-Instruct` (public, ~1 GB fp16) from Hugging Face, runs short-decode and ~1.2k-token prefill generations at TP=1 and TP=2, checks determinism, and **greps vLLM's own log** for the selected backend. Set `HF_TOKEN` as a Kaggle secret only for gated models.",
   [code(FIND), code('''import subprocess, sys, os
env = dict(os.environ, PYTHONPATH=str(SCRIPTS) + ":" + os.environ.get("PYTHONPATH", ""), VLLM_LOGGING_LEVEL="INFO")
for tp in (1, 2):
    with open(f"/kaggle/working/stokastisk-acceptance/vllm_tp{tp}.log", "w") as f:
        rc = subprocess.run([sys.executable, str(SCRIPTS / "05_inference.py"), str(tp), "Qwen/Qwen2.5-0.5B-Instruct"],
                            env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print("TP", tp, "rc", rc)'''), code('''import re
for tp in (1, 2):
    log = open(f"/kaggle/working/stokastisk-acceptance/vllm_tp{tp}.log", errors="replace").read()
    chosen = re.findall(r"Using (\\S*TRITON_ATTN\\S*) (?:attention )?backend", log)
    flash = re.findall(r"(?i)(?:using|selected)[^\\n]*flash[_ ]?attn", log)
    print(f"TP{tp}: triton lines={chosen[:2]} flash-lines={flash[:2]}")
    assert chosen, "vLLM log never reported TRITON_ATTN -- inspect the log"
    assert not flash, "FlashAttention appears in the log"
print(open("/kaggle/working/stokastisk-acceptance/05_inference_tp1.json").read()[:1500])''')]),
 "06_stokastisk_tensor_parallel_benchmark": ("TP=1 vs TP=2 benchmark and acceptance verdict", "Same model, prompts and sampling for both runs (one process per TP). Reports measured tokens/s and GPU memory; **no speed-up is assumed** (the T4s have no P2P). Ends with the overall acceptance verdict.",
   [code(FIND), code('''import subprocess, sys, os
env = dict(os.environ, PYTHONPATH=str(SCRIPTS) + ":" + os.environ.get("PYTHONPATH", ""))
for tp in (1, 2):
    rc = subprocess.run([sys.executable, str(SCRIPTS / "06_tp_benchmark.py"), str(tp), "Qwen/Qwen2.5-1.5B-Instruct"], env=env).returncode
    print("TP", tp, "rc", rc)'''), code('!cd {SCRIPTS} && python 07_summary.py')]),
}
for name, (title, body, cells) in NB.items():
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
          "language_info": {"name": "python"}}, "cells": [md(f"# {title}\n\n{HEAD}\n\n{body}")] + cells}
    (OUT / f"{name}.json.tmp").write_text(json.dumps(nb, indent=1)); (OUT / f"{name}.json.tmp").rename(OUT / f"{name}.ipynb")
