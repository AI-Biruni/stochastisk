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
 "02_stokastisk_installation": ("Verified installation", "Verifies SHA256SUMS **before** installing; plans from a before-snapshot and aborts untouched on any conflict; installs both wheels `--no-deps`, then only the 8 exact pins in `requirements/kaggle-t4-core.txt` (also `--no-deps`; **no resolver ever runs**). Success: all PASS, every preinstalled package unchanged (torch, Triton, Transformers 5.x, CUDA 13.0 packages), no FlashAttention/FlashInfer, imports + `TRITON_ATTN` class resolve. Re-running changes nothing. Use `--dry-run` to see the plan only.",
   [code(FIND), code('!python {SCRIPTS}/02_install.py {BUNDLE} --dry-run'), code('!python {SCRIPTS}/02_install.py {BUNDLE}')]),
 "03_stokastisk_single_gpu_validation": ("Single-GPU SM75 validation (each T4 individually)", "On each GPU: PyTorch fp32/fp16 matmul vs CPU (PyTorch, *not* vLLM code), a hand-written Triton kernel (Triton JIT, *not* vLLM's attention), and two real vLLM native SM75 kernels (`rms_norm`, `silu_and_mul`). Also inspects every installed vLLM `.so` (no CUDA 12 NEEDED, cubins exactly sm_75). vLLM's Triton attention kernel is only exercised in notebook 05.",
   [code(FIND), code('!cd {SCRIPTS} && python 03_single_gpu.py')]),
 "04_stokastisk_dual_gpu_nccl_validation": ("Dual-GPU NCCL validation", "Real `torchrun` 2-rank all-reduce / all-gather / reduce-scatter with process-group teardown. NCCL_DEBUG logs are saved so the actual transport (SHM vs sockets) can be read; nothing here assumes P2P.",
   [code(FIND), code('''import os
os.makedirs(os.environ["STOKASTISK_OUT"], exist_ok=True)   # NCCL_DEBUG_FILE / torchrun need the directory to exist first
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
    print("TP", tp, "rc", rc)'''), code(r'''import re, json
OUT = "/kaggle/working/stokastisk-acceptance"
ev = {}
for tp in (1, 2):
    log = open(f"{OUT}/vllm_tp{tp}.log", errors="replace").read()
    chosen = re.findall(r"Using (?:AttentionBackendEnum\.)?TRITON_ATTN backend", log)
    other = re.findall(r"(?i)using[^\n]*(?:FLASH_?ATTN|FLASHINFER|FLASHMLA|FLEX_ATTENTION|XFORMERS)[^\n]*backend", log)
    ev[tp] = {"triton_lines": len(chosen), "other_backend_lines": other[:3], "log_bytes": len(log)}
    print(f"TP{tp}: {ev[tp]}")
    assert len(chosen) >= tp, f"expected >= {tp} 'Using ...TRITON_ATTN backend' log line(s) (one per worker); inspect {OUT}/vllm_tp{tp}.log"
    assert not other, "a non-Triton attention backend was reported"
    assert "Selected backend" not in log or "is not valid" not in log, "vLLM rejected the selected backend"
json.dump(ev, open(f"{OUT}/05_backend_log_evidence.json", "w"), indent=1)'''), code('''print(open("/kaggle/working/stokastisk-acceptance/05_inference_tp1.json").read()[:1500])''')]),
 "06_stokastisk_tensor_parallel_benchmark": ("TP=1 vs TP=2 benchmark and acceptance verdict", "Same model, prompts and sampling for both runs (one process per TP). Reports measured decode tokens/s, prefill tokens/s, latency and GPU memory, and re-checks the `TRITON_ATTN` engine config; **no speed-up is assumed** (the T4s have no P2P). Ends with the overall acceptance verdict.",
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
