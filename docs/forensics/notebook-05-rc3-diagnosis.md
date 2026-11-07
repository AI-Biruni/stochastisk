# Notebook 05 diagnosis (rc3 Kaggle run, 2026-11-07)

**Result:** TP=1 and TP=2 both ended `rc 1`; no `05_inference_tp*.json` was written. The notebook's failing cell (13, `AssertionError`) is only a *symptom*: no backend log line exists because the engine never started.

## Root cause (confirmed from `vllm_tp1.log` and `vllm_tp2.log`, independently)
vLLM printed `We must use the spawn multiprocessing start method ... Reasons: CUDA is initialized`. `multiprocessing` spawn **re-imports the launching script** in the engine-core child. `scripts/kaggle/05_inference.py` ran everything at module level, so the child re-executed it and hit:

```
RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
  ... if __name__ == '__main__': freeze_support()
-> stokastisk.errors.EngineError: vLLM engine initialisation failed: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
The duplicated `non-default args` block in each log is that child re-running the script. TP=2 failed identically, before any NCCL activity (0 NCCL lines).
`06_tp_benchmark.py` had the same structure and would have failed the same way.

## What was NOT the cause (checked)
Wrong/stale bundle or wheels (rc3 hashes verified on Kaggle), model download, Triton/SM75/native libs, NCCL, GPU memory, the backend regex (matches vLLM 0.25.1 `platforms/cuda.py:416`, `"Using %s backend."` of an `AttentionBackendEnum`).

## Evidence NOT available (reported as NOT_RUN, not PASS)
Model load, tokenizer init (only `Resolved architecture: Qwen2ForCausalLM`), generation, Triton backend log line, Triton cache artifacts (both dirs empty), GPU memory.

## Fix (rc4, commit 1bcf673)
Engine code under `if __name__ == "__main__"`, `run_guarded` (a crash always writes an ERROR JSON + traceback), `05_backend_evidence.py`, per-notebook verdict, verdict cell appended to notebooks 05/06, tests in `tests/test_notebook05_spawn.py` (including a `__mp_main__` re-import simulation that fails on the old scripts). **Not yet executed on Kaggle.**
