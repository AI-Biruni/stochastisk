# Release status
STATICALLY VERIFIED (wheels, ELF/cubins, FlashAttention/FlashInfer absence, dependency profile) · LOCALLY TESTED (SDK 20 tests, installer logic 16 tests, import sweep on CPU) · **KAGGLE TEST PENDING** (all six notebooks) → RELEASE CANDIDATE.
`KAGGLE DUAL-T4 RUNTIME ACCEPTANCE: PENDING USER EXECUTION`

## rc3 (supersedes rc2)
- **Kaggle evidence for rc1-dataset content:** notebook 03 passed 20/20 (old script); notebook 04 ran NCCL all-reduce/all-gather/reduce-scatter correctly on both T4s but its summary FAILED. Root cause: the shipped `stokastisk` wheel (rc1 *and* rc2) contained a stale `nccl.py` lacking the teardown keys the summary checks. See `reports/rc2-notebook-04-forensic-diagnosis.md`.
- **rc3:** `stokastisk` wheel rebuilt from `sdk/` and verified byte-equal to it (`tests/test_wheel_matches_source.py`); notebook 04 rewritten (launcher with timeout, no forced NCCL transport overrides, broadcast + barrier, teardown evidence); notebooks regenerated.
- `velocita` (`stage.zip: stage/velocita/__init__.py`) is an unused leftover: not read by the builder, in no wheel, not importable after install (`evidence/stage-package-namespace-audit.json`).
- vLLM wheel and native binaries are byte-identical to rc1/rc2 (no native rebuild; audits carried over).
- rc3 Kaggle acceptance: **03–06 must be re-run by the user; 05–06 never executed.**

## rc2 (supersedes rc1)
- **Kaggle result for rc1:** notebooks 01 and 02 passed; notebook 03 failed with `ModuleNotFoundError: No module named 'stokastisk'` because every Kaggle notebook runs in a *fresh session* and notebook 02's installation does not persist.
- **rc2:** notebooks 03–06 first run the idempotent, SHA256-verified installer (Internet must be ON), then print/verify script and wheel hashes. `03_single_gpu.py` was rewritten with isolated stages, full tracebacks, per-extension / per-GPU checks and a recursive ELF audit.
- Wheels and native binaries are byte-identical to rc1 (no native rebuild; rc1 audits carried over).
- rc2 Kaggle acceptance: **03–06 PENDING USER EXECUTION**.

## rc5 (supersedes rc4) — NOT production-ready until Kaggle notebooks 05 and 06 pass
- **Kaggle notebook 05, rc4 run** (after the spawn guard worked): TP=1 failed in `EngineCore` with `ModuleNotFoundError: vllm.v1.attention.backends.flashinfer`
  (sampler); TP=2 failed in both Workers with `ModuleNotFoundError: vllm.distributed.device_communicators.flashinfer_all_reduce`.
  Both are default-path imports of modules this Flash-free build does not ship. TRITON_ATTN selection and model loading had succeeded (TP=1).
- **Fix:** `patches/0003-t4-flashinfer-free-default-path.patch` (pure Python; native objects byte-identical to rc4); SDK pins
  `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ALLREDUCE_USE_FLASHINFER=0`.
- **NCCL note:** Kaggle's image has both `nvidia-nccl-cu13 2.28.9` and `nvidia-nccl-cu12 2.32.3`; both wheels install `nvidia/nccl/lib/libnccl.so.2`, the path torch's
  RUNPATH resolves, so they overwrite each other. NCCL's own log reports `2.32.3+cuda12.9` as the loaded library. This is a property of the Kaggle baseline
  (present before our installer runs), not of our wheels. Collectives pass. The notebooks now record the mapped library paths.
- The 25 `flash*`/`flashinfer*` files in the vLLM wheel are upstream Python adapter modules; no FlashAttention/FlashInfer/FlashMLA native code is shipped.
