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
