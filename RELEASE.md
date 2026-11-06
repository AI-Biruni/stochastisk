# Release status
STATICALLY VERIFIED (wheels, ELF/cubins, FlashAttention/FlashInfer absence, dependency profile) · LOCALLY TESTED (SDK 20 tests, installer logic 16 tests, import sweep on CPU) · **KAGGLE TEST PENDING** (all six notebooks) → RELEASE CANDIDATE.
`KAGGLE DUAL-T4 RUNTIME ACCEPTANCE: PENDING USER EXECUTION`

## rc2 (supersedes rc1)
- **Kaggle result for rc1:** notebooks 01 and 02 passed; notebook 03 failed with `ModuleNotFoundError: No module named 'stokastisk'` because every Kaggle notebook runs in a *fresh session* and notebook 02's installation does not persist.
- **rc2:** notebooks 03–06 first run the idempotent, SHA256-verified installer (Internet must be ON), then print/verify script and wheel hashes. `03_single_gpu.py` was rewritten with isolated stages, full tracebacks, per-extension / per-GPU checks and a recursive ELF audit.
- Wheels and native binaries are byte-identical to rc1 (no native rebuild; rc1 audits carried over).
- rc2 Kaggle acceptance: **03–06 PENDING USER EXECUTION**.
