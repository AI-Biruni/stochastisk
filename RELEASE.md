# Release status

**Current candidate: `stokastisk-0.3.0-kaggle-t4-sm75-rc6`** — locally tested; **not yet executed on Kaggle**. Not production-ready until notebooks 01–06 pass on it.

## What Kaggle (2 × T4) has shown, on the previous candidate (rc5), run by the owner
| Notebook | Result |
|---|---|
| 01–03 | PASS (earlier candidates) |
| 04 NCCL | PASS (collectives, teardown); WARNING: the loaded NCCL is 2.32.3+cuda12.9 (see below) |
| 05 vLLM inference | TP=1 and TP=2: real Qwen2.5-0.5B generation completed, deterministic, `TRITON_ATTN` selected, `kernel_unified_attention` compiled, no Flash library mapped. Only the shutdown check failed (it counted a `resource_tracker` helper / zombie; GPU memory was fully released) — corrected in rc6, untested |
| 06 TP benchmark (Qwen2.5-1.5B) | PASS. Measured TP=1 **1032.9** tok/s vs TP=2 **545.1** tok/s (**0.53×**: TP=2 is slower on this PHB, no-P2P topology). Prefill 90.8k vs 54.3k tok/s |

## Known facts
* **NCCL runtime:** Kaggle's image contains both `nvidia-nccl-cu13 2.28.9` and `nvidia-nccl-cu12 2.32.3`; both install `nvidia/nccl/lib/libnccl.so.2` (the path torch's RUNPATH resolves), so they overwrite each other. vLLM logs `using nccl==2.32.3` and NCCL reports `2.32.3+cuda12.9`. Property of the Kaggle baseline (present before our installer; protected), not of our wheels. `torch.cuda.nccl.version()` (2.28.9) is the compile-time version.
* **Flash-free:** no FlashAttention/FlashInfer/FlashMLA native code or distribution. The vLLM wheel keeps upstream Python adapter modules named `flash*`; they are inert. Two default-path imports of removed FlashInfer modules were guarded by `patches/0003` (pure Python).
* **Native binaries:** 6 objects, `sm_75` cubins only (no PTX), `libcudart.so.13`, GLIBC ≤ 2.38, no `.so.12`. Built once with CUDA 13.0.48 and carried over unchanged since rc1 (`scripts/audit/native_audit.py`).
* TP=2 ≠ faster: measured above.

## History (details in git)
rc1–rc3: stale SDK wheel (`nccl.py`) fixed; notebook 04 rewritten. rc4: vLLM `spawn` re-import crash (no `__main__` guard) fixed. rc5: default-path FlashInfer imports (sampler, TP=2 communicator) fixed. rc6: shutdown measurement, Kaggle hard-limit module, HF CDN installer, repository cleanup, single project name `stokastisk`.
