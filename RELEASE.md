# Release status

Current release: `stokastisk-0.3.0-kaggle-t4-sm75-rc6` — **verified on Kaggle (2 × Tesla T4): notebooks 01–06 all PASS.**

Hub: <https://huggingface.co/waqasm86/stokastisk> · archive `releases/stokastisk-0.3.0-kaggle-t4-sm75-rc6/…-kaggle-input.zip`
(SHA256 `22e46e5f427b823b2edd49962905b0e2070d0bc49ccb5c20f9947f193dc1179b`).

## Kaggle acceptance, rc6 (run by the owner; Python 3.13.16, torch 2.11.0+cu130, Triton 3.6.0, vLLM 0.25.1, transformers 5.20.2, driver 580.178.04)
| Notebook | Result | Key evidence |
|---|---|---|
| 01 environment audit | PASS | nproc 4, RAM 31.3 GiB, 2 × T4 (14.56 GiB visible), `/kaggle/working` 19.5 GiB, topology PHB |
| 02 installation | PASS | protected packages unchanged; after-state = before-state + 8 pins |
| 03 single GPU | PASS | native extensions load; ELF audit sm_75 / `libcudart.so.13` |
| 04 dual-GPU NCCL | PASS (1 WARNING) | all collectives pass; transport `SHM/direct`; WARNING: runtime NCCL 2.32.3 ≠ compiled 2.28.9 (below) |
| 05 vLLM inference | PASS | Qwen2.5-0.5B-Instruct, TP=1 and TP=2: real generation, `TRITON_ATTN` selected, Triton attention kernels compiled, GPU memory 13.9 GB peak → 3 MiB after close, no engine process left |
| 06 TP benchmark | PASS | Qwen2.5-1.5B-Instruct, 4096 new tokens: TP=1 **949.0 tok/s**, TP=2 **521.3 tok/s** (0.55×); prefill 9648 tokens: 97.5k vs 41.0k tok/s |

What is **not** proven: the Kaggle runs attached the release as a Kaggle dataset. The Hub-CDN download path (`fetch_release.py`) was verified locally (parallel ranges, SHA256) but has not been run on Kaggle.
Models other than Qwen2.5-0.5B/1.5B-Instruct were not tested.

## Known facts
* **TP=2 is slower than TP=1** for these models: the two T4s sit behind a PCIe host bridge (PHB) with no peer access. Use TP=2 only when a model does not fit one GPU.
* **NCCL runtime:** Kaggle's image contains both `nvidia-nccl-cu13 2.28.9` and `nvidia-nccl-cu12 2.32.3`; both install `nvidia/nccl/lib/libnccl.so.2` (the path torch's RUNPATH resolves), so the later one wins. vLLM logs `using nccl==2.32.3`.
  This is a property of the Kaggle baseline (present before our installer; protected), not of our wheels. `torch.cuda.nccl.version()` (2.28.9) is the compile-time version.
* **Flash-free:** no FlashAttention/FlashInfer/FlashMLA native code or distribution. The vLLM wheel keeps upstream Python adapter modules named `flash*`; they are inert. `patches/0003` guards the two default-path FlashInfer imports (sampler, TP=2 communicator).
* **Native binaries:** 6 objects, `sm_75` cubins only (no PTX), `libcudart.so.13`, GLIBC ≤ 2.38, no `.so.12`. Built once with CUDA 13.0.48 and carried over unchanged since rc1 (`scripts/audit/native_audit.py`).

## History (details in git)
rc1–rc3: stale SDK wheel fixed; notebook 04 rewritten. rc4: vLLM `spawn` re-import crash (no `__main__` guard). rc5: default-path FlashInfer imports (sampler, TP=2 communicator). rc6: shutdown measurement, Kaggle hard-limit module, HF CDN installer, single project name `stokastisk`; first release to pass 01–06 on Kaggle.
