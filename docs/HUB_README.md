---
license: apache-2.0
library_name: stokastisk
tags: [vllm, kaggle, tesla-t4, sm75, cuda-13, triton, tensor-parallel, inference, python-3.13]
---
# Stokastisk 0.3.0 — vLLM 0.25.1 for Kaggle's 2 × Tesla T4 (SM75)

A Python 3.13 SDK plus a **T4-only vLLM 0.25.1 build with standalone CUDA 13.0 binaries**. It installs next to Kaggle's own PyTorch 2.11.0+cu130 and never replaces it.
Source, docs and tooling: <https://github.com/AI-Biruni/stokastisk>

| | |
|---|---|
| Release | `stokastisk-0.3.0-kaggle-t4-sm75-rc6` |
| Python / PyTorch / Triton | 3.13 / 2.11.0+cu130 (Kaggle's) / 3.6.0 (Kaggle's) |
| CUDA | Toolkit 13.0 only — binaries link `libcudart.so.13`; `sm_75` cubins, no PTX |
| Attention | `TRITON_ATTN` only — no FlashAttention / FlashInfer / FlashMLA code |
| Hardware | Kaggle, 2 × Tesla T4 (SM75, ~15 GiB each) |

## Status: verified on Kaggle
The project owner ran this exact release on Kaggle (2 × T4). **Notebooks 01–06 all PASS**: environment audit, install, single-GPU, dual-GPU NCCL, real inference at TP=1 and TP=2, and a TP benchmark.

| Measured (Qwen2.5-1.5B-Instruct, 4096 new tokens) | TP=1 | TP=2 |
|---|---|---|
| Decode throughput | **949 tok/s** | 521 tok/s |
| Prefill (9648 tokens) | 97.5k tok/s | 41.0k tok/s |

**Use TP=1 whenever the model fits one T4.** The two GPUs connect through the host bridge with no peer-to-peer, so TP=2 is slower (0.55×). Use it only for models that do not fit in 15 GiB.
Caveats: the Kaggle runs attached the archive as a dataset (this CDN path was verified locally with SHA256, not yet on Kaggle). Only Qwen2.5-0.5B/1.5B-Instruct were tested.
Kaggle's own image resolves NCCL to the cu12 build 2.32.3 (two NCCL packages share one file); collectives pass, and our binaries do not link it. Details in the GitHub `RELEASE.md`.

## Kaggle hard limits the SDK is built for (`stokastisk.LIMITS`)
4 CPU cores (thread budget split across TP workers) · ~30 GiB RAM · 20 GiB `/kaggle/working` · 2 × T4 (SM75, ~15 GiB VRAM each).

## Fast install (Hub CDN)
Kaggle: Internet ON, Accelerator *GPU T4 × 2*. `fetch_release.py` is stdlib-only: one ~37 MB archive, 4 parallel range requests over the CDN, verified against its `.sha256`.
```python
import urllib.request, runpy, subprocess, sys
open("/tmp/fetch_release.py", "wb").write(urllib.request.urlopen("https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py").read())
bundle = runpy.run_path("/tmp/fetch_release.py")["fetch_release"]("stokastisk-0.3.0-kaggle-t4-sm75-rc6")
subprocess.run([sys.executable, f"{bundle}/scripts/kaggle/02_install.py", str(bundle)], check=True)   # --no-deps, protected packages untouched
```
Then, in a script (vLLM uses `spawn`, so keep the `__main__` guard):
```python
import stokastisk
def main():
    llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(tensor_parallel_size=1))
    print(llm.complete("The capital of France is"), llm.verify_attention_backend()); llm.close()
if __name__ == "__main__":
    main()
```

## Files
| Path | What |
|---|---|
| `latest.json` | machine-readable pointer: release name, status, sizes, SHA256 |
| `scripts/fetch_release.py` | CDN downloader (stdlib) |
| `releases/<name>/<name>-kaggle-input.zip` (+ `.sha256`) | the complete release (installer, scripts, wheels, requirements, docs) — also usable as a Kaggle dataset |
| `releases/<name>/wheels/` | `stokastisk-0.3.0-py3-none-any.whl` and `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` for direct access |
| `releases/<name>/manifests/` | `SHA256SUMS` and `RELEASE-MANIFEST.json` |

The documentation inside the archive was written before the Kaggle run and still says "release candidate"; this card and `latest.json` carry the verified status.

## License
Apache-2.0. vLLM is Apache-2.0 © the vLLM project.
