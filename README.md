# Stokastisk 0.3.0 — vLLM 0.25.1 on Kaggle's 2 × Tesla T4 (SM75)

Python 3.13 SDK (`import stokastisk`) around a T4-only vLLM 0.25.1 build: CUDA 13.0, PyTorch 2.11.0+cu130 (Kaggle's own), Triton 3.6.0,
**Triton attention only** (no FlashAttention / FlashInfer / FlashMLA native code). Native extensions carry `sm_75` cubins only.

**Status: release candidate. Production-ready only after notebooks 01–06 pass on Kaggle with the exact release.** See `RELEASE.md` for what has and has not been verified.

## Kaggle hard limits this SDK is built for (`stokastisk.LIMITS`)
| Resource | Limit |
|---|---|
| CPU | 4 cores (`nproc`); the budget is split across TP workers (`OMP_NUM_THREADS` etc.) |
| System RAM | ~30 GiB |
| `/kaggle/working` | 20 GiB (evidence directories are checked against it; models/wheels go to `/tmp` and the HF cache) |
| GPUs | 2 × Tesla T4, SM75, ~15 GiB VRAM each, PHB topology, no peer access |

## Install (fast, CDN-backed)
Internet ON, accelerator *GPU T4 × 2*. One ~37 MB archive from the Hugging Face CDN (4 parallel range requests, SHA256-verified), then the
safe installer (`--no-deps`, protected Kaggle packages untouched, every file checked against `SHA256SUMS` first):
```python
import urllib.request, runpy
url = "https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py"
open("/tmp/fetch_release.py", "wb").write(urllib.request.urlopen(url).read())
bundle = runpy.run_path("/tmp/fetch_release.py")["fetch_release"]("stokastisk-0.3.0-kaggle-t4-sm75-rc6")
import subprocess, sys; subprocess.run([sys.executable, f"{bundle}/scripts/kaggle/02_install.py", str(bundle)], check=True)
```
The six notebooks in `notebooks/` do exactly this on their own when the release dataset is not attached.

```python
import stokastisk
llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(tensor_parallel_size=2))
print(llm.complete("The capital of France is"))
```
Docs: `INSTALL.md`, `BUILD.md`, `docs/ARCHITECTURE.md`, `docs/DEPENDENCIES.md`.
